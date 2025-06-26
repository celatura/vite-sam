import { useState, useRef, useEffect, type MouseEvent } from "react";
import {
  SamModel,
  AutoProcessor,
  RawImage,
  PreTrainedModel,
  Processor,
  Tensor,
  type SamImageProcessorResult,
} from "@huggingface/transformers";

interface MarkPoint {
  position: number[];
  label: number;
}

const ImageSegmentation = () => {
  const [imgUrl, setImgUrl] = useState<string>("");
  const [markPoints, setMarkPoints] = useState<MarkPoint[]>([]);
  const [statusLabel, setStatusLabel] = useState("");

  const imageContainerRef = useRef<HTMLDivElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<PreTrainedModel>(null);
  const processorRef = useRef<Processor>(null);
  const imageInputRef = useRef<RawImage>(null);
  const imageProcessed = useRef<SamImageProcessorResult>(null);
  const imageEmbeddings = useRef<Tensor>(null);
  const isEncodingRef = useRef<boolean>(false);
  const isDecodingRef = useRef<boolean>(false);
  const isMultiMaskModeRef = useRef<boolean>(false);

  useEffect(() => {
    const initModel = async () => {
      if (modelRef.current || processorRef.current) {
        return;
      }

      setStatusLabel("模型加载中...");
      const model_id = "Xenova/slimsam-77-uniform";
      modelRef.current = await SamModel.from_pretrained(model_id, {
        dtype: "fp16", // or "fp32"
        device: "webgpu",
      });
      processorRef.current = await AutoProcessor.from_pretrained(model_id, {});

      setStatusLabel("模型加载完成");
    };
    initModel();
  }, []);

  const encode = async (url: string) => {
    if (!modelRef.current || !processorRef.current || !url || isEncodingRef.current) {
      return;
    }
    isEncodingRef.current = true;
    setStatusLabel("正在提取图像嵌入...");

    imageInputRef.current = await RawImage.fromURL(url);

    // 更新界面
    setImgUrl(url);

    // 重新计算图像嵌入
    imageProcessed.current = await processorRef.current(imageInputRef.current);
    imageEmbeddings.current = await (modelRef.current as any).get_image_embeddings(imageProcessed.current);

    setStatusLabel("嵌入提取完成!");
    isEncodingRef.current = false;
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e2: ProgressEvent<FileReader>) => encode(e2.target?.result as any);
    reader.readAsDataURL(file);
    e.target.value = "";
  };

  const updateMaskOverlay = (mask: RawImage, scores: Float32Array) => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) {
      return;
    }
    const maskContext = maskCanvas.getContext("2d") as CanvasRenderingContext2D;

    // 更新画布尺寸
    if (maskCanvas.width !== mask.width || maskCanvas.height !== mask.height) {
      maskCanvas.width = mask.width;
      maskCanvas.height = mask.height;
    }

    // 为像素数据分配缓冲区
    const imageData = maskContext.createImageData(maskCanvas.width, maskCanvas.height);

    // 选择最佳的蒙版
    const numMasks = scores.length; // 3
    let bestIndex = 0;
    for (let i = 1; i < numMasks; ++i) {
      if (scores[i] > scores[bestIndex]) {
        bestIndex = i;
      }
    }

    // 用颜色填充蒙版
    const pixelData = imageData.data;
    for (let i = 0; i < pixelData.length; ++i) {
      if (mask.data[numMasks * i + bestIndex] === 1) {
        const offset = 4 * i;
        pixelData[offset] = 0; // red
        pixelData[offset + 1] = 114; // green
        pixelData[offset + 2] = 189; // blue
        pixelData[offset + 3] = 255; // alpha
      }
    }

    // 将图像数据绘制到上下文
    maskContext.putImageData(imageData, 0, 0);
  };

  const decode = async (markPoints: MarkPoint[]) => {
    if (!modelRef.current || !imageEmbeddings.current || !processorRef.current || !imageProcessed.current) {
      return;
    }

    if (isDecodingRef.current) {
      return;
    }
    isDecodingRef.current = true;

    // 准备解码的输入
    const reshaped = imageProcessed.current.reshaped_input_sizes[0];
    const points = markPoints.map((x) => [x.position[0] * reshaped[1], x.position[1] * reshaped[0]]).flat(Infinity);
    const labels = markPoints.map((x) => BigInt(x.label)).flat(Infinity);

    const num_points = markPoints.length;
    const input_points = new Tensor("float32", points, [1, 1, num_points, 2]);
    const input_labels = new Tensor("int64", labels, [1, 1, num_points]);

    // 生成蒙版
    const { pred_masks, iou_scores } = await modelRef.current({
      ...imageEmbeddings.current,
      input_points,
      input_labels,
    });

    // 对蒙版进行后处理
    const masks = await (processorRef.current as any).post_process_masks(
      pred_masks,
      imageProcessed.current.original_sizes,
      imageProcessed.current.reshaped_input_sizes,
    );

    isDecodingRef.current = false;

    updateMaskOverlay(RawImage.fromTensor(masks[0][0]), iou_scores.data);
  };

  const clamp = (x: number, min = 0, max = 1) => {
    return Math.max(Math.min(x, max), min);
  };

  // 获取鼠标相对于容器的坐标
  const getPoint = (e: MouseEvent) => {
    const bb = imageContainerRef.current!.getBoundingClientRect();

    const mouseX = clamp((e.clientX - bb.left) / bb.width);
    const mouseY = clamp((e.clientY - bb.top) / bb.height);

    return {
      position: [mouseX, mouseY],
      label:
        e.button === 2 // 右键
          ? 0 // 负标记
          : 1, // 正标记
    };
  };

  const onMouseDown = (e: MouseEvent) => {
    if (e.button !== 0 && e.button !== 2) {
      return;
    }
    if (!imageEmbeddings.current) {
      return;
    }
    if (!isMultiMaskModeRef.current) {
      isMultiMaskModeRef.current = true;
    }

    const point = getPoint(e);
    const newPoints = [...markPoints, point];

    setMarkPoints(newPoints);
    decode(newPoints);
  };

  const onMouseMove = (e: MouseEvent) => {
    if (!imageEmbeddings.current || isMultiMaskModeRef.current) {
      // 如果图像尚未编码，或者已经进行选区，则忽略鼠标移动事件
      return;
    }

    const newPoints = [getPoint(e)];
    decode(newPoints);
  };

  const clearPointsAndMask = () => {
    // 清除标记点
    isMultiMaskModeRef.current = false;
    setMarkPoints([]);

    if (maskCanvasRef.current) {
      const maskContext = maskCanvasRef.current.getContext("2d");
      maskContext!.clearRect(0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height);
      return;
    }
  };

  const handleReset = () => {
    // 重置状态
    imageInputRef.current = null;
    imageProcessed.current = null;
    imageEmbeddings.current = null;
    isEncodingRef.current = false;
    isDecodingRef.current = false;

    clearPointsAndMask();
    setImgUrl("");
    setStatusLabel("模型加载完成");
  };

  const handleCut = async () => {
    if (!maskCanvasRef.current || !imageInputRef.current) {
      return;
    }
    const maskContext = maskCanvasRef.current.getContext("2d");
    const [w, h] = [maskCanvasRef.current.width, maskCanvasRef.current.height];

    // 获取蒙版像素数据（并将其用作缓冲区）
    const maskImageData = maskContext!.getImageData(0, 0, w, h);

    // 创建新画布来保存剪切图
    const cutCanvas = new OffscreenCanvas(w, h);
    const cutContext = cutCanvas.getContext("2d");

    // 将图像像素数据复制到剪切画布
    const maskPixelData = maskImageData.data;
    const imagePixelData = imageInputRef.current.data;
    for (let i = 0; i < w * h; ++i) {
      const sourceOffset = 3 * i; // RGB
      const targetOffset = 4 * i; // RGBA

      if (maskPixelData[targetOffset + 3] > 0) {
        // 仅复制不透明像素
        for (let j = 0; j < 3; ++j) {
          maskPixelData[targetOffset + j] = imagePixelData[sourceOffset + j];
        }
      }
    }
    cutContext!.putImageData(maskImageData, 0, 0);

    // 下载图像
    const link = document.createElement("a");
    link.download = "image.png";
    link.href = URL.createObjectURL(await cutCanvas.convertToBlob());
    link.click();
    link.remove();
  };

  return (
    <div className="flex h-dvh items-center justify-center">
      <div className="flex flex-col items-center justify-center overflow-auto p-6">
        <div className="relative flex h-[420px] w-[640px] items-center justify-center rounded-lg border-2 border-dashed">
          {!imgUrl && (
            <label
              id="upload-button"
              htmlFor="upload"
              className="flex cursor-pointer flex-col items-center justify-center gap-2 text-lg"
            >
              <svg width="25" height="25" viewBox="0 0 25 25" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path
                  fill="#000"
                  d="M3.5 24.3a3 3 0 0 1-1.9-.8c-.5-.5-.8-1.2-.8-1.9V2.9c0-.7.3-1.3.8-1.9.6-.5 1.2-.7 2-.7h18.6c.7 0 1.3.2 1.9.7.5.6.7 1.2.7 2v18.6c0 .7-.2 1.4-.7 1.9a3 3 0 0 1-2 .8H3.6Zm0-2.7h18.7V2.9H3.5v18.7Zm2.7-2.7h13.3c.3 0 .5 0 .6-.3v-.7l-3.7-5a.6.6 0 0 0-.6-.2c-.2 0-.4 0-.5.3l-3.5 4.6-2.4-3.3a.6.6 0 0 0-.6-.3c-.2 0-.4.1-.5.3l-2.7 3.6c-.1.2-.2.4 0 .7.1.2.3.3.6.3Z"
                ></path>
              </svg>
              点击上传图片
            </label>
          )}
          {imgUrl && (
            <div
              ref={imageContainerRef}
              className="relative h-full w-max overflow-hidden"
              onMouseDown={onMouseDown}
              onMouseMove={onMouseMove}
              onContextMenu={(e) => e.preventDefault()}
            >
              <img className="h-full max-w-none" src={imgUrl} />

              <canvas ref={maskCanvasRef} className="absolute top-0 left-0 z-1 h-full w-full opacity-60"></canvas>

              {markPoints.map((point, index) => {
                switch (point.label) {
                  case 1:
                    return (
                      <div
                        key={index}
                        className="absolute z-2 h-4 w-4 -translate-x-2/4 -translate-y-2/4 rounded-[50%] bg-[turquoise]"
                        style={{
                          top: `${point.position[1] * 100}%`,
                          left: `${point.position[0] * 100}%`,
                        }}
                      />
                    );
                  case 0:
                    return (
                      <div
                        key={index}
                        className="absolute z-2 h-4 w-4 -translate-x-2/4 -translate-y-2/4 rounded-[50%] bg-[pink]"
                        style={{
                          top: `${point.position[1] * 100}%`,
                          left: `${point.position[0] * 100}%`,
                        }}
                      />
                    );
                  default:
                    return null;
                }
              })}
            </div>
          )}
        </div>

        <div className="my-2 text-base">{statusLabel}</div>

        <div className="flex items-center gap-3">
          <button
            className="inline-flex cursor-pointer items-center justify-center gap-2 rounded-md bg-[#1f1f1f] px-4 py-2 text-sm font-medium text-white"
            onClick={handleReset}
          >
            重置图片
          </button>
          <button
            className="inline-flex cursor-pointer items-center justify-center gap-2 rounded-md bg-[#1f1f1f] px-4 py-2 text-sm font-medium text-white"
            onClick={clearPointsAndMask}
          >
            清除标记点
          </button>
          <button
            className="inline-flex cursor-pointer items-center justify-center gap-2 rounded-md bg-[#1f1f1f] px-4 py-2 text-sm font-medium text-white"
            onClick={handleCut}
          >
            裁剪图片
          </button>
        </div>

        <div className="mt-1">鼠标左键点击正标记，右键点击负标记</div>
        <input id="upload" type="file" accept="image/*" onChange={handleFileUpload} className="hidden" />
      </div>
    </div>
  );
};

export default ImageSegmentation;
