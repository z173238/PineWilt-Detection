"""YOLO 模型加载与批量推理.

包括：设备解析、letterbox 预处理、批量预测、坐标映射。
"""

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils import nms


def resolve_device(device: str) -> torch.device:
    """解析设备参数字符串为 torch.device."""
    device = str(device).strip().lower()
    if device == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def letterbox_rgb(image: np.ndarray, size: int) -> tuple:
    """等比缩放并居中填充 RGB 图像到目标尺寸.

    Args:
        image: 输入图像 (H, W, 3) uint8.
        size: 目标正方形边长.

    Returns:
        (padded_image, gain, pad_x, pad_y)
    """
    height, width = image.shape[:2]
    gain = min(size / height, size / width)
    new_width = int(round(width * gain))
    new_height = int(round(height * gain))
    resized = Image.fromarray(image).resize((new_width, new_height), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = int(round((size - new_width) / 2 - 0.1))
    pad_y = int(round((size - new_height) / 2 - 0.1))
    canvas.paste(resized, (pad_x, pad_y))
    return np.asarray(canvas, dtype=np.uint8), gain, pad_x, pad_y


def scale_letterbox_boxes(boxes: np.ndarray, gain: float, pad_x: int, pad_y: int,
                          original_shape: tuple) -> np.ndarray:
    """将 letterbox 空间中的边界框映射回原始图像坐标."""
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes[:, :4] /= gain
    height, width = original_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes


def get_class_name(names, cls_id: int) -> str:
    """从模型 names 中解析类别名称."""
    if isinstance(names, dict):
        return names.get(int(cls_id), cls_id)
    if isinstance(names, (list, tuple)) and 0 <= int(cls_id) < len(names):
        return names[int(cls_id)]
    return cls_id


def predict_png_batch(model, batch: list, conf: float, iou: float,
                      max_det: int = 300, imgsz: int = 640) -> list:
    """对一批 PNG 瓦片执行 YOLO 推理.

    Args:
        model: ultralytics YOLO 模型实例.
        batch: 瓦片信息列表，每项 {"path": Path, "tile_id": int, "x_off": int, "y_off": int}.
        conf: 置信度阈值.
        iou: 瓦片级 NMS IoU 阈值.
        max_det: 每瓦片最大检出数.
        imgsz: 推理图像尺寸.

    Returns:
        检测结果列表，每项为包含全局像素坐标的字典.
    """
    if not batch:
        return []
    tensors = []
    meta = []
    for item in batch:
        rgb = np.asarray(Image.open(item["path"]).convert("RGB"), dtype=np.uint8)
        padded, gain, pad_x, pad_y = letterbox_rgb(rgb, imgsz)
        tensor = torch.tensor(
            np.ascontiguousarray(padded.transpose(2, 0, 1)).copy(), dtype=torch.float32
        ) / 255.0
        tensors.append(tensor)
        meta.append((rgb.shape, gain, pad_x, pad_y))
    device = next(model.model.parameters()).device
    images = torch.stack(tensors, dim=0).to(device)
    with torch.inference_mode():
        preds = model.model(images)
    results = nms.non_max_suppression(
        preds,
        conf,
        iou,
        max_det=max_det,
        nc=0,
        end2end=getattr(model.model, "end2end", False),
    )
    detections = []
    names = getattr(model, "names", None) or getattr(model.model, "names", {})
    for item, result, item_meta in zip(batch, results, meta):
        if result is None or len(result) == 0:
            continue
        original_shape, gain, pad_x, pad_y = item_meta
        boxes = result[:, :4].detach().cpu().numpy()
        boxes = scale_letterbox_boxes(boxes, gain, pad_x, pad_y, original_shape)
        confs = result[:, 4].detach().cpu().numpy()
        cls_ids = result[:, 5].detach().cpu().numpy().astype(int)
        x_off = item["x_off"]
        y_off = item["y_off"]
        tile_id = item["tile_id"]
        for box, conf_val, cls_id in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = box.tolist()
            detections.append(
                {
                    "cls_id": int(cls_id),
                    "cls_name": str(get_class_name(names, cls_id)),
                    "conf": float(conf_val),
                    "x1_pix": float(x1 + x_off),
                    "y1_pix": float(y1 + y_off),
                    "x2_pix": float(x2 + x_off),
                    "y2_pix": float(y2 + y_off),
                    "tile_id": int(tile_id),
                }
            )
    return detections


def load_model(model_path: str, device: str = "0"):
    """加载 YOLO 模型并移至指定设备.

    Args:
        model_path: 模型权重文件路径.
        device: 设备字符串.

    Returns:
        ultralytics YOLO 模型实例（已设为 eval 模式）.
    """
    model = YOLO(str(model_path))
    dev = resolve_device(device)
    model.model.to(dev)
    model.model.eval()
    return model
