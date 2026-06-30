"""检测结果后处理.

包括：像素→地理坐标转换、全局跨瓦片 NMS、ESRI Shapefile 输出。
"""

import fiona
import numpy as np
from shapely.geometry import Polygon, mapping


def pixel_polygon(transform, x1: float, y1: float, x2: float, y2: float) -> Polygon:
    """将像素坐标边界框转换为地理坐标 Polygon."""
    points = [
        transform * (x1, y1),
        transform * (x2, y1),
        transform * (x2, y2),
        transform * (x1, y2),
    ]
    return Polygon(points)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """计算单个框与一组框的 IoU（向量化）."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    boxes_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    return inter / np.maximum(union, 1e-9)


def global_nms(detections: list, iou_threshold: float) -> list:
    """跨瓦片全局类别感知贪心 NMS.

    按置信度降序排列，抑制同类中 IoU 超过阈值的低置信度检测。
    """
    if not detections:
        return []
    boxes = np.array(
        [[d["x1_pix"], d["y1_pix"], d["x2_pix"], d["y2_pix"]] for d in detections],
        dtype=np.float32,
    )
    scores = np.array([d["conf"] for d in detections], dtype=np.float32)
    classes = np.array([d["cls_id"] for d in detections], dtype=np.int32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[current], boxes[rest])
        same_class = classes[rest] == classes[current]
        order = rest[~((ious > iou_threshold) & same_class)]
    return [detections[i] for i in keep]


def add_detection_geometries(detections: list, transform) -> list:
    """为每个检测结果附加地理坐标 Polygon geometry."""
    for detection in detections:
        detection["geometry"] = pixel_polygon(
            transform,
            detection["x1_pix"],
            detection["y1_pix"],
            detection["x2_pix"],
            detection["y2_pix"],
        )
    return detections


def save_shapefile(detections: list, out_path: str, crs, merge_iou: float = 0.5) -> int:
    """对检测结果执行全局 NMS 并保存为 ESRI Shapefile.

    Args:
        detections: 检测结果列表.
        out_path: 输出 .shp 文件路径.
        crs: 坐标参考系（rasterio CRS 对象）.
        merge_iou: 全局 NMS IoU 阈值.

    Returns:
        合并后的最终检测数量.
    """
    merged = global_nms(detections, merge_iou)
    schema = {
        "geometry": "Polygon",
        "properties": {
            "cls_id": "int",
            "cls_name": "str:80",
            "conf": "float",
            "x1_pix": "float",
            "y1_pix": "float",
            "x2_pix": "float",
            "y2_pix": "float",
            "tile_id": "int",
        },
    }
    open_kwargs = {
        "driver": "ESRI Shapefile",
        "schema": schema,
        "encoding": "utf-8",
    }
    if crs:
        open_kwargs["crs_wkt"] = crs.to_wkt() if hasattr(crs, "to_wkt") else str(crs)
    with fiona.open(str(out_path), "w", **open_kwargs) as sink:
        for detection in merged:
            sink.write(
                {
                    "geometry": mapping(detection["geometry"]),
                    "properties": {
                        "cls_id": int(detection["cls_id"]),
                        "cls_name": str(detection["cls_name"]),
                        "conf": float(detection["conf"]),
                        "x1_pix": float(detection["x1_pix"]),
                        "y1_pix": float(detection["y1_pix"]),
                        "x2_pix": float(detection["x2_pix"]),
                        "y2_pix": float(detection["y2_pix"]),
                        "tile_id": int(detection["tile_id"]),
                    },
                }
            )
    return len(merged)
