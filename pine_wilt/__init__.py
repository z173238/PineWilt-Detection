"""松材线虫病枯死树检测 - 推理核心包.

基于 YOLO 的大幅面无人机遥感影像目标检测。
支持 GeoTIFF 滑窗推理、子进程容错、全局 NMS 和 Shapefile 输出。
"""

from pine_wilt.engine import run_pipeline

__all__ = ["run_pipeline"]
