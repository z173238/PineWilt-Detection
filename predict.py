#!/usr/bin/env python3
"""松材线虫病枯死树检测 - 统一推理入口.

用法:
    python predict.py --image <tif路径> --out <输出shp路径> [可选参数]

示例:
    python predict.py --image uav_image.tif --out results/detections.shp
    python predict.py --image uav_image.tif --out results/detections.shp --conf 0.5 --device 0,1
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="松材线虫病枯死树检测 - GeoTIFF 滑窗推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python predict.py --image input.tif --out output.shp
  python predict.py --image input.tif --out output.shp --conf 0.5 --device 0
  python predict.py --image input.tif --out output.shp --start-tile 1 --end-tile 500
        """,
    )

    # ---- 必选参数 ----
    parser.add_argument("--image", required=True, help="输入 GeoTIFF 影像路径")
    parser.add_argument("--out", required=True, help="输出 Shapefile 路径 (.shp)")

    # ---- 模型参数 ----
    parser.add_argument(
        "--model",
        default="runs/detect/runs/pine_wilt/yolo26l_exp2/weights/best.pt",
        help="YOLO 模型权重路径 (默认: best.pt)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理图像尺寸 (默认: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou", type=float, default=0.7, help="瓦片级 NMS IoU 阈值 (默认: 0.7)")
    parser.add_argument("--merge-iou", type=float, default=0.5, help="全局跨瓦片 NMS IoU 阈值 (默认: 0.5)")
    parser.add_argument("--max-det", type=int, default=300, help="每瓦片最大检测数 (默认: 300)")
    parser.add_argument("--device", default="0", help='推理设备: "0", "cpu" 等 (默认: 0)')

    # ---- 瓦片参数 ----
    parser.add_argument("--tile-size", type=int, default=640, help="瓦片像素尺寸 (默认: 640)")
    parser.add_argument("--overlap", type=int, default=128, help="瓦片间重叠像素 (默认: 128)")

    # ---- 性能参数 ----
    parser.add_argument("--batch-size", type=int, default=8, help="每批推理瓦片数 (默认: 8)")
    parser.add_argument("--chunk-size", type=int, default=50, help="每个子进程处理的瓦片数 (默认: 50)")

    # ---- 范围控制 ----
    parser.add_argument("--start-tile", type=int, default=1, help="起始瓦片索引，1-based (默认: 1)")
    parser.add_argument("--end-tile", type=int, default=0, help="结束瓦片索引，0=全部 (默认: 0)")
    parser.add_argument("--skip-tiles", default="", help="要跳过的瓦片索引，逗号分隔")

    # ---- 运行时选项 ----
    parser.add_argument("--tile-cache-dir", default="", help="PNG 瓦片缓存目录 (默认: <out>_tiles/)")
    parser.add_argument("--log-every", type=int, default=50, help="每 N 个瓦片打印进度 (默认: 50)")
    parser.add_argument("--crash-log", default="", help="崩溃瓦片记录文件路径")
    parser.add_argument("--keep-tiles", action="store_true", help="保留缓存 PNG 瓦片")
    parser.add_argument("--overwrite-tiles", action="store_true", help="覆盖已有缓存瓦片")
    parser.add_argument("--no-subprocess", action="store_true", help="禁用子进程容错，直接推理")
    parser.add_argument("--debug", action="store_true", help="打印瓦片级调试信息")

    # ---- 内部参数（子进程通信用）----
    parser.add_argument("--worker-json", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-start-tile", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-end-tile", type=int, default=0, help=argparse.SUPPRESS)

    return parser.parse_args()


def main():
    args = parse_args()

    # 将 --no-subprocess 转换为 engine 能理解的 subprocess 标志
    args.subprocess = not args.no_subprocess

    # 确保项目根目录在 sys.path 中，使 ultralytics 可导入
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pine_wilt.engine import run_pipeline

    run_pipeline(args)


if __name__ == "__main__":
    main()
