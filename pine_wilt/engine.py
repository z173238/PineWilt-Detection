"""推理流水线编排.

支持子进程容错模式（默认）和直接推理模式。
子进程模式下每 chunk_size 个瓦片启动一个子进程，崩溃时自动二分重试。
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import rasterio

from pine_wilt.tiling import get_tile_records, parse_skip_tiles, write_tile
from pine_wilt.inference import load_model, predict_png_batch
from pine_wilt.postprocess import add_detection_geometries, save_shapefile


def batched(items: list, batch_size: int):
    """将列表按 batch_size 分组迭代."""
    for start in range(0, len(items), batch_size):
        yield items[start: start + batch_size]


def run_worker(args):
    """子进程 Worker：准备瓦片缓存 → 加载模型 → 批量推理 → 写 JSON."""
    image_path = Path(args.image)
    model_path = Path(args.model)
    tile_cache_dir = Path(args.tile_cache_dir)
    assert image_path.exists(), f"Input image not found: {image_path}"
    assert model_path.exists(), f"Model weights not found: {model_path}"
    worker_start = args.worker_start_tile
    worker_end = args.worker_end_tile
    skip_tiles = parse_skip_tiles(args.skip_tiles)
    candidates = []
    with rasterio.open(image_path) as dataset:
        records = get_tile_records(dataset, args.tile_size, args.overlap)
        for record in records:
            tile_id = record["tile_id"]
            if tile_id < worker_start or tile_id > worker_end or tile_id in skip_tiles:
                continue
            png_path = write_tile(
                dataset, record, tile_cache_dir,
                tile_size=args.tile_size, debug=args.debug, overwrite=args.overwrite_tiles,
            )
            if png_path is None:
                continue
            candidates.append({**record, "path": png_path})
            if args.log_every > 0 and tile_id % args.log_every == 0:
                print(f"Prepared tile {tile_id}, valid_tiles={len(candidates)}", flush=True)
    model = load_model(str(model_path), args.device)
    detections = []
    for batch in batched(candidates, max(1, args.batch_size)):
        detections.extend(predict_png_batch(
            model, batch,
            conf=args.conf, iou=args.iou, max_det=args.max_det, imgsz=args.imgsz,
        ))
        last_tile = batch[-1]["tile_id"]
        if args.log_every > 0:
            print(f"Predicted through tile {last_tile}, detections={len(detections)}", flush=True)
    if not args.keep_tiles:
        for item in candidates:
            try:
                item["path"].unlink(missing_ok=True)
            except OSError:
                pass
    worker_json = Path(args.worker_json)
    worker_json.parent.mkdir(parents=True, exist_ok=True)
    worker_json.write_text(json.dumps(detections), encoding="utf-8")
    print(f"Saved worker detections JSON: {worker_json}", flush=True)


def _run_child_process(args, start_tile: int, end_tile: int, temp_dir: Path,
                       crashed_tiles: list) -> list:
    """启动一个子进程处理瓦片范围；崩溃时二分重试."""
    json_path = temp_dir / f"detections_{start_tile}_{end_tile}.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "predict.py"),
        "--image", str(args.image),
        "--model", str(args.model),
        "--out", str(args.out),
        "--tile-cache-dir", str(args.tile_cache_dir),
        "--tile-size", str(args.tile_size),
        "--overlap", str(args.overlap),
        "--imgsz", str(args.imgsz),
        "--conf", str(args.conf),
        "--iou", str(args.iou),
        "--merge-iou", str(args.merge_iou),
        "--device", str(args.device),
        "--max-det", str(args.max_det),
        "--batch-size", str(args.batch_size),
        "--log-every", str(args.log_every),
        "--skip-tiles", args.skip_tiles,
        "--worker-json", str(json_path),
        "--worker-start-tile", str(start_tile),
        "--worker-end-tile", str(end_tile),
    ]
    if args.keep_tiles:
        cmd.append("--keep-tiles")
    if args.overwrite_tiles:
        cmd.append("--overwrite-tiles")
    if args.debug:
        cmd.append("--debug")
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode == 0:
        if result.stdout:
            print(result.stdout, end="")
        if json_path.exists():
            return json.loads(json_path.read_text(encoding="utf-8"))
        return []
    print(f"Child failed for tiles {start_tile}-{end_tile}, returncode={result.returncode}", flush=True)
    if result.stdout:
        print(result.stdout[-4000:], end="")
    if result.stderr:
        print(result.stderr[-4000:], end="")
    if start_tile == end_tile:
        crashed_tiles.append(start_tile)
        print(f"Recorded crashed tile: {start_tile}", flush=True)
        return []
    mid_tile = (start_tile + end_tile) // 2
    return (
        _run_child_process(args, start_tile, mid_tile, temp_dir, crashed_tiles)
        + _run_child_process(args, mid_tile + 1, end_tile, temp_dir, crashed_tiles)
    )


def run_pipeline(args):
    """主推理流水线：切分 → 子进程推理 → 合并 → 输出 Shapefile.

    Args:
        args: argparse.Namespace，包含所有运行时参数.
    """
    # ---- Worker 模式（子进程内部入口）----
    if getattr(args, "worker_json", None):
        run_worker(args)
        return

    # ---- 主流程 ----
    image_path = Path(args.image)
    model_path = Path(args.model)
    out_path = Path(args.out)
    assert image_path.exists(), f"Input image not found: {image_path}"
    assert model_path.exists(), f"Model weights not found: {model_path}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.tile_cache_dir:
        args.tile_cache_dir = str(out_path.parent / f"{out_path.stem}_tiles")
    tile_cache_dir = Path(args.tile_cache_dir)
    tile_cache_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(image_path) as dataset:
        records = get_tile_records(dataset, args.tile_size, args.overlap)
        total = len(records)
        end_tile = args.end_tile or total
        transform = dataset.transform
        crs = dataset.crs
        print(f"Image: {image_path}", flush=True)
        print(f"Size: {dataset.width} x {dataset.height}, CRS: {crs}", flush=True)
        print(f"Tiles: {total}, tile_size={args.tile_size}, overlap={args.overlap}", flush=True)
        print(f"Processing tiles {args.start_tile} to {end_tile}, chunk_size={args.chunk_size}", flush=True)
        print(f"Tile cache: {tile_cache_dir}", flush=True)

    use_subprocess = getattr(args, "subprocess", True)
    all_detections = []
    crashed_tiles = []

    if use_subprocess:
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            start = args.start_tile
            while start <= end_tile:
                chunk_end = min(start + args.chunk_size - 1, end_tile)
                print(f"Running child chunk: {start}-{chunk_end}", flush=True)
                all_detections.extend(
                    _run_child_process(args, start, chunk_end, temp_dir, crashed_tiles)
                )
                start = chunk_end + 1
    else:
        # 直接推理模式（无子进程容错）
        skip_tiles = parse_skip_tiles(args.skip_tiles)
        candidates = []
        with rasterio.open(image_path) as dataset:
            for record in records:
                tile_id = record["tile_id"]
                if tile_id < args.start_tile or tile_id > end_tile or tile_id in skip_tiles:
                    continue
                png_path = write_tile(
                    dataset, record, tile_cache_dir,
                    tile_size=args.tile_size, debug=args.debug,
                    overwrite=args.overwrite_tiles,
                )
                if png_path is None:
                    continue
                candidates.append({**record, "path": png_path})
                if args.log_every > 0 and tile_id % args.log_every == 0:
                    print(f"Prepared tile {tile_id}, valid_tiles={len(candidates)}", flush=True)
        model = load_model(str(model_path), args.device)
        for batch in batched(candidates, max(1, args.batch_size)):
            all_detections.extend(predict_png_batch(
                model, batch,
                conf=args.conf, iou=args.iou, max_det=args.max_det, imgsz=args.imgsz,
            ))
            last_tile = batch[-1]["tile_id"]
            if args.log_every > 0:
                print(f"Predicted through tile {last_tile}, detections={len(all_detections)}", flush=True)
        if not args.keep_tiles:
            for item in candidates:
                try:
                    item["path"].unlink(missing_ok=True)
                except OSError:
                    pass

    # 记录崩溃瓦片
    if crashed_tiles:
        crash_log = Path(args.crash_log) if getattr(args, "crash_log", None) else out_path.with_suffix(".crashed_tiles.txt")
        crash_log.parent.mkdir(parents=True, exist_ok=True)
        crash_log.write_text("\n".join(str(tile) for tile in crashed_tiles) + "\n", encoding="utf-8")
        print(f"Crashed tiles: {','.join(str(tile) for tile in crashed_tiles)}", flush=True)
        print(f"Saved crash log: {crash_log}", flush=True)

    # 附加地理坐标 → 全局 NMS → 输出 Shapefile
    detections = add_detection_geometries(all_detections, transform)
    print(f"Raw detections: {len(detections)}", flush=True)
    merged_count = save_shapefile(detections, out_path, crs, args.merge_iou)
    print(f"Merged detections: {merged_count}", flush=True)
    print(f"Saved Shapefile: {out_path}", flush=True)
