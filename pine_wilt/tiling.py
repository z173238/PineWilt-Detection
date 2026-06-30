"""GeoTIFF 瓦片切分与预处理.

包括：瓦片网格计算、影像窗口读取、百分位归一化、PNG 缓存。
"""

import numpy as np
from PIL import Image
from rasterio.windows import Window


def tile_offsets(length: int, tile_size: int, overlap: int) -> list:
    """计算单轴方向上的瓦片起始像素偏移量."""
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than tile-size")
    offsets = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if offsets[-1] != last:
        offsets.append(last)
    return offsets


def to_uint8(image: np.ndarray) -> np.ndarray:
    """逐波段 2%-98% 百分位拉伸到 uint8."""
    if image.dtype == np.uint8:
        return image
    output = np.zeros(image.shape, dtype=np.uint8)
    for band_idx in range(image.shape[2]):
        band = image[:, :, band_idx].astype(np.float32)
        valid = np.isfinite(band)
        if not valid.any():
            continue
        low, high = np.percentile(band[valid], (2, 98))
        if high <= low:
            high = float(band[valid].max())
            low = float(band[valid].min())
        if high <= low:
            continue
        output[:, :, band_idx] = np.clip((band - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    return output


def read_tile(dataset, x_off: int, y_off: int, tile_size: int) -> np.ndarray:
    """从 rasterio 数据集中读取一个瓦片窗口，返回 RGB uint8 图像."""
    width = min(tile_size, dataset.width - x_off)
    height = min(tile_size, dataset.height - y_off)
    indexes = list(range(1, min(dataset.count, 3) + 1))
    data = dataset.read(indexes=indexes, window=Window(x_off, y_off, width, height), boundless=False)
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    elif data.shape[0] == 2:
        data = np.concatenate([data, data[:1]], axis=0)
    image = np.moveaxis(data, 0, -1)
    image = to_uint8(image)
    return np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)


def tile_png_path(tile_cache_dir, tile_index: int, x_off: int, y_off: int):
    """生成瓦片 PNG 缓存文件路径."""
    return tile_cache_dir / f"tile_{tile_index:06d}_x{x_off}_y{y_off}.png"


def get_tile_records(dataset, tile_size: int, overlap: int) -> list:
    """计算整幅影像的瓦片网格，返回包含 tile_id/x_off/y_off 的记录列表."""
    x_offsets = tile_offsets(dataset.width, tile_size, overlap)
    y_offsets = tile_offsets(dataset.height, tile_size, overlap)
    records = []
    tile_index = 0
    for y_off in y_offsets:
        for x_off in x_offsets:
            tile_index += 1
            records.append({"tile_id": tile_index, "x_off": x_off, "y_off": y_off})
    return records


def parse_skip_tiles(skip_tiles: str) -> set:
    """解析逗号分隔的要跳过的瓦片索引."""
    return {int(x.strip()) for x in skip_tiles.split(",") if x.strip()}


def write_tile(dataset, record: dict, tile_cache_dir, tile_size: int = 640,
               debug: bool = False, overwrite: bool = False):
    """从影像读取一个瓦片，写入 PNG 缓存文件.

    Args:
        dataset: rasterio 打开的影像数据集.
        record: 瓦片记录 (tile_id, x_off, y_off).
        tile_cache_dir: PNG 缓存目录.
        tile_size: 瓦片尺寸（像素）.
        debug: 是否打印调试信息.
        overwrite: 是否覆盖已有缓存.

    Returns:
        PNG 文件路径，如果瓦片无效则返回 None.
    """
    tile_index = record["tile_id"]
    x_off = record["x_off"]
    y_off = record["y_off"]
    png_path = tile_png_path(tile_cache_dir, tile_index, x_off, y_off)
    if png_path.exists() and not overwrite:
        return png_path
    tile = read_tile(dataset, x_off, y_off, tile_size)
    if tile.size == 0 or tile.max() == tile.min():
        if debug:
            value = int(tile.min()) if tile.size else "empty"
            print(f"Blank tile {tile_index}: x={x_off}, y={y_off}, value={value}", flush=True)
        return None
    if debug:
        print(
            f"Tile {tile_index}: x={x_off}, y={y_off}, "
            f"min={int(tile.min())}, max={int(tile.max())}, "
            f"mean={tile.mean():.2f}, std={tile.std():.2f}",
            flush=True,
        )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tile).save(png_path)
    return png_path
