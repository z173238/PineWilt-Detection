# 松材线虫病枯死树目标检测

基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 框架的无人机遥感影像枯死树（松材线虫病）检测系统。

## 算法简介

采用 **YOLO26l** 目标检测模型，对大幅面无人机正射影像（GeoTIFF）进行**滑窗推理**：

1. **瓦片切分** — 将超大 GeoTIFF 按 640×640 像素切分为瓦片，相邻瓦片重叠 128 像素以避免边界漏检
2. **图像归一化** — 逐波段 2%–98% 百分位拉伸至 uint8，适配多光谱/高光谱无人机数据
3. **批量推理** — 每批 8 个瓦片送入 YOLO 模型进行目标检测
4. **全局 NMS** — 跨瓦片合并重叠区域的重复检测框，按类别执行贪心非极大值抑制
5. **坐标输出** — 像素坐标转换为地理坐标（CRS），输出 ESRI Shapefile

### 模型指标

| 指标 | 值 |
|------|-----|
| 模型 | YOLO26l |
| mAP@0.5 | **0.8549** |
| mAP@0.5:0.95 | **0.4003** |
| 输入尺寸 | 640×640 |
| 类别数 | 1 (`bad_tree`) |
| 训练集 | 4301 张 |
| 硬件 | 2×RTX 5090 32GB |
| 训练轮次 | 70（best at epoch 20） |

## 目录结构

```
├── predict.py              # 统一推理入口
├── train.py                # 模型训练脚本
├── evaluate.py             # 模型评估脚本
├── pine_wilt/              # 推理核心包
│   ├── __init__.py
│   ├── engine.py           # 推理流水线编排（子进程容错）
│   ├── tiling.py           # GeoTIFF 瓦片切分与归一化
│   ├── inference.py        # YOLO 模型加载与批量推理
│   └── postprocess.py      # NMS 与 Shapefile 输出
├── datasetyolov11/         # 训练数据集
├── ultralytics/            # YOLO 核心框架
└── runs/                   # 训练/推理输出（gitignore）
```

## 环境要求

- Python >= 3.8
- PyTorch >= 1.8.0（推荐 CUDA 版本）
- 地理数据库：rasterio、fiona、shapely

```bash
# 安装 ultralytics 核心依赖
pip install -e .

# 安装地理数据处理依赖
pip install rasterio fiona shapely
```

## 快速开始

### 1. 获取数据集

数据集来自 Roboflow（松材线虫病树检测，CC BY 4.0 许可）：

```bash
# 方式一：直接从 Roboflow 下载
# https://universe.roboflow.com/project-dkq3q/-9pmdt/dataset/8

# 方式二：使用 Roboflow API（需先 pip install roboflow）
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_API_KEY')
project = rf.workspace('project-dkq3q').project('-9pmdt')
dataset = project.version(8).download('yolov8')
"
```

下载后将数据集目录命名为 `datasetyolov11/` 放在项目根目录，结构如下：

```
datasetyolov11/
├── data.yaml
├── train/images/   # 4301 张训练图片
├── train/labels/   # YOLO 格式标注
├── valid/images/   # 验证集
├── valid/labels/
├── test/images/    # 测试集
└── test/labels/
```

### 2. 获取预训练权重

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26l.pt
```

### 3. 训练模型

```bash
python train.py
```

默认使用 `datasetyolov11/` 数据集和 `yolo26l.pt` 预训练权重，输出至 `runs/pine_wilt/`。

训练配置可在 `train.py` 顶部的配置区域修改（epochs、batch、device 等）。

### 4. 推理

**基本用法：**

```bash
python predict.py --image /path/to/uav_image.tif --out results/detections.shp
```

**常用参数：**

```bash
python predict.py \
  --image /path/to/uav_image.tif \
  --out results/detections.shp \
  --model runs/detect/runs/pine_wilt/yolo26l_exp2/weights/best.pt \
  --device 0 \
  --conf 0.25 \
  --batch-size 8
```

**双 GPU 并行推理：**

```bash
# 获取总瓦片数
TOTAL=$(python -c "
from pine_wilt.tiling import get_tile_records
import rasterio
with rasterio.open('image.tif') as ds:
    print(len(get_tile_records(ds, 640, 128)))
")
MID=$((TOTAL / 2))

python predict.py --image image.tif --out gpu0.shp --device 0 --start-tile 1 --end-tile $MID &
python predict.py --image image.tif --out gpu1.shp --device 1 --start-tile $((MID+1)) --end-tile $TOTAL &
wait

ogrmerge.py -single -o merged.shp gpu0.shp gpu1.shp
```

### 5. 模型评估

```bash
# 修改 evaluate.py 中的 MODEL_PATH 为实际 best.pt 路径后运行
python evaluate.py
```

## 推理参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image` | (必填) | 输入 GeoTIFF 影像路径 |
| `--out` | (必填) | 输出 Shapefile 路径 |
| `--model` | best.pt | YOLO 模型权重路径 |
| `--conf` | 0.25 | 置信度阈值 |
| `--iou` | 0.7 | 瓦片级 NMS IoU 阈值 |
| `--merge-iou` | 0.5 | 全局跨瓦片 NMS IoU 阈值 |
| `--tile-size` | 640 | 瓦片像素尺寸 |
| `--overlap` | 128 | 瓦片间重叠像素 |
| `--imgsz` | 640 | YOLO 推理图像尺寸 |
| `--batch-size` | 8 | 每批推理瓦片数 |
| `--chunk-size` | 50 | 每个子进程处理的瓦片数 |
| `--device` | 0 | 推理设备（"0", "cpu" 等） |
| `--max-det` | 300 | 每瓦片最大检测数 |
| `--start-tile` | 1 | 起始瓦片索引（1-based） |
| `--end-tile` | 0 | 结束瓦片索引（0=全部） |
| `--skip-tiles` | "" | 跳过的瓦片索引（逗号分隔） |
| `--keep-tiles` | False | 保留缓存 PNG 瓦片 |
| `--no-subprocess` | False | 禁用子进程容错 |
| `--debug` | False | 打印瓦片级调试信息 |

## 容错机制

脚本采用**子进程 + 二分重试**的容错策略：

- 每 `chunk_size`（默认 50）个瓦片启动一个独立子进程执行推理，避免 GPU 显存累积
- 子进程崩溃时，自动将瓦片范围对半拆分并分别重试
- 单个瓦片重试失败后记录到 `.crashed_tiles.txt`，不阻塞其余瓦片
- 推理结束后打印失败瓦片列表，可根据 crash log 用 `--skip-tiles` 跳过重跑

## 输出 Shapefile 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `cls_id` | int | 类别 ID |
| `cls_name` | str | 类别名称 |
| `conf` | float | 置信度 |
| `x1_pix` | float | 检测框左上角 X（像素坐标） |
| `y1_pix` | float | 检测框左上角 Y（像素坐标） |
| `x2_pix` | float | 检测框右下角 X（像素坐标） |
| `y2_pix` | float | 检测框右下角 Y（像素坐标） |
| `tile_id` | int | 来源瓦片编号 |

## 更新日志

- **2025-06-30** — 工程化重构：统一 `predict.py` 入口，模块化 `pine_wilt/` 包，完善 README
- **2025-06-01** — 新增双 GPU 并行推理、子进程容错与崩溃重试机制
- **2025-05-27** — YOLO26l 实验 2（最终训练），SGD 优化器 + label smoothing + weight decay
- **2025-05-26** — 项目初始化，YOLO11l / YOLO26l 基线实验

## 许可

本项目基于 [Ultralytics AGPL-3.0](LICENSE) 许可协议。
