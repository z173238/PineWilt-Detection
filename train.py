"""
松材线虫病目标检测 - YOLO26l 训练脚本
使用官方 yolo26l.pt 预训练权重
将整个项目目录上传服务器后，直接运行此脚本，无需修改路径
"""

from pathlib import Path
from ultralytics import YOLO

# ========== 配置区域（如需调整可修改这里）==========
EPOCHS = 150
IMGSZ = 640
BATCH = 64      # 2×RTX5090 32GB，每卡 32，显存盈余大
PATIENCE = 50
PROJECT = "runs/pine_wilt"
NAME = "yolo26l_exp2"
DEVICE = "0,1"  # 2×RTX 5090
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "datasetyolov11"
MODEL_PATH = BASE_DIR / "yolo26l.pt"
YAML_PATH = BASE_DIR / "pine_wilt_server.yaml"

assert DATASET_PATH.exists(), f"数据集目录不存在: {DATASET_PATH}"
assert MODEL_PATH.exists(), f"预训练权重不存在: {MODEL_PATH}"

data_yaml = f"""path: {DATASET_PATH.as_posix()}
train: train/images
val: valid/images
test: test/images
nc: 1
names:
  0: bad_tree
"""

with open(YAML_PATH, "w") as f:
    f.write(data_yaml)

print(f"数据集路径: {DATASET_PATH}")
print(f"预训练权重: {MODEL_PATH}")
print(f"模型: yolo26l  |  轮次: {EPOCHS}  |  分辨率: {IMGSZ}  |  Batch: {BATCH}")

model = YOLO(str(MODEL_PATH))
results = model.train(
    data=str(YAML_PATH),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=PATIENCE,
    project=PROJECT,
    name=NAME,
    device=DEVICE,
    optimizer="SGD",
    lr0=0.0005,
    cos_lr=True,
    copy_paste=0.3,
    weight_decay=0.001,
    label_smoothing=0.1,
    flipud=0.5,
    degrees=10.0,
)

print("\n训练完成！")
print(f"最优权重: {PROJECT}/{NAME}/weights/best.pt")
