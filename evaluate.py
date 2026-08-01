"""
松材线虫病目标检测 - 模型评估脚本.

在测试集上计算 mAP 等指标，并输出可视化推理结果。
"""

from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

# ========== 配置区域 ==========
MODEL_PATH = BASE_DIR / "runs/detect/runs/pine_wilt/yolo26l_exp2/weights/best.pt"
DATA_YAML = BASE_DIR / "datasetyolov11" / "data.yaml"
TEST_IMGS = BASE_DIR / "datasetyolov11" / "test" / "images"
# ==============================

# 若 best.pt 不存在，尝试自动查找
if not MODEL_PATH.exists():
    candidates = sorted(BASE_DIR.glob("runs/**/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        MODEL_PATH = candidates[0]
        print(f"自动选择权重: {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"未找到权重文件，请修改 MODEL_PATH。\n搜索路径: {MODEL_PATH}")

print(f"权重文件: {MODEL_PATH}")
print(f"数据配置: {DATA_YAML}")
print(f"测试图片: {TEST_IMGS}")

model = YOLO(str(MODEL_PATH))

# 在测试集上计算 mAP 等指标
print("\n===== 测试集评估 =====")
metrics = model.val(data=str(DATA_YAML), split="test")
print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
print(f"mAP@0.5   : {metrics.box.map50:.4f}")
print(f"mAP@0.5:95: {metrics.box.map:.4f}")

# 在测试图片上推理并保存结果图
print("\n===== 推理并保存可视化结果 =====")
results = model.predict(
    source=str(TEST_IMGS),
    save=True,
    conf=0.25,
    project="runs/pine_wilt",
    name="test_predict",
)
print("结果图保存至: runs/pine_wilt/test_predict/")
