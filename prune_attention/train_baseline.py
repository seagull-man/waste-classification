"""
训练基准 YOLOv8n-cls 模型
从预训练权重开始微调
"""
import os
import sys
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_PATH, MODEL_PATH, RUNS_DIR,
    BATCH_SIZE, IMGSZ, EPOCHS, LR0, WEIGHT_DECAY,
    MOMENTUM, WARMUP_EPOCHS, PATIENCE, DEVICE_ID,
)


def train_baseline():
    """训练基准 YOLOv8n-cls 模型"""
    model_name = "baseline"
    project_dir = os.path.join(RUNS_DIR, model_name)

    print(f"[INFO] 开始训练基准模型: {model_name}")
    print(f"[INFO] 数据集路径: {DATA_PATH}")
    print(f"[INFO] 预训练权重: {MODEL_PATH}")

    # 用预训练权重加载模型
    model = YOLO(MODEL_PATH)

    results = model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        lr0=LR0,
        weight_decay=WEIGHT_DECAY,
        momentum=MOMENTUM,
        warmup_epochs=WARMUP_EPOCHS,
        patience=PATIENCE,
        device=DEVICE_ID,
        project=project_dir,
        name="train",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        verbose=True,
    )

    best_pt = os.path.join(project_dir, "train", "weights", "best.pt")
    print(f"\n[SUCCESS] 基准模型训练完成!")
    print(f"[SUCCESS] 最佳模型: {best_pt}")

    return results


if __name__ == "__main__":
    train_baseline()