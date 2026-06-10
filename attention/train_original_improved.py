"""
YOLOv8n-cls 训练策略改进版本
用于垃圾四分类任务

主要改进：
1. Label Smoothing：缓解模型过度自信
2. Dropout：增强分类头正则化
3. 更大的 Weight Decay：降低过拟合风险
4. 更严格的 Early Stopping：减少无效训练
"""

import os
import json
import random
import sys
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from path_config import DATA_PATH, CLASSIFY_PATH
except ImportError:
    DATA_PATH = "/root/autodl-tmp/garbage_4cls"
    CLASSIFY_PATH = "/root/autodl-tmp/runs/classify"

def set_seed(seed=42):
    """固定随机种子，增强实验可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def main():
    set_seed(42)

    print("=" * 70)
    print("YOLOv8n-cls 训练策略改进版本")
    print("=" * 70)

    data_path = DATA_PATH
    imgsz = 320

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集路径不存在: {data_path}")

    print(f"\n📂 数据集路径: {data_path}")
    print("🎯 目标类别数: 4")
    print("📋 改进策略:")
    print("  - Label Smoothing: 0.1")
    print("  - Dropout: 0.1")
    print("  - Weight Decay: 0.001")
    print("  - Early Stopping patience: 15")

    print("\n📥 加载 YOLOv8n-cls 预训练模型...")
    model = YOLO("yolov8n-cls.pt")

    print("\n🚀 开始训练 YOLOv8n-cls 训练策略改进版...")

    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=imgsz,
        batch=32,
        name="waste_cls_improved_v1",
        device="cuda:0",
        amp=True,
        workers=8,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.001,
        warmup_epochs=3.0,

        degrees=15.0,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        mixup=0.15,

        label_smoothing=0.1,
        dropout=0.1,
        patience=15,

        verbose=True,
        save=True,
        save_period=10,
        plots=True
    )

    print("\n" + "=" * 70)
    print("✅ YOLOv8n-cls 训练策略改进版训练完成！")
    print("=" * 70)

    save_dir = os.path.join(CLASSIFY_PATH, "waste_cls_improved_v1")
    os.makedirs(save_dir, exist_ok=True)

    training_info = {
        "model_type": "YOLOv8n-cls with Training Strategy Improvements",
        "pretrained": "yolov8n-cls.pt",
        "attention": "None",
        "structure_improvement": "None",
        "training_strategy": [
            "label_smoothing",
            "dropout",
            "stronger_weight_decay",
            "stricter_early_stopping"
        ],
        "seed": 42,
        "epochs": 100,
        "imgsz": imgsz,
        "batch_size": 32,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.001,
        "warmup_epochs": 3.0,
        "label_smoothing": 0.1,
        "dropout": 0.1,
        "patience": 15,
        "data_augmentation": {
            "degrees": 15.0,
            "translate": 0.1,
            "scale": 0.2,
            "fliplr": 0.5,
            "mixup": 0.15
        },
        "note": "该版本不改变 YOLOv8n-cls 网络结构，仅通过标签平滑、Dropout、权重衰减和早停策略增强泛化能力。"
    }

    info_path = os.path.join(save_dir, "training_info.json")

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)

    print(f"📁 最佳模型保存路径: {save_dir}/weights/best.pt")
    print(f"💾 训练信息保存到: {info_path}")

if __name__ == "__main__":
    main()