import os
import json
import random
import numpy as np
import torch
from ultralytics import YOLO

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)

    print("=" * 70)
    print("YOLOv8s 原始模型训练（更大的模型）")
    print("=" * 70)

    data_path = "/root/autodl-tmp/garbage_4cls"

    print("\n📥 加载 YOLOv8s-cls 预训练模型...")
    model = YOLO("yolov8s-cls.pt")

    print("\n🚀 开始训练 YOLOv8s-cls...")
    print(f"📂 数据集路径: {data_path}")
    print("🎯 目标类别数: 4")

    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=320,
        batch=32,
        name="waste_cls_yolov8s",
        device="cuda:0",
        amp=True,
        workers=8,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        degrees=15.0,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        patience=20,
        verbose=True,
        save=True,
        save_period=10,
        plots=True
    )

    print("\n" + "=" * 70)
    print("✅ YOLOv8s 训练完成！")
    print("=" * 70)

    save_dir = "runs/classify/waste_cls_yolov8s"
    os.makedirs(save_dir, exist_ok=True)

    training_info = {
        "model_type": "YOLOv8s-cls",
        "pretrained": "yolov8s-cls.pt",
        "attention": "None",
        "seed": 42,
        "epochs": 100,
        "imgsz": 320,
        "batch_size": 32,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "data_augmentation": {
            "degrees": 15.0,
            "translate": 0.1,
            "scale": 0.2,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.15
        },
        "note": "YOLOv8s-cls 更大的模型，用于垃圾分类任务。"
    }

    info_path = os.path.join(save_dir, "training_info.json")

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)

    print(f"📁 最佳模型保存路径: {save_dir}/weights/best.pt")
    print(f"💾 训练信息保存到: {info_path}")

if __name__ == "__main__":
    main()
