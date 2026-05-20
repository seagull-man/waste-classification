"""
YOLOv8 原始模型训练脚本（对照组）
用于与SE-Block模型进行性能对比

训练参数与SE模型保持完全一致，确保对比的公平性
"""

import os
import json
from ultralytics import YOLO


def main():
    print("="*70)
    print("YOLOv8 原始模型训练（对照组）")
    print("="*70)

    data_path = '/root/autodl-tmp/garbage_4cls'

    model = YOLO('yolov8n-cls.pt')

    print(f"\n🚀 开始训练原始模型...")
    print(f"📂 数据集路径: {data_path}")
    print(f"🎯 目标类别数: 4")

    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=320,
        batch=32,
        name="waste_cls_original",
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
        save_period=10
    )

    print("\n" + "="*70)
    print("✅ 原始模型训练完成！")
    print("="*70)
    print(f"📁 模型保存路径: runs/classify/waste_cls_original/weights/best.pt")

    save_dir = 'runs/classify/waste_cls_original'
    os.makedirs(save_dir, exist_ok=True)

    training_info = {
        'model_type': 'Original YOLOv8n-cls',
        'epochs': 100,
        'batch_size': 32,
        'optimizer': 'AdamW',
        'learning_rate': 0.001,
        'weight_decay': 0.0005,
        'data_augmentation': {
            'degrees': 15.0,
            'translate': 0.1,
            'scale': 0.2,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.15
        },
        'note': '原始模型，不含任何注意力机制'
    }

    info_path = f'{save_dir}/training_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    print(f"💾 训练信息保存到: {info_path}")

if __name__ == "__main__":
    main()
