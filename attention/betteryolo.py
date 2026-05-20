"""
YOLOv8 原始模型高级优化策略
针对垃圾分类任务的专属优化
"""

import os
import json
from ultralytics import YOLO


def main():
    print("="*70)
    print("YOLOv8 原始模型（高级优化版）")
    print("="*70)

    data_path = '/root/autodl-tmp/garbage_4cls'

    print("\n📥 加载预训练YOLOv8模型...")
    model = YOLO('yolov8n-cls.pt')

    print("\n🚀 开始高级优化训练...")
    print("  🔧 专属垃圾分类任务的优化策略")

    results = model.train(
        data=data_path,
        # 1. 训练参数优化
        epochs=150,         # 更长的训练时间
        imgsz=416,          # 更大的输入尺寸（捕获更多细节）
        batch=64,           # 充分利用4090的48GB显存
        name='waste_cls_optimized',
        device='cuda:0',
        amp=True,
        workers=16,         # 充分利用128核CPU
        optimizer='SGD',     # SGD更适合分类任务的收敛
        lr0=0.01,           # 更高的初始学习率
        lrf=0.0001,         # 更低的最终学习率
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        patience=35,         # 更大的耐心值，避免过早停止
        verbose=True,
        save=True,
        save_period=10,
        
        # 2. 激进的数据增强（针对垃圾分类特点）
        mosaic=1.0,
        mixup=0.25,          # 更强的样本混合
        copy_paste=0.15,     # 复制粘贴增强
        degrees=45.0,        # 更大的旋转角度（垃圾可能有不同朝向）
        translate=0.25,      # 更大的平移
        scale=0.4,           # 更大的缩放（垃圾大小不一）
        fliplr=0.5,
        flipud=0.2,          # 上下翻转（某些垃圾上下不影响分类）
        hsv_h=0.3,           # 更强的颜色增强（垃圾颜色多样）
        hsv_s=0.6,
        hsv_v=0.6,
        erasing=0.3,         # 随机擦除（模拟遮挡）
        
        # 3. 正则化和泛化
        dropout=0.1,         # 轻微的dropout
        close_mosaic=10,     # 最后10个epoch关闭mosaic，专注于精细化学习
        cos_lr=True,         # 余弦学习率衰减
    )

    print("\n" + "="*70)
    print("✅ 高级优化训练完成！")
    print("="*70)
    print(f"📁 模型保存路径: runs/classify/waste_cls_optimized/weights/best.pt")


if __name__ == "__main__":
    main()