"""
YOLOv8 + 冻结Backbone + ECA 注意力（终极方案）
核心思路：完全保留预训练特征，只让注意力学习如何优化
"""

import os
import json
import torch
import torch.nn as nn
from ultralytics import YOLO


class ECA(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


def add_eca_and_freeze(model, k_size=3):
    """
    添加ECA模块并冻结backbone
    """
    model_module = model.model.model
    
    target_layers = [6, 8]
    
    for layer_idx in target_layers:
        if layer_idx >= len(model_module):
            continue
            
        temp_input = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            for i in range(layer_idx + 1):
                temp_input = model_module[i](temp_input)
        out_channels = temp_input.shape[1]

        original_layer = model_module[layer_idx]

        class WrapperWithECA(nn.Module):
            def __init__(self, base_layer, channels, ks):
                super().__init__()
                self.base = base_layer
                self.eca = ECA(channels, ks)

            def forward(self, x):
                x = self.base(x)
                x = self.eca(x)
                return x

        wrapped = WrapperWithECA(original_layer, out_channels, k_size)
        model_module[layer_idx] = wrapped
    
    # 冻结backbone的所有原始层
    for name, param in model.named_parameters():
        # 只冻结非ECA层的参数
        if 'eca' not in name.lower() and 'classifier' not in name.lower():
            param.requires_grad = False
    
    # 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  冻结参数: {total_params - trainable_params:,}")
    
    return model


def main():
    print("="*70)
    print("YOLOv8 + ECA (冻结Backbone) 终极方案")
    print("="*70)

    data_path = '/root/autodl-tmp/garbage_4cls'

    print("\n📥 加载预训练YOLOv8模型...")
    model = YOLO('yolov8n-cls.pt')

    print("\n🔧 添加ECA并冻结Backbone...")
    model = add_eca_and_freeze(model, k_size=3)

    print("\n🚀 开始训练（只训练ECA和分类器）...")

    results = model.train(
        data=data_path,
        epochs=50,
        imgsz=320,
        batch=64,
        name='waste_cls_eca_frozen',
        device='cuda:0',
        amp=True,
        workers=16,
        optimizer='AdamW',       # 用AdamW更适合微调
        lr0=0.001,               # 较高学习率（因为只训少量参数）
        lrf=0.01,
        patience=20,
        verbose=True,
        save=True,
        save_period=10,
        warmup_epochs=3
    )

    print("\n" + "="*70)
    print("✅ 训练完成！")
    print("="*70)


if __name__ == "__main__":
    main()