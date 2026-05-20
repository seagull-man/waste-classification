"""
YOLOv8 + ECA-Net 轻量注意力机制
ECA: Efficient Channel Attention
特点：无降维、参数极少、适合小数据集
"""

import os
import json
import torch
import torch.nn as nn
from ultralytics import YOLO


class ECA(nn.Module):
    """Efficient Channel Attention Module
    
    相比SE的优势：
    1. 不使用全连接层降维，避免信息丢失
    2. 参数量极少（只有kernel_size*k个参数）
    3. 使用1D卷积捕获局部通道关系，更高效
    """
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 根据通道数自适应卷积核大小
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)  # [B, 1, C]
        y = self.conv(y)                     # 局部通道交互
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualECA(nn.Module):
    """残差式ECA模块
    
    初始时gamma=0，不影响原始特征
    训练中如果ECA有用，gamma自动增大
    """
    def __init__(self, channel, k_size=3):
        super(ResidualECA, self).__init__()
        self.eca = ECA(channel, k_size)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gamma * self.eca(x)


def add_eca_to_yolov8(model, k_size=3):
    """
    在YOLOv8 backbone中添加残差ECA模块
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
                self.residual_eca = ResidualECA(channels, ks)

            def forward(self, x):
                x = self.base(x)
                x = self.residual_eca(x)
                return x

        wrapped = WrapperWithECA(original_layer, out_channels, k_size)
        model_module[layer_idx] = wrapped

    return model


def main():
    print("="*70)
    print("YOLOv8 + ECA-Net 轻量注意力（参数极少的方案）")
    print("="*70)

    data_path = '/root/autodl-tmp/garbage_4cls'

    print("\n📥 加载预训练YOLOv8模型...")
    model = YOLO('yolov8n-cls.pt')

    print("\n🔧 添加ECA注意力模块...")
    model = add_eca_to_yolov8(model, k_size=3)

    print("\n🚀 开始训练（更长epoch + 更强增强）...")

    results = model.train(
        data=data_path,
        epochs=50,         # 更多epoch让ECA充分学习
        imgsz=320,
        batch=64,
        name='waste_cls_eca',
        device='cuda:0',
        amp=True,
        workers=16,
        optimizer='auto',
        patience=30,        # 更大的耐心值
        verbose=True,
        save=True,
        save_period=10,
        warmup_epochs=5,
        mosaic=1.0,
        mixup=0.1,          # 轻度mixup
        copy_paste=0.05     # 轻度copy_paste
    )

    print("\n" + "="*70)
    print("✅ 训练完成！")
    print("="*70)


if __name__ == "__main__":
    main()
