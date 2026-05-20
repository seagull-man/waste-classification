"""
YOLOv8 + CBAM 注意力机制
"""

import os
import torch
import torch.nn as nn
from ultralytics import YOLO


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=4, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out


def add_cbam_to_yolov8(model):
    """在YOLOv8最后一层添加CBAM"""
    model_module = model.model.model
    layer_idx = 8  # 最后一个backbone层
    
    if layer_idx >= len(model_module):
        return model
        
    # 获取输出通道数
    temp_input = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        for i in range(layer_idx + 1):
            temp_input = model_module[i](temp_input)
    out_channels = temp_input.shape[1]

    original_layer = model_module[layer_idx]

    class WrapperWithCBAM(nn.Module):
        def __init__(self, base_layer, channels):
            super().__init__()
            self.base = base_layer
            self.cbam = CBAM(channels)

        def forward(self, x):
            x = self.base(x)
            x = self.cbam(x)
            return x

    wrapped = WrapperWithCBAM(original_layer, out_channels)
    model_module[layer_idx] = wrapped
    
    return model


def main():
    print("="*70)
    print("YOLOv8 + CBAM 注意力机制")
    print("="*70)

    data_path = '/root/autodl-tmp/garbage_4cls'

    print("\n📥 加载预训练YOLOv8模型...")
    model = YOLO('yolov8n-cls.pt')

    print("\n🔧 添加CBAM注意力...")
    model = add_cbam_to_yolov8(model)

    print("\n🚀 开始训练...")

    results = model.train(
        data=data_path,
        epochs=50,
        imgsz=320,
        batch=64,
        name='waste_cls_cbam',
        device='cuda:0',
        amp=True,
        workers=8,
        optimizer='AdamW',
        lr0=0.0005,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.00005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.00005,
        patience=None,
        close_mosaic=10,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=5.0,
        translate=0.05,
        scale=0.1,
        fliplr=0.5,
        verbose=True,
        save=True,
        save_period=10,
        plots=True
    )

    print("\n" + "="*70)
    print("✅ 训练完成！")
    print("="*70)


if __name__ == "__main__":
    main()