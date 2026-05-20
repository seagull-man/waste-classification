"""
YOLOv8n-cls + CBAM 注意力机制
用于垃圾分类任务

改进点：
1. 使用残差式 CBAM，降低对预训练特征分布的破坏
2. 支持在多个 backbone 层后插入 CBAM
3. 训练参数与原始 YOLOv8n-cls 对照组保持一致，保证公平对比
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# 导入统一路径配置
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_config import DATA_PATH, CLASSIFY_PATH

def set_seed(seed=42):
    """固定随机种子，增强实验可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ChannelAttention(nn.Module):
    """CBAM 通道注意力模块"""

    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        hidden_planes = max(in_planes // ratio, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_planes, in_planes, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """CBAM 空间注意力模块"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)

        return self.sigmoid(x)

class CBAM(nn.Module):
    """
    残差式 CBAM 模块

    普通 CBAM：
        out = attention(x) * x

    这里使用残差形式：
        out = x + gamma * attention(x)

    这样可以减少新增注意力模块对 YOLOv8 预训练特征的破坏。
    """

    def __init__(self, in_planes, ratio=4, kernel_size=7, gamma_init=0.1):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def forward(self, x):
        identity = x

        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out

        out = identity + self.gamma * out

        return out

class WrapperWithCBAM(nn.Module):
    """将原 YOLOv8 层与 CBAM 包装在一起"""

    def __init__(self, base_layer, channels):
        super(WrapperWithCBAM, self).__init__()
        self.base = base_layer
        self.cbam = CBAM(channels)

    def forward(self, x):
        x = self.base(x)
        x = self.cbam(x)
        return x

def get_layer_output_channels(model_module, layer_idx, imgsz=320):
    """
    通过一次前向传播自动推断指定层输出通道数
    """

    temp_input = torch.randn(1, 3, imgsz, imgsz)

    with torch.no_grad():
        for i in range(layer_idx + 1):
            temp_input = model_module[i](temp_input)

    out_channels = temp_input.shape[1]

    return out_channels

def add_cbam_to_yolov8_cls(model, layer_indices=(4, 6, 8), imgsz=320):
    """
    给 YOLOv8 分类模型添加 CBAM 注意力模块

    参数：
        model: ultralytics.YOLO 模型
        layer_indices: 插入 CBAM 的层索引
        imgsz: 输入图像尺寸，用于自动推断通道数
    """

    model_module = model.model.model

    print("\n🔍 YOLOv8-cls 模型结构：")
    for idx, layer in enumerate(model_module):
        print(f"Layer {idx}: {layer.__class__.__name__}")

    print("\n🔧 开始插入 CBAM 模块...")

    for layer_idx in layer_indices:
        if layer_idx >= len(model_module):
            print(f"⚠️ 跳过 Layer {layer_idx}，因为模型没有这一层")
            continue

        out_channels = get_layer_output_channels(
            model_module=model_module,
            layer_idx=layer_idx,
            imgsz=imgsz
        )

        original_layer = model_module[layer_idx]
        model_module[layer_idx] = WrapperWithCBAM(
            base_layer=original_layer,
            channels=out_channels
        )

        print(f"✅ 已在 Layer {layer_idx} 后添加 CBAM，输出通道数: {out_channels}")

    return model

def main():
    set_seed(42)

    print("=" * 70)
    print("YOLOv8n-cls + CBAM 注意力机制训练")
    print("=" * 70)

    data_path = DATA_PATH
    imgsz = 320

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集路径不存在: {data_path}")

    print(f"\n📂 数据集路径: {data_path}")
    print("🎯 目标类别数: 4")

    print("\n📥 加载 YOLOv8n-cls 预训练模型...")
    model = YOLO("yolov8n-cls.pt")

    print("\n🔧 添加 CBAM 注意力模块...")
    model = add_cbam_to_yolov8_cls(
        model=model,
        layer_indices=(4, 6, 8),
        imgsz=imgsz
    )

    print("\n🚀 开始训练 YOLOv8n-cls + CBAM...")

    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=imgsz,
        batch=32,
        name="new_cbam",
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
        mixup=0.15,

        patience=20,
        verbose=True,
        save=True,
        save_period=10,
        plots=True
    )

    print("\n" + "=" * 70)
    print("✅ YOLOv8n-cls + CBAM 训练完成！")
    print("=" * 70)

    save_dir = os.path.join(CLASSIFY_PATH, "new_cbam")
    os.makedirs(save_dir, exist_ok=True)

    training_info = {
        "model_type": "YOLOv8n-cls + Residual CBAM",
        "pretrained": "yolov8n-cls.pt",
        "cbam_layers": [4, 6, 8],
        "cbam_type": "Residual CBAM",
        "epochs": 100,
        "imgsz": imgsz,
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
            "mixup": 0.15
        },
        "note": "在 YOLOv8n-cls 的中高层特征后加入残差式 CBAM，用于垃圾分类任务。"
    }

    info_path = os.path.join(save_dir, "training_info.json")

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)

    print(f"📁 最佳模型保存路径: {save_dir}/weights/best.pt")
    print(f"💾 训练信息保存到: {info_path}")

if __name__ == "__main__":
    main()