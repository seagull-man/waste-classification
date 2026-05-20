"""
YOLOv8 + Residual SE-Block (残差旁路方案)
不破坏预训练权重，SE作为辅助分支存在
"""

import os
import json
import torch
import torch.nn as nn
from ultralytics import YOLO


class ResidualSE(nn.Module):
    """残差式SE注意力模块
    
    不改变原有特征流，而是在旁边加一个SE分支，
    输出时用可学习的加权融合
    """
    def __init__(self, channel, reduction=4):
        super(ResidualSE, self).__init__()
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
        # 可学习的融合权重
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # 原始特征不变
        identity = x
        
        # SE分支计算通道权重
        se_weight = self.se(x).view(x.size(0), x.size(1), 1, 1)
        
        # 加权融合：初始时gamma=0，相当于没有SE
        # 训练过程中如果SE有用，gamma会自动增大
        out = identity + self.gamma * x * se_weight
        
        return out


def add_residual_se_to_yolov8(model, reduction=4):
    """
    在YOLOv8 backbone中添加残差SE模块（作为旁路）
    """
    model_module = model.model.model
    
    # 只在第6和第8层后添加（关键特征提取层）
    target_layers = [6, 8]
    
    for layer_idx in target_layers:
        if layer_idx >= len(model_module):
            continue
            
        # 获取输出通道数
        temp_input = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            for i in range(layer_idx + 1):
                temp_input = model_module[i](temp_input)
        out_channels = temp_input.shape[1]
        
        # 包装：原层 + 残差SE旁路
        original_layer = model_module[layer_idx]
        
        class WrapperWithResidualSE(nn.Module):
            def __init__(self, base_layer, channels, red):
                super().__init__()
                self.base = base_layer
                self.residual_se = ResidualSE(channels, red)
                
            def forward(self, x):
                x = self.base(x)
                x = self.residual_se(x)
                return x
        
        wrapped = WrapperWithResidualSE(original_layer, out_channels, reduction)
        model_module[layer_idx] = wrapped
    
    return model


def main():
    print("="*70)
    print("YOLOv8 + Residual SE-Block (残差旁路方案)")
    print("="*70)

    data_path = '/root/autodl-tmp/garbage_4cls'

    print("\n📥 加载预训练YOLOv8模型...")
    model = YOLO('yolov8n-cls.pt')

    print("\n🔧 添加残差SE旁路（不破坏原结构）...")
    model = add_residual_se_to_yolov8(model, reduction=4)

    print("\n🚀 开始训练...")
    
    results = model.train(
        data=data_path,
        epochs=50,          # 多给点时间让SE学习
        imgsz=320,
        batch=64,
        name='waste_cls_residual_se',
        device='cuda:0',
        amp=True,
        workers=16,
        optimizer='auto',
        patience=25,         # 更大的耐心值
        verbose=True,
        save=True,
        save_period=10,
        warmup_epochs=5      # 更长的warmup
    )

    print("\n" + "="*70)
    print("✅ 训练完成！")
    print("="*70)


if __name__ == "__main__":
    main()