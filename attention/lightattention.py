"""
YOLOv8 + SimAM 轻量级注意力
"""

import os
import torch
import torch.nn as nn
from ultralytics import YOLO


class SimAM(nn.Module):
    """无参数注意力模块"""
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)) ** 2
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


def add_simam_to_yolov8(model):
    """在YOLOv8中添加SimAM"""
    model_module = model.model.model
    layer_idx = 8  # 最后一层backbone
    
    if layer_idx >= len(model_module):
        return model
        
    original_layer = model_module[layer_idx]

    class WrapperWithSimAM(nn.Module):
        def __init__(self, base_layer):
            super().__init__()
            self.base = base_layer
            self.simam = SimAM()

        def forward(self, x):
            x = self.base(x)
            x = self.simam(x)
            return x

    wrapped = WrapperWithSimAM(original_layer)
    model_module[layer_idx] = wrapped
    
    return model


def main():
    print("="*70)
    print("YOLOv8 + SimAM 轻量级注意力")
    print("="*70)

    data_path = '/root/autodl-tmp/garbage_4cls'

    print("\n📥 加载预训练YOLOv8模型...")
    model = YOLO('yolov8n-cls.pt')

    print("\n🔧 添加SimAM注意力...")
    model = add_simam_to_yolov8(model)

    print("\n🚀 开始训练...")

    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=320,
        batch=32,
        name='waste_cls_simam',
        device='cuda:0',
        amp=True,
        workers=8,
        patience=30,
        verbose=True,
        save=True,
        save_period=10
    )

    print("\n" + "="*70)
    print("✅ 训练完成！")
    print("="*70)


if __name__ == "__main__":
    main()