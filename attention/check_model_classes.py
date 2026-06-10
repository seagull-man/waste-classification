#!/usr/bin/env python3
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from ultralytics import YOLO
import torch
import torch.nn as nn

# 必须先注册自定义模块，否则加载 .pt 时会报错
import ultralytics.nn.tasks as tasks_module

class ECABlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        channels = None
        gamma = 2
        b = 1
        if args:
            for arg in args:
                if isinstance(arg, int) and arg > 0 and channels is None:
                    channels = arg
        if 'c1' in kwargs:
            channels = kwargs['c1']
        elif 'c2' in kwargs and channels is None:
            channels = kwargs['c2']
        elif 'channels' in kwargs:
            channels = kwargs['channels']
        if channels is None:
            self._initialized = False
            self._gamma = gamma
            self._b = b
            self.avg_pool = None
            self.conv = None
            self.sigmoid = None
        else:
            self._initialize(channels, gamma, b)

    def _initialize(self, channels, gamma=2, b=1):
        t = int(abs((torch.log2(torch.tensor(channels)) + b) / gamma))
        kernel_size = t if t % 2 == 1 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._initialized = True

    def forward(self, x):
        if not hasattr(self, '_initialized') or not self._initialized:
            self._initialize(x.size(1), getattr(self, '_gamma', 2), getattr(self, '_b', 1))
        y = self.avg_pool(x).squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

tasks_module.__dict__["ECABlock"] = ECABlock

# 加载模型
MODEL_PATH = os.path.join(CURRENT_DIR, "runs", "classify", "waste_cls_yolov8n_eca-2", "weights", "best.pt")

print("正在加载模型...")
try:
    model = YOLO(MODEL_PATH)
    print("模型加载成功！")
    print(f"\n模型类别信息：")
    print(f"model.names = {model.names}")
    print(f"\n类别数量：{len(model.names)}")
    
    # 打印索引和名称对应关系：
    print("\n类别索引 -> 名称对应关系：")
    for idx, name in model.names.items():
        print(f"  {idx} -> {name}")
except Exception as e:
    print(f"错误：{e}")
