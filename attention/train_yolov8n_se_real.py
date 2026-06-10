import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

import ultralytics.nn.tasks as tasks_module


class SEBlock(nn.Module):
    """SE 通道注意力模块 (兼容两种初始化)
    1. parse_model 特殊处理触发 → args = [ch[f], *args] → SEBlock(128, 16)
    2. 未触发 → args = [16] → SEBlock(16)
    两种情况都在第一次 forward 时根据真实输入通道数动态初始化，确保维度正确。
    """
    def __init__(self, *args):
        super().__init__()
        reduction = 16
        for arg in args:
            if isinstance(arg, int) and arg > 0:
                reduction = arg
                break
        self._reduction = reduction
        self._initialized = False

    def _initialize(self, channels):
        hidden_channels = max(channels // self._reduction, 1)
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid()
        )
        # 移动到当前设备
        if hasattr(self, 'device'):
            self.to(self.device)
        self._initialized = True

    def forward(self, x):
        if not self._initialized:
            self._initialize(x.size(1))
            # 同时对齐设备和数据类型（支持 AMP 混合精度）
            self.to(device=x.device, dtype=x.dtype)
        identity = x
        b, c, _, _ = x.shape
        weight = self.avg_pool(x).view(b, c)
        weight = self.fc(weight).view(b, c, 1, 1)
        refined = x * weight
        return identity + self.gamma * (refined - identity)


tasks_module.__dict__['SEBlock'] = SEBlock


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def check_se_params(model, title=""):
    print("\n" + "=" * 70)
    print(f"检查 SEBlock 参数: {title}")
    print("=" * 70)

    se_keys = [
        k for k in model.model.state_dict().keys()
        if "fc" in k or "gamma" in k
    ]

    print(f"SE 相关参数数量: {len(se_keys)}")
    for key in se_keys:
        print(key)

    if len(se_keys) > 0:
        print("✅ 当前模型中已检测到 SEBlock 参数")
    else:
        print("❌ 当前模型中没有检测到 SEBlock 参数")

def main():
    set_seed(42)

    print("=" * 70)
    print("YOLOv8n-cls + SEBlock 真实结构训练")
    print("=" * 70)

    data_path = "/root/autodl-tmp/garbage_4cls"
    yaml_path = "/root/autodl-tmp/waste-classification/attention/yolov8n-cls-se.yaml"

    model = YOLO(yaml_path)
    model.load("yolov8n-cls.pt")

    check_se_params(model, "加载预训练权重后，训练前")

    print("\n🚀 开始训练 SE-YOLOv8n 模型...")
    print(f"📂 数据集路径: {data_path}")
    print(f"🧩 模型结构文件: {yaml_path}")
    print("🎯 目标类别数: 4")

    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=320,
        batch=32,
        name="waste_cls_yolov8n_se_real",
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
        mixup=0.05,
        patience=20,
        verbose=True,
        save=True,
        save_period=10,
        plots=True
    )

    save_dir = "runs/classify/waste_cls_yolov8n_se_real"
    os.makedirs(save_dir, exist_ok=True)

    training_info = {
    "model_type": "Lightweight YOLOv8n-cls + SEBlock",
    "pretrained": "yolov8n-cls.pt",
    "yaml": yaml_path,
    "attention": "Residual SEBlock",
    "optimization_method": "Reduce high-level C2f repeats and add SEBlock",
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
        "mixup": 0.05
    },
    "note": "减少网络后段 C2f 模块重复次数以降低计算量，并引入残差式 SEBlock 对高层通道特征进行重标定，在相同输入尺寸下兼顾推理速度与分类准确率。"
}

    info_path = os.path.join(save_dir, "training_info.json")

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("✅ SE-YOLOv8n 模型训练完成！")
    print("=" * 70)
    print(f"📁 最佳模型保存路径: {save_dir}/weights/best.pt")
    print(f"💾 训练信息保存到: {info_path}")

if __name__ == "__main__":
    main()
