import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# 首先导入 ultralytics.nn.tasks 模块，然后注入 ECABlock
import ultralytics.nn.tasks as tasks_module


class ECABlock(nn.Module):
    """Efficient Channel Attention
    用一维卷积替代全连接层，实现局部跨通道交互，参数增量几乎为零。
    此版本兼容多种参数初始化方式
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        # 处理各种可能的参数形式
        channels = None
        gamma = 2
        b = 1
        
        if args:
            # 尝试从参数中提取通道数
            for arg in args:
                if isinstance(arg, int) and arg > 0 and channels is None:
                    channels = arg
        
        # 如果有 kwargs，也尝试从中提取
        if 'c1' in kwargs:
            channels = kwargs['c1']
        elif 'c2' in kwargs and channels is None:
            channels = kwargs['c2']
        elif 'channels' in kwargs:
            channels = kwargs['channels']
        
        # 如果没有明确指定通道数，我们将在第一次前向传播时初始化
        if channels is None:
            self._initialized = False
            self._gamma = gamma
            self._b = b
            self.avg_pool = None
            self.conv = None
            self.sigmoid = None
        else:
            # 直接初始化
            self._initialize(channels, gamma, b)

    def _initialize(self, channels, gamma=2, b=1):
        """初始化模块的内部组件"""
        t = int(abs((torch.log2(torch.tensor(channels)) + b) / gamma))
        kernel_size = t if t % 2 == 1 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, '_initialized') or not self._initialized:
            self._initialize(x.size(1), getattr(self, '_gamma', 2), getattr(self, '_b', 1))
        y = self.avg_pool(x).squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


# 关键！直接在 ultralytics.nn.tasks 模块的全局作用域中添加 ECABlock
# 这样当 parse_model 调用 globals()['ECABlock'] 时就能找到了

# 方法：通过在 tasks_module 的全局字典中添加引用
tasks_module.__dict__['ECABlock'] = ECABlock

# 验证是否已添加
if 'ECABlock' in dir(tasks_module):
    print("✓ ECABlock successfully registered in ultralytics.nn.tasks!")
else:
    print("✗ Failed to register ECABlock")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def check_eca_params(model, title=""):
    print("\n" + "=" * 70)
    print(f"检查 ECABlock 参数: {title}")
    print("=" * 70)

    eca_keys = [
        k for k in model.model.state_dict().keys()
        if "conv.weight" in k and ("model.7" in k or "model.10" in k)
    ]

    print(f"ECA 参数数量: {len(eca_keys)}")

    for key in eca_keys:
        print(key, model.model.state_dict()[key].shape)

    if len(eca_keys) > 0:
        print("✅ 当前模型中已检测到 ECABlock 参数")
    else:
        print("❌ 当前模型中没有检测到 ECABlock 参数")

def main():
    set_seed(42)

    print("=" * 70)
    print("YOLOv8n-cls + ECABlock 真实结构训练")
    print("=" * 70)

    data_path = "/root/autodl-tmp/garbage_4cls"
    yaml_path = "/root/autodl-tmp/waste-classification/attention/yolov8n-cls-eca.yaml"

    model = YOLO(yaml_path)
    model.load("yolov8n-cls.pt")

    check_eca_params(model, "加载预训练权重后，训练前")

    print("\n🚀 开始训练 ECA-YOLOv8n 模型...")
    print(f"📂 数据集路径: {data_path}")
    print(f"🧩 模型结构文件: {yaml_path}")
    print("🎯 目标类别数: 4")

    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=320,
        batch=32,
        name="waste_cls_yolov8n_eca",
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

    save_dir = "runs/classify/waste_cls_yolov8n_eca"
    os.makedirs(save_dir, exist_ok=True)

    training_info = {
        "model_type": "YOLOv8n-cls + ECABlock",
        "pretrained": "yolov8n-cls.pt",
        "yaml": yaml_path,
        "attention": "ECABlock",
        "eca_insert_layers": ["model.7", "model.10"],
        "eca_kernel_size": 3,
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
        "note": "通过 Ultralytics 源码注册 ECABlock，并在 YAML 中定义模型结构，确保 ECABlock 被训练、保存和加载。"
    }

    info_path = os.path.join(save_dir, "training_info.json")

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("✅ ECA-YOLOv8n 模型训练完成！")
    print("=" * 70)
    print(f"📁 最佳模型保存路径: {save_dir}/weights/best.pt")
    print(f"💾 训练信息保存到: {info_path}")

if __name__ == "__main__":
    main()
