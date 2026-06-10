import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    用于增强重要通道特征，抑制无效通道特征。
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden_channels = max(channels // reduction, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)

        return x * y

class SEWrapper(nn.Module):
    """
    包装 YOLOv8 中已有模块：
    原模块输出 -> SEBlock -> 输出

    这样可以保留原始 YOLOv8n-cls 的预训练权重。
    """

    def __init__(self, module, reduction=16):
        super().__init__()

        self.module = module
        self.reduction = reduction
        self.se = None

        # 保留 Ultralytics 模型推理所需属性
        self.i = getattr(module, "i", None)
        self.f = getattr(module, "f", -1)
        self.type = getattr(module, "type", module.__class__.__name__)
        self.np = getattr(module, "np", 0)

    def _build_se(self, x):
        channels = x.shape[1]
        self.se = SEBlock(channels, self.reduction).to(
            device=x.device,
            dtype=x.dtype
        )

    def forward(self, x):
        x = self.module(x)

        if self.se is None:
            self._build_se(x)

        return self.se(x)

def add_se_attention(model, mode="late", reduction=16):
    """
    给 YOLOv8n-cls 加入 SE 注意力机制。

    mode:
        late: 只在最后两个 C2f 模块后加入 SE，推荐，速度损失较小
        all:  在所有 C2f 模块后加入 SE，可能精度更高，但 FPS 略低
    """

    modules = model.model.model

    c2f_indices = [
        index for index, module in enumerate(modules)
        if module.__class__.__name__ == "C2f"
    ]

    if mode == "late":
        selected_indices = c2f_indices[-2:]
    elif mode == "all":
        selected_indices = c2f_indices
    else:
        raise ValueError("mode 只能是 'late' 或 'all'")

    for index in selected_indices:
        modules[index] = SEWrapper(modules[index], reduction=reduction)

    return selected_indices

def initialize_se_layers(model, imgsz=320, device="cuda:0"):
    """
    用一次假输入初始化 SEBlock。
    这样可以保证训练开始前 SE 参数已经注册到模型中。
    """

    model.model.to(device)
    model.model.eval()

    dummy_input = torch.zeros(1, 3, imgsz, imgsz).to(device)

    with torch.no_grad():
        model.model(dummy_input)

    model.model.train()

def main():
    set_seed(42)

    print("=" * 70)
    print("YOLOv8 + SE-Block 注意力机制训练")
    print("=" * 70)

    data_path = "/root/autodl-tmp/garbage_4cls"

    model = YOLO("yolov8n-cls.pt")

    se_mode = "late"
    se_reduction = 16

    selected_se_layers = add_se_attention(
        model=model,
        mode=se_mode,
        reduction=se_reduction
    )

    initialize_se_layers(
        model=model,
        imgsz=320,
        device="cuda:0"
    )

    print("\n🚀 开始训练 SE-YOLOv8 模型...")
    print(f"📂 数据集路径: {data_path}")
    print("🎯 目标类别数: 4")
    print(f"🧠 SE 插入模式: {se_mode}")
    print(f"📌 SE 插入层索引: {selected_se_layers}")
    print(f"⚙️ SE reduction: {se_reduction}")

    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=320,
        batch=32,
        name="waste_cls_yolov8n_se",
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
        mosaic=1.0,
        mixup=0.05,
        patience=20,
        verbose=True,
        save=True,
        save_period=10,
        plots=True
    )

    print("\n" + "=" * 70)
    print("✅ SE-YOLOv8 模型训练完成！")
    print("=" * 70)

    save_dir = "runs/classify/waste_cls_yolov8n_se"
    os.makedirs(save_dir, exist_ok=True)

    training_info = {
        "model_type": "YOLOv8n-cls + SEBlock",
        "pretrained": "yolov8n-cls.pt",
        "attention": "SEBlock",
        "se_mode": se_mode,
        "se_reduction": se_reduction,
        "se_insert_layers": selected_se_layers,
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
            "mosaic": 1.0,
            "mixup": 0.05
        },
        "note": "在 YOLOv8n-cls 的后两层 C2f 后加入 SEBlock，保留原始预训练权重，提升通道特征表达能力。"
    }

    info_path = os.path.join(save_dir, "training_info.json")

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)

    print(f"📁 最佳模型保存路径: {save_dir}/weights/best.pt")
    print(f"💾 训练信息保存到: {info_path}")

if __name__ == "__main__":
    main()