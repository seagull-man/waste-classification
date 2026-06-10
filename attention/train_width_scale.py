#!/usr/bin/env python3
"""
宽度缩放（Width Scaling）+ SE 注意力 = 正确的小模型优化方案

原理：
  YOLOv8 通过 width_multiple / depth_multiple 控制模型大小。
  减小这两个参数 → 整个模型等比例缩小 → 层间依赖不变 → 真正提速。

对比之前的 torch.nn.utils.prune 方案：
  prune 方案: 简单裁通道 → 残差连接断裂 → 参数量不变、FPS 反而下降 ❌
  宽度缩放:   整体等比例缩小 → 结构完整 → 真正减少参数、提升 FPS ✅

用法：
  python train_width_scale.py
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from ultralytics import YOLO


# ===================== SE 注意力模块（复用你已有的） =====================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEWrapper(nn.Module):
    """包装模块：原模块 → SEBlock"""
    def __init__(self, module, reduction=16):
        super().__init__()
        self.module = module
        self.se = None
        self.reduction = reduction
        self.i = getattr(module, "i", None)
        self.f = getattr(module, "f", -1)
        self.type = getattr(module, "type", module.__class__.__name__)
        self.np = getattr(module, "np", 0)

    def _build_se(self, x):
        c = x.shape[1]
        self.se = SEBlock(c, self.reduction).to(device=x.device, dtype=x.dtype)

    def forward(self, x):
        x = self.module(x)
        if self.se is None:
            self._build_se(x)
        return self.se(x)


# ===================== 配置 =====================

BASE_DIR    = Path("/root/autodl-tmp/waste-classification/attention")
DATA_PATH   = "/root/autodl-tmp/garbage_4cls"
DEVICE      = "cuda:0"
IMAGE_SIZE  = 320
BATCH_SIZE  = 32
EPOCHS      = 80

# 三个变体：不同宽度的 YAML 配置
VARIANTS = [
    {
        "name": "yolov8_slim_se",
        "yaml": str(BASE_DIR / "yolov8-slim-se.yaml"),
        "desc": "YOLOv8-Slim + SE",
        "scales": "w=0.25, d=0.20",
    },
    {
        "name": "yolov8_micro_se",
        "yaml": str(BASE_DIR / "yolov8-micro-se.yaml"),
        "desc": "YOLOv8-Micro + SE",
        "scales": "w=0.20, d=0.15",
    },
]

# 同时训练一个不带注意力的 slim 版本，用于对比
BASELINE_VARIANT = {
    "name": "yolov8_slim_baseline",
    "yaml": str(BASE_DIR / "yolov8-slim-se.yaml"),
    "desc": "YOLOv8-Slim (无注意力)",
    "scales": "w=0.25, d=0.20",
    "no_attention": True,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def add_se_attention(model, mode="late", reduction=16):
    """在 YOLOv8 后两层 C2f 后加入 SE 注意力"""
    modules = model.model.model
    c2f_indices = [i for i, m in enumerate(modules) if m.__class__.__name__ == "C2f"]
    selected = c2f_indices[-2:] if mode == "late" else c2f_indices
    for idx in selected:
        modules[idx] = SEWrapper(modules[idx], reduction=reduction)
    return selected


def init_se_layers(model, imgsz=320, device="cuda:0"):
    """用假输入初始化 SE 层"""
    model.model.to(device)
    model.model.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz).to(device)
    with torch.no_grad():
        model.model(dummy)
    model.model.train()


def train_variant(variant):
    """训练一个宽度缩放变体"""
    set_seed(42)

    print("\n" + "=" * 70)
    print(f"训练: {variant['desc']}")
    print(f"YAML:  {variant['yaml']}")
    print(f"缩放:  {variant['scales']}")
    print("=" * 70)

    # Step 1: 用自定义 YAML 构建更小的模型结构
    print("\n[1] 从 YAML 构建瘦身模型结构...")
    model = YOLO(variant["yaml"])

    # Step 2: 加载 yolov8n-cls.pt 的预训练权重（部分匹配）
    print("[2] 加载 yolov8n-cls.pt 预训练权重（部分迁移）...")
    model = model.load("yolov8n-cls.pt")
    print("    ✓ 匹配的层从预训练权重初始化，新增/变化层随机初始化")

    # Step 3: 添加 SE 注意力（除非是 baseline）
    if not variant.get("no_attention", False):
        print("[3] 添加 SE 注意力机制 (late mode)...")
        se_layers = add_se_attention(model, mode="late", reduction=16)
        init_se_layers(model, imgsz=IMAGE_SIZE, device=DEVICE)
        print(f"    SE 插入层: {se_layers}")
    else:
        print("[3] 跳过注意力（baseline 对照组）")

    # Step 4: 训练
    print(f"\n[4] 开始训练 ({EPOCHS} epochs)...")
    model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=str(BASE_DIR / "runs" / "classify"),
        name=variant["name"],
        device=DEVICE,
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
    )

    best_pt = (BASE_DIR / "runs" / "classify" / variant["name"] /
               "weights" / "best.pt")
    print(f"\n✅ {variant['desc']} 训练完成！")
    print(f"📁 模型: {best_pt}")

    return best_pt


# ===================== 主流程 =====================

def main():
    print("=" * 70)
    print("宽度缩放（Width Scaling）+ SE 注意力 优化方案")
    print("=" * 70)
    print(f"\n原始 yolov8n: scales=[0.33, 0.25, 1024] → ~1.45M params")
    print(f"Slim 版:       scales=[0.25, 0.20, 1024] → ~0.95M params")
    print(f"Micro 版:      scales=[0.20, 0.15, 1024] → ~0.65M params")
    print()

    # 选择要训练的变体
    print("请选择要训练的变体:")
    print("  1 - 全部（Slim+SE + Micro+SE + Slim Baseline）")
    print("  2 - 仅 Slim+SE")
    print("  3 - 仅 Micro+SE")
    print("  4 - 仅 Slim Baseline（无注意力）")

    try:
        choice = input("\n输入选择 (1-4) [默认=2]: ").strip() or "2"
    except (EOFError, KeyboardInterrupt):
        choice = "2"

    variants_to_train = []
    if choice == "1":
        variants_to_train = VARIANTS + [BASELINE_VARIANT]
    elif choice == "2":
        variants_to_train = [VARIANTS[0]]  # Slim+SE
    elif choice == "3":
        variants_to_train = [VARIANTS[1]]  # Micro+SE
    elif choice == "4":
        variants_to_train = [BASELINE_VARIANT]
    else:
        variants_to_train = [VARIANTS[0]]

    results = {}
    for v in variants_to_train:
        best_pt = train_variant(v)
        results[v["name"]] = str(best_pt)

    # 保存结果索引
    results_path = BASE_DIR / "width_scale_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "date": datetime.now().isoformat(),
            "variants": [
                {**v, "best_pt": results.get(v["name"], "N/A")}
                for v in variants_to_train
            ]
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果索引已保存至: {results_path}")

    print("\n" + "=" * 70)
    print("全部训练完成！接下来运行对比脚本查看效果:")
    print("  python Doubalcomparison.py")
    print("=" * 70)


if __name__ == "__main__":
    main()