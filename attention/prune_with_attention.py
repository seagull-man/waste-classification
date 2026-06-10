#!/usr/bin/env python3
"""
注意力机制 + 结构化剪枝 组合优化脚本

流程：
  1. 加载预训练 YOLOv8n-cls 模型
  2. 添加 SE 注意力机制
  3. 对 Conv2d 层进行 L1 结构化剪枝（按比例裁剪通道）
  4. 微调训练恢复精度
  5. 对比剪枝前后的 Accuracy / FPS / 参数量 / 模型大小

原理：
  - 注意力增强特征表达 → 模型有更多冗余可被安全裁剪
  - 结构化剪枝移除整个通道 → GPU 上真正加速
  - 微调恢复精度 → 精度不降的前提下减少参数量、提升 FPS
"""

import os
import json
import time
import copy
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from ultralytics import YOLO
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.metrics import accuracy_score


# ===================== 配置 =====================
DATA_PATH   = "/root/autodl-tmp/garbage_4cls"
VAL_DIR     = Path("/root/autodl-tmp/garbage_4cls/val")
OUTPUT_DIR  = Path("/root/autodl-tmp/waste-classification/attention/runs/classify/pruned_se")
BASE_MODEL  = "yolov8n-cls.pt"          # 基座模型
DEVICE      = "cuda:0"

# 剪枝参数
PRUNE_RATIO = 0.3        # 剪枝比例（30% 的通道被裁掉）
PRUNE_NORM  = 1           # 用 L1 范数衡量通道重要性（L2=2 也可）

# 微调参数
FINE_TUNE_EPOCHS = 30
FINE_TUNE_LR     = 0.0005
IMAGE_SIZE       = 320
BATCH_SIZE       = 32

# FPS 测试参数
WARMUP_ITERS = 50
SPEED_ITERS  = 200

CLASS_NAMES = ["hazardous", "kitchen", "other", "recyclable"]


# ===================== SE 注意力模块 =====================

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


# ===================== 工具函数 =====================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def add_se_attention(model, mode="late", reduction=16):
    """在 YOLOv8 后两层 C2f 后加入 SE 注意力"""
    modules = model.model.model
    c2f_indices = [i for i, m in enumerate(modules) if m.__class__.__name__ == "C2f"]
    selected = c2f_indices[-2:] if mode == "late" else c2f_indices
    for idx in selected:
        modules[idx] = SEWrapper(modules[idx], reduction=reduction)
    return selected


def init_se_layers(model, imgsz=320):
    """用假输入初始化 SE 层"""
    model.model.to(DEVICE)
    model.model.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz).to(DEVICE)
    with torch.no_grad():
        model.model(dummy)
    model.model.train()


def get_params_m(model):
    """统计参数量（百万）"""
    if hasattr(model, "model") and model.model is not None:
        net = model.model
    else:
        net = model
    return sum(p.numel() for p in net.parameters()) / 1e6


def get_model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def measure_fps(model, imgsz=320):
    """测量 FPS 和单张推理时间"""
    if hasattr(model, "model") and model.model is not None:
        net = model.model
    else:
        net = model
    dummy = torch.randn(1, 3, imgsz, imgsz).to(DEVICE)
    net.eval()
    net.to(DEVICE)

    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = net(dummy)

    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(SPEED_ITERS):
            _ = net(dummy)
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return SPEED_ITERS / elapsed, (elapsed / SPEED_ITERS) * 1000


def evaluate_accuracy(model):
    """在验证集上计算 Top-1 Accuracy"""
    y_true, y_pred = [], []
    val_dir = str(VAL_DIR)

    for class_idx, class_name in enumerate(sorted(os.listdir(val_dir))):
        class_dir = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        # 过滤损坏图片
        valid_images = []
        for p in images:
            try:
                with Image.open(p) as img:
                    img.verify()
                valid_images.append(p)
            except Exception:
                pass

        if not valid_images:
            continue

        results = model.predict(source=valid_images, verbose=False, device=DEVICE, stream=False)
        for r in results:
            y_true.append(class_idx)
            y_pred.append(r.probs.top1)

    return accuracy_score(y_true, y_pred)


# ===================== 结构化剪枝 =====================

def apply_structured_pruning(model, ratio=0.3, norm=1):
    """
    对模型中所有 Conv2d 层进行 Ln-norm 结构化剪枝。
    按 norm 范数衡量每个输出通道的重要性，裁掉最不重要的 ratio 比例通道。
    """
    net = model.model
    pruned_count = 0
    total_conv = 0

    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 1:
            total_conv += 1
            try:
                prune.ln_structured(
                    module, name="weight", amount=ratio, n=norm, dim=0
                )
                pruned_count += 1
            except Exception as e:
                print(f"  [跳过] {name}: {e}")

    print(f"\n  剪枝统计: {pruned_count}/{total_conv} 个 Conv2d 层被剪枝")

    # 使剪枝永久化（去掉 mask，只保留剪枝后的权重）
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except Exception:
                pass

    return model


# ===================== 主流程 =====================

def main():
    set_seed(42)

    print("=" * 70)
    print("注意力机制 + 结构化剪枝 组合优化")
    print(f"剪枝比例: {PRUNE_RATIO*100:.0f}% 通道  |  L{PRUNE_NORM} 范数")
    print("=" * 70)

    # ========== Step 1: 加载基座模型 + 加注意力 ==========
    print("\n[Step 1] 加载预训练 YOLOv8n-cls 模型...")
    model = YOLO(BASE_MODEL)
    params_before_se = get_params_m(model)
    print(f"  原始参数量: {params_before_se:.2f} M")

    print("\n[Step 2] 添加 SE 注意力机制 (late mode)...")
    se_layers = add_se_attention(model, mode="late", reduction=16)
    init_se_layers(model, imgsz=IMAGE_SIZE)
    params_after_se = get_params_m(model)
    print(f"  加 SE 后参数量: {params_after_se:.2f} M")
    print(f"  SE 插入层索引: {se_layers}")

    # ========== Step 2: 先训练一轮（注意力+基座联合训练） ==========
    print("\n[Step 3] 训练注意力增强模型（为剪枝做准备）...")
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    model.train(
        data=DATA_PATH,
        epochs=50,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name="pruned_se/pretrain",
        device=DEVICE,
        amp=True,
        workers=8,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        patience=15,
        verbose=True,
        save=True,
    )

    # 加载最佳权重
    best_pt = OUTPUT_DIR / "pretrain" / "weights" / "best.pt"
    print(f"\n  加载最佳权重: {best_pt}")
    model = YOLO(str(best_pt))

    # ----- 剪枝前评估 -----
    print("\n" + "=" * 70)
    print("剪枝前 - 性能评估")
    print("=" * 70)
    acc_before = evaluate_accuracy(model)
    fps_before, ms_before = measure_fps(model, imgsz=IMAGE_SIZE)
    params_before = get_params_m(model)
    size_before = get_model_size_mb(best_pt)

    print(f"  Accuracy : {acc_before:.4f}")
    print(f"  FPS      : {fps_before:.2f}")
    print(f"  推理时间 : {ms_before:.2f} ms")
    print(f"  参数量   : {params_before:.2f} M")
    print(f"  模型大小 : {size_before:.2f} MB")

    # ========== Step 3: 结构化剪枝 ==========
    print("\n" + "=" * 70)
    print(f"[Step 4] 应用 L{PRUNE_NORM} 结构化剪枝 (剪掉 {PRUNE_RATIO*100:.0f}% 通道)...")
    print("=" * 70)

    model = apply_structured_pruning(model, ratio=PRUNE_RATIO, norm=PRUNE_NORM)
    params_after_prune = get_params_m(model)
    print(f"  剪枝后参数量: {params_after_prune:.2f} M")
    print(f"  参数减少: {(1 - params_after_prune/params_before)*100:.1f}%")

    # ----- 剪枝后（微调前）评估 -----
    print("\n  剪枝后（微调前）评估...")
    acc_after_prune = evaluate_accuracy(model)
    print(f"  Accuracy : {acc_after_prune:.4f} (下降 {acc_before - acc_after_prune:.4f})")

    # ========== Step 4: 微调恢复精度 ==========
    print("\n" + "=" * 70)
    print(f"[Step 5] 微调剪枝后模型 ({FINE_TUNE_EPOCHS} epochs, lr={FINE_TUNE_LR})...")
    print("=" * 70)

    # 重新添加 SE 注意力（剪枝可能影响了 wrapper）
    model = add_se_attention(model, mode="late", reduction=16)

    model.train(
        data=DATA_PATH,
        epochs=FINE_TUNE_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name="pruned_se/finetune",
        device=DEVICE,
        amp=True,
        workers=8,
        optimizer="AdamW",
        lr0=FINE_TUNE_LR,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0001,
        warmup_epochs=2.0,
        patience=10,
        verbose=True,
        save=True,
    )

    # 加载微调后最佳权重
    finetune_pt = OUTPUT_DIR / "finetune" / "weights" / "best.pt"
    print(f"\n  加载微调后最佳权重: {finetune_pt}")
    model = YOLO(str(finetune_pt))

    # ========== Step 5: 最终评估 ==========
    print("\n" + "=" * 70)
    print("剪枝 + 微调后 - 最终性能评估")
    print("=" * 70)

    acc_after = evaluate_accuracy(model)
    fps_after, ms_after = measure_fps(model, imgsz=IMAGE_SIZE)
    params_after = get_params_m(model)
    size_after = get_model_size_mb(finetune_pt)

    print(f"  Accuracy : {acc_after:.4f}")
    print(f"  FPS      : {fps_after:.2f}")
    print(f"  推理时间 : {ms_after:.2f} ms")
    print(f"  参数量   : {params_after:.2f} M")
    print(f"  模型大小 : {size_after:.2f} MB")

    # ========== 汇总对比 ==========
    print("\n\n" + "=" * 80)
    print(" " * 25 + "剪枝前后对比汇总")
    print("=" * 80)
    print(f"{'指标':<20} {'剪枝前(SE)':>15} {'剪枝后(SE)':>15} {'变化':>15}")
    print("-" * 65)
    print(f"{'Accuracy':<20} {acc_before:>15.4f} {acc_after:>15.4f} {acc_after - acc_before:>+14.4f}")
    print(f"{'FPS':<20} {fps_before:>15.2f} {fps_after:>15.2f} {((fps_after/fps_before)-1)*100:>+13.1f}%")
    print(f"{'推理时间(ms)':<20} {ms_before:>15.2f} {ms_after:>15.2f} {((ms_after/ms_before)-1)*100:>+13.1f}%")
    print(f"{'参数量(M)':<20} {params_before:>15.2f} {params_after:>15.2f} {((params_after/params_before)-1)*100:>+13.1f}%")
    print(f"{'模型大小(MB)':<20} {size_before:>15.2f} {size_after:>15.2f} {((size_after/size_before)-1)*100:>+13.1f}%")
    print("=" * 80)

    # 保存结果
    report = {
        "method": "SE Attention + L1 Structured Pruning",
        "prune_ratio": PRUNE_RATIO,
        "prune_norm": f"L{PRUNE_NORM}",
        "fine_tune_epochs": FINE_TUNE_EPOCHS,
        "before": {
            "accuracy": round(acc_before, 4),
            "fps": round(fps_before, 2),
            "inference_time_ms": round(ms_before, 2),
            "params_m": round(params_before, 2),
            "size_mb": round(size_before, 2),
        },
        "after": {
            "accuracy": round(acc_after, 4),
            "fps": round(fps_after, 2),
            "inference_time_ms": round(ms_after, 2),
            "params_m": round(params_after, 2),
            "size_mb": round(size_after, 2),
        },
        "improvement": {
            "accuracy_change": round(acc_after - acc_before, 4),
            "fps_speedup": f"{((fps_after/fps_before)-1)*100:.1f}%",
            "params_reduction": f"{((1-params_after/params_before))*100:.1f}%",
            "size_reduction": f"{((1-size_after/size_before))*100:.1f}%",
        }
    }

    report_path = OUTPUT_DIR / "prune_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存至: {report_path}")

    print("\n✅ 注意力 + 剪枝组合优化完成！")
    print(f"📁 剪枝后模型: {finetune_pt}")


if __name__ == "__main__":
    main()