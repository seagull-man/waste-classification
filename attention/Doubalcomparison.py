#!/usr/bin/env python3
"""
模型性能对比脚本
对比 4 个 YOLOv8n-cls 变体模型的性能指标：
  - 分类准确率（Accuracy）
  - 精确率（Precision）
  - 召回率（Recall）
  - F1 分数（F1-Score）
  - 推理速度（FPS）
  - 单张图像推理时间（ms）
  - 参数量（M）
  - 模型大小（MB）
"""

import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from ultralytics import YOLO
from PIL import Image, ImageFile

# 允许加载截断/轻微损坏的图片（避免因个别损坏图片导致整个流程中断）
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ===================== 配置 =====================
BASE_DIR = Path("/root/autodl-tmp/waste-classification/attention/runs/classify")
VAL_DIR = Path("/root/autodl-tmp/garbage_4cls/val")
OUTPUT_DIR = Path("/root/autodl-tmp/waste-classification/attention")

# 模型配置：(目录名, 显示名称)
MODELS = [
    ("waste_cls_original", "YOLOv8n-cls (Original)"),
    ("waste_cls_yolov8n_se_real", "YOLOv8n-cls + SE"),
    ("yolov8_slim_se", "YOLOv8-Slim + SE (w=0.25)"),
    ("yolov8_micro_se", "YOLOv8-Micro + SE (w=0.20)"),
    ("waste_cls_yolov8n_eca", "YOLOv8n-cls + ECA"),
    ("new_cbam_fixed", "YOLOv8n-cls + CBAM"),
]

# 数据集类别名称（按字母序排列）
CLASS_NAMES = ["hazardous", "kitchen", "other", "recyclable"]

# 推理速度测试参数
WARMUP_ITERS = 10   # 预热迭代次数
SPEED_ITERS = 100    # 正式测试迭代次数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===================== 工具函数 =====================

def get_model_size_mb(model_path):
    """获取模型文件大小（MB）"""
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 * 1024)


def get_param_count(model, model_name):
    """
    获取模型参数量（百万）。
    优先使用 torch 统计；若模型为 YOLO 实例，则通过其内部 model 属性访问。
    """
    try:
        # YOLO 对象: model.model 是 nn.Module
        if hasattr(model, "model") and model.model is not None:
            net = model.model
        else:
            net = model

        if hasattr(net, "parameters"):
            total = sum(p.numel() for p in net.parameters())
            return total / 1e6
    except Exception as e:
        print(f"  [警告] 无法统计 {model_name} 参数量: {e}")

    # fallback: 直接检查 .pt 文件
    try:
        ckpt = torch.load(str(BASE_DIR / model_name / "weights" / "best.pt"),
                          map_location="cpu", weights_only=False)
        if "model" in ckpt and hasattr(ckpt["model"], "parameters"):
            total = sum(p.numel() for p in ckpt["model"].parameters())
            return total / 1e6
        if "ema" in ckpt and hasattr(ckpt["ema"], "parameters"):
            total = sum(p.numel() for p in ckpt["ema"].parameters())
            return total / 1e6
    except Exception:
        pass

    return None


def collect_predictions(model, val_dir):
    """
    在验证集上运行推理，收集预测标签和真实标签。
    返回 (y_true, y_pred) 两个列表。
    """
    y_true, y_pred = [], []

    # 遍历每个类别文件夹
    for class_idx, class_name in enumerate(sorted(os.listdir(val_dir))):
        class_dir = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # 收集所有图片，过滤损坏文件
        all_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        images, skipped = [], 0
        for f in all_files:
            img_path = os.path.join(class_dir, f)
            try:
                with Image.open(img_path) as img:
                    img.verify()  # 验证图片完整性
                images.append(img_path)
            except Exception:
                skipped += 1

        if not images:
            print(f"  [跳过] {class_name}: 无有效图片")
            continue

        if skipped > 0:
            print(f"  {class_name}: {len(images)} 张有效图片 (跳过 {skipped} 张损坏)")
        else:
            print(f"  推理 {class_name}: {len(images)} 张图片...")

        results = model.predict(
            source=images,
            verbose=False,
            device=DEVICE,
            stream=False,
        )

        for r in results:
            y_true.append(class_idx)
            y_pred.append(r.probs.top1)

    return np.array(y_true), np.array(y_pred)


def measure_speed(model, imgsz=320):
    """
    测量单张推理时间和 FPS。
    使用随机张量作为输入，排除 I/O 影响。
    """
    # 构造随机输入 (1, 3, imgsz, imgsz)
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(DEVICE)

    # 获取内部 nn.Module
    if hasattr(model, "model") and model.model is not None:
        net = model.model
    else:
        net = model

    net.eval()
    net.to(DEVICE)

    # 预热
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = net(dummy_input)

    # 同步 CUDA 后计时
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(SPEED_ITERS):
            _ = net(dummy_input)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    total_time = end - start
    avg_time_ms = (total_time / SPEED_ITERS) * 1000
    fps = SPEED_ITERS / total_time

    return fps, avg_time_ms


# ===================== 主流程 =====================

def main():
    results = []

    for model_dir, display_name in MODELS:
        model_path = BASE_DIR / model_dir / "weights" / "best.pt"
        if not model_path.exists():
            print(f"[跳过] {display_name}: 未找到模型文件 {model_path}")
            continue

        print(f"\n{'='*60}")
        print(f"正在评估: {display_name}")
        print(f"模型路径: {model_path}")
        print(f"{'='*60}")

        # --- 1. 模型大小 ---
        size_mb = get_model_size_mb(model_path)
        print(f"  模型大小: {size_mb:.2f} MB")

        # --- 2. 加载模型 ---
        model = YOLO(str(model_path))

        # --- 3. 参数量 ---
        params_m = get_param_count(model, model_dir)
        print(f"  参数量: {params_m:.2f} M" if params_m else "  参数量: N/A")

        # --- 4. 验证集推理 & 分类指标 ---
        print("  在验证集上推理...")
        y_true, y_pred = collect_predictions(model, VAL_DIR)

        acc = accuracy_score(y_true, y_pred)
        prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # 加权平均
        prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        print(f"  Accuracy (Top-1): {acc:.4f}")
        print(f"  Precision (Macro): {prec_macro:.4f}  | Weighted: {prec_weighted:.4f}")
        print(f"  Recall    (Macro): {rec_macro:.4f}  | Weighted: {rec_weighted:.4f}")
        print(f"  F1-Score  (Macro): {f1_macro:.4f}  | Weighted: {f1_weighted:.4f}")

        # 每类指标
        per_class_report = classification_report(
            y_true, y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        # --- 5. 推理速度 (使用随机张量) ---
        print("  测量推理速度...")
        fps, time_ms = measure_speed(model, imgsz=320)
        print(f"  FPS: {fps:.2f}")
        print(f"  单张推理时间: {time_ms:.2f} ms")

        # --- 汇总 ---
        results.append({
            "模型": display_name,
            "Accuracy": round(acc, 4),
            "Precision (Macro)": round(prec_macro, 4),
            "Precision (Weighted)": round(prec_weighted, 4),
            "Recall (Macro)": round(rec_macro, 4),
            "Recall (Weighted)": round(rec_weighted, 4),
            "F1-Score (Macro)": round(f1_macro, 4),
            "F1-Score (Weighted)": round(f1_weighted, 4),
            "FPS": round(fps, 2),
            "推理时间 (ms)": round(time_ms, 2),
            "参数量 (M)": round(params_m, 2) if params_m else "N/A",
            "模型大小 (MB)": round(size_mb, 2),
            # 每类详细指标
            **{
                f"{cls}_precision": round(per_class_report[cls]["precision"], 4)
                for cls in CLASS_NAMES
            },
            **{
                f"{cls}_recall": round(per_class_report[cls]["recall"], 4)
                for cls in CLASS_NAMES
            },
            **{
                f"{cls}_f1": round(per_class_report[cls]["f1-score"], 4)
                for cls in CLASS_NAMES
            },
        })

    # ===================== 输出结果 =====================

    print("\n\n" + "=" * 80)
    print(" " * 30 + "模型性能对比总览")
    print("=" * 80)

    # 核心指标表
    summary_cols = [
        "模型", "Accuracy", "Precision (Weighted)", "Recall (Weighted)",
        "F1-Score (Weighted)", "Precision (Macro)", "Recall (Macro)",
        "F1-Score (Macro)", "FPS", "推理时间 (ms)", "参数量 (M)", "模型大小 (MB)",
    ]
    df_summary = pd.DataFrame(results)[summary_cols]
    print(df_summary.to_string(index=False))

    # 每类指标表
    print("\n" + "-" * 80)
    print("每类别 Precision / Recall / F1-Score")
    print("-" * 80)
    per_class_cols = ["模型"]
    for cls in CLASS_NAMES:
        per_class_cols += [f"{cls}_precision", f"{cls}_recall", f"{cls}_f1"]
    df_per_class = pd.DataFrame(results)[per_class_cols]
    print(df_per_class.to_string(index=False))

    # 保存为 CSV
    csv_path = OUTPUT_DIR / "model_comparison.csv"
    df_full = pd.DataFrame(results)
    df_full.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n详细结果已保存至: {csv_path}")

    # 保存为 JSON（便于后续处理）
    json_path = OUTPUT_DIR / "model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"JSON 结果已保存至: {json_path}")

    print("\n评估完成！")


if __name__ == "__main__":
    main()
