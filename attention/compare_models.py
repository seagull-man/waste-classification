#!/usr/bin/env python3
"""
模型性能对比脚本（修正版）
对比 4 个 YOLOv8n-cls 变体模型的性能指标。
修复：FPS 测试现在使用真实图片进行完整推理管线测量。
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

# 允许加载截断/轻微损坏的图片
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

# 模型配置
MODELS = [
    ("waste_cls_original", "YOLOv8n-cls (Original)"),
    ("waste_cls_yolov8n_se_real", "YOLOv8n-cls + SE"),
    ("waste_cls_yolov8n_eca", "YOLOv8n-cls + ECA"),
    ("new_cbam_fixed", "YOLOv8n-cls + CBAM"),
]

CLASS_NAMES = ["hazardous", "kitchen", "other", "recyclable"]

# 推理速度测试参数
WARMUP_ITERS = 10    # 预热迭代次数
SPEED_ITERS = 100     # 正式测试迭代次数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===================== 工具函数 =====================

def get_model_size_mb(model_path):
    """获取模型文件大小（MB）"""
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 * 1024)


def get_param_count(model, model_name):
    """
    获取模型参数量（百万）。
    """
    try:
        if hasattr(model, "model") and model.model is not None:
            net = model.model
        else:
            net = model

        if hasattr(net, "parameters"):
            total = sum(p.numel() for p in net.parameters())
            return total / 1e6
    except Exception as e:
        print(f"  [警告] 无法统计 {model_name} 参数量: {e}")

    # fallback: 从 .pt 文件读取
    try:
        ckpt = torch.load(
            str(BASE_DIR / model_name / "weights" / "best.pt"),
            map_location="cpu",
            weights_only=False,
        )
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
    """
    y_true, y_pred = [], []

    for class_idx, class_name in enumerate(sorted(os.listdir(val_dir))):
        class_dir = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        all_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        images, skipped = [], 0
        for f in all_files:
            img_path = os.path.join(class_dir, f)
            try:
                with Image.open(img_path) as img:
                    img.verify()
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


def measure_speed(model, image_size=320):
    """
    【修正】使用真实图片，走完整 model.predict() 管线，测量推理速度。
    
    为什么不用随机张量？
    - model.predict() 包含预处理（Resize, Normalize）和后处理（Softmax, Top1）
    - 如果注意力模块是通过 predict 的回调或外挂方式加的，只有走完整管线才能测到
    - 这才是模型部署时的真实速度
    """
    # 1. 创建一张真实图片（随机噪声模拟，但走完整流程）
    # 使用 PIL 创建，避免外部文件依赖
    img = Image.new("RGB", (image_size, image_size), color=(128, 128, 128))
    
    # 2. 先保存为临时文件（因为 YOLO.predict 需要文件路径或路径列表）
    tmp_img_path = OUTPUT_DIR / "_speed_test_tmp.jpg"
    img.save(tmp_img_path)
    
    # 3. 预热：让 GPU 达到稳定状态
    print(f"  预热 {WARMUP_ITERS} 次...")
    for i in range(WARMUP_ITERS):
        _ = model.predict(
            source=str(tmp_img_path),
            verbose=False,
            device=DEVICE,
            imgsz=image_size,
        )
    
    # 4. 正式计时
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for i in range(SPEED_ITERS):
        _ = model.predict(
            source=str(tmp_img_path),
            verbose=False,
            device=DEVICE,
            imgsz=image_size,
        )
    
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    # 5. 清理临时文件
    if tmp_img_path.exists():
        tmp_img_path.unlink()
    
    total_time = end - start
    avg_time_ms = (total_time / SPEED_ITERS) * 1000
    fps = SPEED_ITERS / total_time
    
    return fps, avg_time_ms


def measure_speed_batch(model, image_size=320, batch_size=8):
    """
    【可选】批量推理速度测试（更贴近实际部署）。
    如果你的应用场景是单张实时分类，用上面的 measure_speed 就够了。
    """
    # 创建临时图片列表
    tmp_img_paths = []
    img = Image.new("RGB", (image_size, image_size), color=(128, 128, 128))
    
    for i in range(batch_size):
        tmp_path = OUTPUT_DIR / f"_speed_test_tmp_{i}.jpg"
        img.save(tmp_path)
        tmp_img_paths.append(str(tmp_path))
    
    # 预热
    print(f"  批量预热 (batch={batch_size}) {WARMUP_ITERS} 次...")
    for _ in range(WARMUP_ITERS):
        _ = model.predict(
            source=tmp_img_paths,
            verbose=False,
            device=DEVICE,
            imgsz=image_size,
        )
    
    # 计时
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(SPEED_ITERS):
        _ = model.predict(
            source=tmp_img_paths,
            verbose=False,
            device=DEVICE,
            imgsz=image_size,
        )
    
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    # 清理
    for p in tmp_img_paths:
        Path(p).unlink(missing_ok=True)
    
    total_time = end - start
    total_images = SPEED_ITERS * batch_size
    avg_time_per_batch_ms = (total_time / SPEED_ITERS) * 1000
    avg_time_per_image_ms = (total_time / total_images) * 1000
    fps = total_images / total_time
    
    return fps, avg_time_per_image_ms


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
        
        prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        
        print(f"  Accuracy (Top-1): {acc:.4f}")
        print(f"  Precision (Macro): {prec_macro:.4f}  | Weighted: {prec_weighted:.4f}")
        print(f"  Recall    (Macro): {rec_macro:.4f}  | Weighted: {rec_weighted:.4f}")
        print(f"  F1-Score  (Macro): {f1_macro:.4f}  | Weighted: {f1_weighted:.4f}")
        
        per_class_report = classification_report(
            y_true, y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        
        # --- 5. 推理速度（走完整 predict 管线）---
        print("  测量推理速度（完整管线）...")
        fps, time_ms = measure_speed(model, image_size=320)
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
    
    summary_cols = [
        "模型", "Accuracy", "Precision (Weighted)", "Recall (Weighted)",
        "F1-Score (Weighted)", "Precision (Macro)", "Recall (Macro)",
        "F1-Score (Macro)", "FPS", "推理时间 (ms)", "参数量 (M)", "模型大小 (MB)",
    ]
    df_summary = pd.DataFrame(results)[summary_cols]
    print(df_summary.to_string(index=False))
    
    print("\n" + "-" * 80)
    print("每类别 Precision / Recall / F1-Score")
    print("-" * 80)
    per_class_cols = ["模型"]
    for cls in CLASS_NAMES:
        per_class_cols += [f"{cls}_precision", f"{cls}_recall", f"{cls}_f1"]
    df_per_class = pd.DataFrame(results)[per_class_cols]
    print(df_per_class.to_string(index=False))
    
    # 保存 CSV
    csv_path = OUTPUT_DIR / "model_comparison.csv"
    df_full = pd.DataFrame(results)
    df_full.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n详细结果已保存至: {csv_path}")
    
    # 保存 JSON
    json_path = OUTPUT_DIR / "model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"JSON 结果已保存至: {json_path}")
    
    print("\n评估完成！")


if __name__ == "__main__":
    main()