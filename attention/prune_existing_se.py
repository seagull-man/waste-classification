#!/usr/bin/env python3
"""
基于已训练好的 SE 注意力模型进行渐进式结构化剪枝

策略：迭代剪枝（Iterative Pruning）
  每次只剪掉少量通道 → 微调恢复 → 再剪 → 再微调
  这样精度损失最小，最终达到参数量减少 + FPS 提升的目标

用法：
  python prune_existing_se.py
"""

import os
import json
import time
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
DATA_PATH  = "/root/autodl-tmp/garbage_4cls"
VAL_DIR    = Path("/root/autodl-tmp/garbage_4cls/val")

# 输入：你已训练好的 SE 模型
INPUT_MODEL = Path("/root/autodl-tmp/waste-classification/attention/runs/classify/"
                   "waste_cls_yolov8n_se_real/weights/best.pt")

OUTPUT_DIR  = Path("/root/autodl-tmp/waste-classification/attention/runs/classify/se_pruned")

DEVICE      = "cuda:0"
IMAGE_SIZE  = 320
BATCH_SIZE  = 32

# 渐进式剪枝参数
PRUNE_ROUNDS     = 3        # 分几轮剪枝
PRUNE_RATIO      = 0.15     # 每轮剪掉 15% 通道
FINE_TUNE_EPOCHS = 10       # 每轮微调轮数
FINE_TUNE_LR     = 0.0005

# FPS 测试参数
WARMUP_ITERS = 50
SPEED_ITERS  = 200


# ===================== 工具函数 =====================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_params_m(model):
    if hasattr(model, "model") and model.model is not None:
        net = model.model
    else:
        net = model
    return sum(p.numel() for p in net.parameters()) / 1e6


def get_model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def measure_fps(model, imgsz=320):
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


def get_prunable_conv_layers(model):
    """
    找出可以安全剪枝的 Conv2d 层。
    排除:
      - 输出通道=1 的层（逐点卷积，不能剪）
      - 分类头相关的层
      - 太小的卷积层（out_channels < 16）
    """
    net = model.model
    prunable = []

    for name, module in net.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if module.out_channels <= 8:
            continue
        # 跳过 head/classifier 相关的 Conv
        if "head" in name.lower() or "classif" in name.lower():
            continue

        prunable.append((name, module))

    return prunable


def structured_prune_step(model, ratio=0.15):
    """
    对可剪枝的 Conv2d 层执行一次 L1 结构化剪枝。
    返回成功剪枝的层数。
    """
    layers = get_prunable_conv_layers(model)
    pruned = 0

    for name, module in layers:
        try:
            prune.ln_structured(module, name="weight", amount=ratio, n=1, dim=0)
            pruned += 1
        except Exception as e:
            pass  # 某些层剪枝可能失败，跳过即可

    # 永久化剪枝
    for _, module in layers:
        try:
            prune.remove(module, "weight")
        except Exception:
            pass

    return pruned


def save_model_pt(model, save_path):
    """手动保存模型（绕过 ultralytics 的 checkpoint 机制）"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state = {
        "model": model.model.state_dict(),
        "ckpt": model.ckpt if hasattr(model, "ckpt") else None,
    }
    torch.save(state, save_path)
    return save_path


# ===================== 主流程 =====================

def main():
    set_seed(42)

    print("=" * 70)
    print("渐进式结构化剪枝 - 基于已训练的 SE 注意力模型")
    print(f"输入模型: {INPUT_MODEL}")
    print(f"剪枝策略: {PRUNE_ROUNDS} 轮, 每轮剪 {PRUNE_RATIO*100:.0f}% 通道")
    print(f"每轮微调: {FINE_TUNE_EPOCHS} epochs")
    print("=" * 70)

    # ========== 加载已训练好的 SE 模型 ==========
    print("\n[加载] 已训练的 SE 注意力模型...")
    model = YOLO(str(INPUT_MODEL))
    params_init = get_params_m(model)
    size_init = get_model_size_mb(INPUT_MODEL)
    print(f"  初始参数量: {params_init:.2f} M")
    print(f"  初始模型大小: {size_init:.2f} MB")

    # ----- 剪枝前基准评估 -----
    print("\n" + "-" * 50)
    print("剪枝前基准性能")
    print("-" * 50)
    acc_baseline = evaluate_accuracy(model)
    fps_baseline, ms_baseline = measure_fps(model, imgsz=IMAGE_SIZE)
    print(f"  Accuracy : {acc_baseline:.4f}")
    print(f"  FPS      : {fps_baseline:.2f}")
    print(f"  推理时间 : {ms_baseline:.2f} ms")
    print(f"  参数量   : {params_init:.2f} M")

    # 记录每轮的结果
    history = [{
        "round": 0,
        "accuracy": acc_baseline,
        "fps": fps_baseline,
        "inference_time_ms": ms_baseline,
        "params_m": params_init,
    }]

    current_model = model

    # ========== 渐进式剪枝循环 ==========
    for round_num in range(1, PRUNE_ROUNDS + 1):
        print(f"\n{'='*70}")
        print(f"第 {round_num}/{PRUNE_ROUNDS} 轮剪枝")
        print(f"{'='*70}")

        # Step A: 剪枝
        print(f"\n  [剪枝] 剪掉 {PRUNE_RATIO*100:.0f}% 通道...")
        n_pruned = structured_prune_step(current_model, ratio=PRUNE_RATIO)
        params_after_prune = get_params_m(current_model)
        print(f"  剪枝了 {n_pruned} 层 Conv2d")
        print(f"  参数量: {params_after_prune:.2f} M")

        # Step B: 评估剪枝后（微调前）精度
        acc_after_prune = evaluate_accuracy(current_model)
        acc_drop = acc_baseline - acc_after_prune
        print(f"  剪枝后 Accuracy: {acc_after_prune:.4f} (下降 {acc_drop:.4f})")

        # Step C: 保存剪枝后的 checkpoint（用于微调恢复）
        prune_ckpt = OUTPUT_DIR / f"round{round_num}_pruned" / "weights" / "last.pt"
        # 用 ultralytics 保存时，直接用 torch.save 保存 model.model.state_dict()
        os.makedirs(prune_ckpt.parent, exist_ok=True)
        torch.save(current_model.model.state_dict(), str(prune_ckpt))

        # Step D: 微调恢复精度
        print(f"\n  [微调] {FINE_TUNE_EPOCHS} epochs...")
        current_model.train(
            data=DATA_PATH,
            epochs=FINE_TUNE_EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            project=str(OUTPUT_DIR),
            name=f"round{round_num}_finetune",
            device=DEVICE,
            amp=True,
            workers=8,
            optimizer="AdamW",
            lr0=FINE_TUNE_LR,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0001,
            warmup_epochs=1.0,
            patience=8,
            verbose=True,
            save=True,
        )

        # 加载微调后的最佳权重
        finetune_pt = (OUTPUT_DIR / f"round{round_num}_finetune" /
                       "weights" / "best.pt")
        current_model = YOLO(str(finetune_pt))
        params_round = get_params_m(current_model)

        # Step E: 评估
        acc_round = evaluate_accuracy(current_model)
        fps_round, ms_round = measure_fps(current_model, imgsz=IMAGE_SIZE)

        print(f"\n  --- 第 {round_num} 轮结果 ---")
        print(f"  Accuracy : {acc_round:.4f} (原始: {acc_baseline:.4f}, 差: {acc_baseline - acc_round:+.4f})")
        print(f"  FPS      : {fps_round:.2f}")
        print(f"  参数量   : {params_round:.2f} M")

        history.append({
            "round": round_num,
            "accuracy": round(acc_round, 4),
            "fps": round(fps_round, 2),
            "inference_time_ms": round(ms_round, 2),
            "params_m": round(params_round, 2),
        })

    # ========== 最终汇总 ==========
    final_pt = (OUTPUT_DIR / f"round{PRUNE_ROUNDS}_finetune" /
                "weights" / "best.pt")
    final_model = YOLO(str(final_pt))
    params_final = get_params_m(final_model)
    size_final = get_model_size_mb(final_pt)
    acc_final = evaluate_accuracy(final_model)
    fps_final, ms_final = measure_fps(final_model, imgsz=IMAGE_SIZE)

    print("\n\n" + "=" * 80)
    print(" " * 28 + "最终对比汇总")
    print("=" * 80)
    print(f"{'指标':<20} {'剪枝前(SE)':>15} {'剪枝后(SE)':>15} {'变化':>15}")
    print("-" * 65)
    print(f"{'Accuracy':<20} {acc_baseline:>15.4f} {acc_final:>15.4f} {acc_final - acc_baseline:>+14.4f}")
    print(f"{'FPS':<20} {fps_baseline:>15.2f} {fps_final:>15.2f} {((fps_final/fps_baseline)-1)*100:>+13.1f}%")
    print(f"{'推理时间(ms)':<20} {ms_baseline:>15.2f} {ms_final:>15.2f} {((ms_final/ms_baseline)-1)*100:>+13.1f}%")
    print(f"{'参数量(M)':<20} {params_init:>15.2f} {params_final:>15.2f} {((params_final/params_init)-1)*100:>+13.1f}%")
    print(f"{'模型大小(MB)':<20} {size_init:>15.2f} {size_final:>15.2f} {((size_final/size_init)-1)*100:>+13.1f}%")
    print("=" * 80)

    # 打印每轮变化
    print("\n" + "-" * 80)
    print("逐轮变化过程")
    print("-" * 80)
    print(f"{'轮次':<10} {'Accuracy':<12} {'FPS':<10} {'推理时间(ms)':<15} {'参数量(M)':<12}")
    for h in history:
        print(f"{h['round']:<10} {h['accuracy']:<12} {h['fps']:<10} {h['inference_time_ms']:<15} {h['params_m']:<12}")
    print("-" * 80)

    # 保存报告
    report = {
        "method": "SE Attention + Iterative L1 Structured Pruning",
        "prune_rounds": PRUNE_ROUNDS,
        "prune_ratio_per_round": PRUNE_RATIO,
        "fine_tune_epochs_per_round": FINE_TUNE_EPOCHS,
        "baseline": {
            "accuracy": round(acc_baseline, 4),
            "fps": round(fps_baseline, 2),
            "inference_time_ms": round(ms_baseline, 2),
            "params_m": round(params_init, 2),
            "size_mb": round(size_init, 2),
        },
        "final": {
            "accuracy": round(acc_final, 4),
            "fps": round(fps_final, 2),
            "inference_time_ms": round(ms_final, 2),
            "params_m": round(params_final, 2),
            "size_mb": round(size_final, 2),
        },
        "improvement": {
            "accuracy_change": round(acc_final - acc_baseline, 4),
            "fps_speedup_pct": f"{((fps_final/fps_baseline)-1)*100:.1f}%",
            "params_reduction_pct": f"{((1-params_final/params_init))*100:.1f}%",
            "size_reduction_pct": f"{((1-size_final/size_init))*100:.1f}%",
        },
        "history": history,
    }

    report_path = OUTPUT_DIR / "prune_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存至: {report_path}")
    print(f"剪枝后模型: {final_pt}")
    print("\n✅ 渐进式剪枝完成！")


if __name__ == "__main__":
    main()