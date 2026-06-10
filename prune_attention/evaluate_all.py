"""
统一评估脚本: 对比所有模型的准确率、FPS、参数量、FLOPs

评估项目:
  1. Top-1 / Top-5 Accuracy (验证集)
  2. FPS (batch=1 推理速度，含预热)
  3. 参数量 (Parameters)
  4. GFLOPs (计算量)
  5. 模型文件大小
  6. 架构验证（注意力模块是否存在）
"""
import os
import sys
import time
import json
import csv
import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_PATH, RUNS_DIR, IMGSZ, DEVICE,
    WARMUP_ITERS, TEST_ITERS, NUM_CLASSES,
)


def get_model_list() -> list[tuple[str, str]]:
    """扫描 RUNS_DIR 获取所有可用模型（含 baseline 和注意力模型）"""
    models = []
    if not os.path.exists(RUNS_DIR):
        return models

    for model_dir in sorted(os.listdir(RUNS_DIR)):
        model_pt = os.path.join(RUNS_DIR, model_dir, "train", "weights", "best.pt")

        # 跳过剪枝的中间目录
        if "_pruned" in model_dir:
            continue

        if os.path.exists(model_pt):
            models.append((model_dir, model_pt))

    return models


def check_attention_layers(model, model_name: str) -> dict:
    """检查模型中是否存在注意力模块"""
    net = model.model
    att_info = {"has_se": False, "has_cbam": False, "has_eca": False}

    for _, module in net.named_modules():
        cls = module.__class__.__name__
        if cls == "SEBlock":
            att_info["has_se"] = True
        elif cls == "CBAM":
            att_info["has_cbam"] = True
        elif cls == "ECABlock":
            att_info["has_eca"] = True

    return att_info


def evaluate_accuracy(model_path: str) -> dict:
    """评估模型的 Top-1 和 Top-5 准确率"""
    model = YOLO(model_path)
    results = model.val(data=DATA_PATH, imgsz=IMGSZ, device=DEVICE, verbose=False)

    acc = {"top1": 0, "top5": 0}
    if hasattr(results, 'top1'):
        acc["top1"] = round(float(results.top1), 2)
    if hasattr(results, 'top5'):
        acc["top5"] = round(float(results.top5), 2)

    return acc


def measure_fps(model_path: str) -> float:
    """测量模型的推理 FPS (batch=1，含 GPU 同步)"""
    model = YOLO(model_path)
    net = model.model
    net.eval()
    net.to(DEVICE)

    dummy_input = torch.randn(1, 3, IMGSZ, IMGSZ).to(DEVICE)

    # 预热
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = net(dummy_input)

    if "cuda" in str(DEVICE):
        torch.cuda.synchronize()

    # 计时
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(TEST_ITERS):
            _ = net(dummy_input)

    if "cuda" in str(DEVICE):
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return round(TEST_ITERS / elapsed, 1)


def count_parameters(model_path: str) -> tuple[int, int]:
    """统计参数量"""
    model = YOLO(model_path)
    net = model.model
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model_path: str) -> float:
    """估算 GFLOPs"""
    try:
        from thop import profile
        model = YOLO(model_path)
        net = model.model
        dummy = torch.randn(1, 3, IMGSZ, IMGSZ).to(DEVICE)
        flops, _ = profile(net, inputs=(dummy,), verbose=False)
        return round(flops / 1e9, 2)
    except ImportError:
        return -1


def get_model_size_mb(model_path: str) -> float:
    """模型文件大小 (MB)"""
    return round(os.path.getsize(model_path) / (1024 * 1024), 2)


def main():
    models = get_model_list()
    if not models:
        print("[ERROR] 没有找到任何训练好的模型!")
        print(f"[TIP] 请先运行 train_baseline.py 和 train_attention.py")
        return

    print(f"\n{'='*90}")
    print(f"找到 {len(models)} 个模型，开始全面评估")
    print(f"{'='*90}\n")

    results = []

    for name, path in models:
        print(f"\n{'─'*60}")
        print(f"[EVAL] {name}")
        print(f"  Path: {path}")

        # 架构检查
        try:
            model = YOLO(path)
            att_info = check_attention_layers(model, name)
            att_types = [k.replace("has_", "") for k, v in att_info.items() if v]
            if att_types:
                print(f"  Architecture: YOLOv8n + {', '.join(att_types).upper()}")
                print(f"    检测到注意力模块: ✓")
            else:
                print(f"  Architecture: YOLOv8n-cls (Baseline)")
        except Exception as e:
            print(f"  [WARN] 架构检查失败: {e}")

        # 准确率
        top1, top5 = 0, 0
        try:
            acc = evaluate_accuracy(path)
            top1 = acc.get("top1", 0)
            top5 = acc.get("top5", 0)
            print(f"  Top-1: {top1}%, Top-5: {top5}%")
        except Exception as e:
            print(f"  [WARN] 准确率评估失败: {e}")

        # FPS
        fps = 0
        try:
            fps = measure_fps(path)
            print(f"  FPS: {fps}")
        except Exception as e:
            print(f"  [WARN] FPS 测量失败: {e}")

        # 参数量
        total_p, trainable_p = 0, 0
        try:
            total_p, trainable_p = count_parameters(path)
            print(f"  Params: {total_p:,}")
        except Exception as e:
            print(f"  [WARN] 参数统计失败: {e}")

        # FLOPs
        flops = -1
        try:
            flops = estimate_flops(path)
            if flops > 0:
                print(f"  FLOPs: {flops} G")
        except Exception:
            pass

        # 文件大小
        size_mb = get_model_size_mb(path)
        print(f"  Size: {size_mb} MB")

        results.append({
            "model": name,
            "attention": ", ".join(att_types) if att_types else "None",
            "top1": top1,
            "top5": top5,
            "fps": fps,
            "params": total_p,
            "flops_g": flops if flops > 0 else "N/A",
            "size_mb": size_mb,
        })

    # 输出表格
    print(f"\n{'='*90}")
    print("评估结果汇总")
    print(f"{'='*90}\n")

    table = []
    for r in results:
        table.append([
            r["model"],
            r["attention"],
            f"{r['top1']:.2f}%",
            f"{r['top5']:.2f}%",
            f"{r['fps']:.1f}",
            f"{r['params']:,}",
            f"{r['flops_g']}G" if isinstance(r['flops_g'], float) else r['flops_g'],
            f"{r['size_mb']:.1f} MB",
        ])

    headers = ["Model", "Attention", "Top-1", "Top-5", "FPS", "Params", "FLOPs", "Size"]
    print(tabulate(table, headers=headers, tablefmt="grid"))

    # 保存 JSON
    output_json = os.path.join(RUNS_DIR, "evaluation_results.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 结果已保存: {output_json}")

    # 保存 CSV
    output_csv = os.path.join(RUNS_DIR, "evaluation_results.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] 结果已保存: {output_csv}")


if __name__ == "__main__":
    main()