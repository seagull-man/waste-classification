#!/usr/bin/env python3
"""
专门测试四个模型 FPS 的脚本
"""

import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO


# ===================== 配置 =====================
BASE_DIR = Path("/root/autodl-tmp/waste-classification/attention/runs/classify")

# 模型配置：(目录名, 显示名称)
MODELS = [
    ("waste_cls_original", "YOLOv8n-cls (Original)"),
    ("waste_cls_yolov8n_se_real", "YOLOv8n-cls + SE"),
    ("waste_cls_yolov8n_eca", "YOLOv8n-cls + ECA"),
    ("new_cbam_fixed", "YOLOv8n-cls + CBAM"),
]

# 测试参数
WARMUP_ITERS = 50    # 预热迭代次数
TEST_ITERS = 500     # 正式测试迭代次数
BATCH_SIZE = 1       # 批次大小
IMAGE_SIZE = 320     # 输入图像大小
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def measure_fps(model, imgsz=320, warmup_iters=50, test_iters=500):
    """
    测量模型的 FPS 和推理时间
    """
    # 构造随机输入
    dummy_input = torch.randn(BATCH_SIZE, 3, imgsz, imgsz).to(DEVICE)
    
    # 获取内部模型
    if hasattr(model, "model") and model.model is not None:
        net = model.model
    else:
        net = model
    
    net.eval()
    net.to(DEVICE)
    
    # 预热
    print(f"  预热 {warmup_iters} 轮...")
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = net(dummy_input)
    
    # 同步 CUDA
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    
    # 正式测试
    print(f"  正式测试 {test_iters} 轮...")
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(test_iters):
            _ = net(dummy_input)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    total_time = end - start
    avg_time_ms = (total_time / test_iters) * 1000
    fps = test_iters / total_time
    
    return fps, avg_time_ms


def main():
    print(f"{'='*60}")
    print(f"模型 FPS 测试")
    print(f"设备: {DEVICE}")
    print(f"输入尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"预热轮数: {WARMUP_ITERS}, 测试轮数: {TEST_ITERS}")
    print(f"{'='*60}\n")
    
    results = []
    
    for model_dir, display_name in MODELS:
        model_path = BASE_DIR / model_dir / "weights" / "best.pt"
        if not model_path.exists():
            print(f"[跳过] {display_name}: 未找到模型文件 {model_path}\n")
            continue
        
        print(f"正在测试: {display_name}")
        print(f"模型路径: {model_path}")
        
        # 加载模型
        model = YOLO(str(model_path))
        
        # 测试 FPS
        fps, avg_time_ms = measure_fps(
            model, 
            imgsz=IMAGE_SIZE,
            warmup_iters=WARMUP_ITERS,
            test_iters=TEST_ITERS
        )
        
        print(f"  FPS: {fps:.2f}")
        print(f"  平均推理时间: {avg_time_ms:.3f} ms\n")
        
        results.append({
            "模型": display_name,
            "FPS": fps,
            "推理时间 (ms)": avg_time_ms
        })
    
    # 打印对比表格
    print(f"{'='*60}")
    print(f"{'模型':<30} | {'FPS':>10} | {'推理时间 (ms)':>15}")
    print(f"{'-'*60}")
    
    for result in results:
        print(f"{result['模型']:<30} | {result['FPS']:>10.2f} | {result['推理时间 (ms)']:>15.3f}")
    
    print(f"{'='*60}\n")
    
    # 找出最快和最慢的模型
    if results:
        fastest = max(results, key=lambda x: x['FPS'])
        slowest = min(results, key=lambda x: x['FPS'])
        
        print(f"最快的模型: {fastest['模型']} ({fastest['FPS']:.2f} FPS)")
        print(f"最慢的模型: {slowest['模型']} ({slowest['FPS']:.2f} FPS)")
        
        if len(results) > 1:
            speed_drop = ((slowest['FPS'] - fastest['FPS']) / fastest['FPS']) * 100
            print(f"速度差异: {abs(speed_drop):.1f}%")


if __name__ == "__main__":
    main()
