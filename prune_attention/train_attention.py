"""
训练注意力增强的 YOLOv8n-cls 模型

支持三种方式运行:
  1. 单模型训练:   python train_attention.py --att se
  2. 顺序训练全部:  python train_attention.py --att all
  3. 并行训练:     开三个终端分别运行 --att se / --att cbam / --att eca

YAML 架构保障:
  - 模型由 YAML 构建，注意力模块从 ultralytics 内置注册表加载
  - 训练后保存的 best.pt 自包含 YAML 结构，重新加载不丢失注意力
"""
import os
import sys
import argparse
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_PATH, MODEL_PATH, RUNS_DIR,
    BATCH_SIZE, IMGSZ, EPOCHS, LR0, WEIGHT_DECAY,
    MOMENTUM, WARMUP_EPOCHS, PATIENCE, DEVICE_ID,
    ATTENTION_TYPES,
)
from build_attention_model import build_attention_model, get_yaml_path


def train_one(att_type: str):
    """训练单个注意力模型"""
    model_name = f"yolov8n_{att_type}"
    project_dir = os.path.join(RUNS_DIR, model_name)
    yaml_path = get_yaml_path(att_type)

    print(f"\n{'='*60}")
    print(f"  训练: {model_name}")
    print(f"  YAML: {yaml_path}")
    print(f"  数据: {DATA_PATH}")
    print(f"  输出: {project_dir}")
    print(f"{'='*60}\n")

    # Step 1: 从 YAML 构建模型 + 加载预训练权重
    model = build_attention_model(att_type, pretrained_weights=MODEL_PATH)

    # Step 2: 训练
    results = model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        lr0=LR0,
        weight_decay=WEIGHT_DECAY,
        momentum=MOMENTUM,
        warmup_epochs=WARMUP_EPOCHS,
        patience=PATIENCE,
        device=DEVICE_ID,
        project=project_dir,
        name="train",
        exist_ok=True,
        pretrained=False,  # 预训练权重已通过 model.load 加载
        optimizer="AdamW",
        verbose=True,
    )

    best_pt = os.path.join(project_dir, "train", "weights", "best.pt")
    print(f"\n[SUCCESS] {model_name} 训练完成!")
    print(f"[SUCCESS] 最佳模型: {best_pt}")

    # Step 3: 验证保存的模型加载后注意力仍在
    verify_saved_model(best_pt, att_type)

    return results


def verify_saved_model(pt_path: str, att_type: str):
    """验证保存的 best.pt 重新加载后注意力模块是否仍然存在"""
    from build_attention_model import verify_attention_modules

    print(f"\n[VERIFY] 重新加载 {pt_path} 验证注意力模块...")
    loaded = YOLO(pt_path)
    verify_attention_modules(loaded, att_type)


def main():
    parser = argparse.ArgumentParser(description="训练 YOLOv8n-cls 注意力模型")
    parser.add_argument(
        "--att", type=str, default="all",
        choices=["se", "cbam", "eca", "all"],
        help="注意力类型: se | cbam | eca | all (顺序训练全部)"
    )
    args = parser.parse_args()

    if args.att == "all":
        # 顺序训练三种
        for att_type in ATTENTION_TYPES:
            try:
                train_one(att_type)
            except Exception as e:
                print(f"[ERROR] {att_type} 训练失败: {e}")
                import traceback
                traceback.print_exc()
        print("\n[SUMMARY] 全部注意力模型训练完成!")
    else:
        # 训练单个（可开多个终端并行）
        train_one(args.att)


if __name__ == "__main__":
    main()