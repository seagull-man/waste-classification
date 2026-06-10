import os
import sys
import json
import random
import numpy as np
import torch
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_DIR)

from cbam_modules import CBAM, ChannelAttention, SpatialAttention
import ultralytics.nn.tasks as tasks_module

tasks_module.__dict__["CBAM"] = CBAM
tasks_module.__dict__["ChannelAttention"] = ChannelAttention
tasks_module.__dict__["SpatialAttention"] = SpatialAttention

DATA_PATH = "/root/autodl-tmp/garbage_4cls"
CLASSIFY_PATH = "/root/autodl-tmp/runs/classify"
YAML_PATH = "/root/autodl-tmp/waste-classification/attention/yolov8n-cls-cbam-replace.yaml"

IMGSZ = 320
EXPERIMENT_NAME = "waste_cls_yolov8n_cbam_replace"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def check_cbam_params(model, title=""):
    print("\n" + "=" * 70)
    print(f"检查 CBAM 参数: {title}")
    print("=" * 70)

    cbam_keys = [
        k for k in model.model.state_dict().keys()
        if "channel_attention" in k
        or "spatial_attention" in k
        or "gamma" in k
    ]

    print(f"CBAM 相关参数数量: {len(cbam_keys)}")
    for key in cbam_keys:
        print(key)

    if len(cbam_keys) > 0:
        print("✅ 当前模型中已检测到 CBAM 参数")
    else:
        print("❌ 当前模型中没有检测到 CBAM 参数")

def print_model_layers(model):
    print("\n" + "=" * 70)
    print("模型逐层结构")
    print("=" * 70)

    for idx, layer in enumerate(model.model.model):
        print(f"Layer {idx}: {layer.__class__.__name__}")
        print(layer)
        print("-" * 70)

def main():
    set_seed(42)

    print("=" * 70)
    print("替换式 YOLOv8n-cls + Residual CBAM 训练")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据集路径不存在: {DATA_PATH}")

    print(f"\n📂 数据集路径: {DATA_PATH}")
    print(f"🧩 模型结构文件: {YAML_PATH}")
    print("🎯 目标类别数: 4")

    model = YOLO(YAML_PATH)
    model.load("yolov8n-cls.pt")

    print_model_layers(model)
    check_cbam_params(model, "加载预训练权重后，训练前")

    print("\n🚀 开始训练替换式 CBAM-YOLOv8n 模型...")

    results = model.train(
        data=DATA_PATH,
        epochs=100,
        imgsz=IMGSZ,
        batch=32,
        name=EXPERIMENT_NAME,
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

    save_dir = os.path.join(CLASSIFY_PATH, EXPERIMENT_NAME)
    os.makedirs(save_dir, exist_ok=True)

    check_cbam_params(model, "训练完成后")

    training_info = {
        "model_type": "Lightweight YOLOv8n-cls + Residual CBAM",
        "pretrained": "yolov8n-cls.pt",
        "yaml": YAML_PATH,
        "attention": "Residual CBAM",
        "optimization_method": "Reduce high-level C2f repeats and add CBAM",
        "seed": 42,
        "epochs": 100,
        "imgsz": IMGSZ,
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
        "note": "减少网络后段 C2f 模块重复次数以降低计算量，并引入残差式 CBAM 对高层特征进行通道与空间注意力增强，在相同输入尺寸下兼顾推理速度与分类准确率。"
    }

    info_path = os.path.join(save_dir, "training_info.json")

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("✅ 替换式 CBAM-YOLOv8n 模型训练完成！")
    print("=" * 70)
    print(f"📁 最佳模型保存路径: {save_dir}/weights/best.pt")
    print(f"💾 训练信息保存到: {info_path}")

if __name__ == "__main__":
    main()