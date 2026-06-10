"""
YOLOv8n-cls + Residual CBAM + Weighted Focal Loss
用于垃圾四分类任务

改进目标：
1. 保留已经验证有效的 Residual CBAM 结构
2. 使用 Weighted Focal Loss 缓解类别不均衡和难样本学习不足
3. 加入 Label Smoothing，降低模型过度自信
4. 使用 Cosine LR，提高训练后期收敛稳定性
"""

import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationTrainer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_DIR)

from cbam_modules import CBAM, WrapperWithCBAM, ChannelAttention, SpatialAttention

try:
    from path_config import DATA_PATH, CLASSIFY_PATH
except ImportError:
    DATA_PATH = "/root/autodl-tmp/garbage_4cls"
    CLASSIFY_PATH = "/root/autodl-tmp/runs/classify"

CBAM_LAYER_INDICES = (4, 6, 8)
IMGSZ = 320
EXPERIMENT_NAME = "new_cbam_focal"

def set_seed(seed=42):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def count_parameters(torch_model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in torch_model.parameters())
    trainable_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
    return total_params, trainable_params

def compute_class_weights(data_path):
    """
    根据 train 目录中的类别样本数计算类别权重

    权重公式：
        weight_i = total_samples / (num_classes * samples_i)

    样本少的类别权重大，样本多的类别权重小。
    """
    train_dir = Path(data_path) / "train"

    if not train_dir.exists():
        raise FileNotFoundError(f"训练集目录不存在: {train_dir}")

    class_dirs = sorted([p for p in train_dir.iterdir() if p.is_dir()])
    class_names = [p.name for p in class_dirs]

    image_suffixes = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    class_counts = []

    for class_dir in class_dirs:
        count = 0
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in image_suffixes:
                count += 1
        class_counts.append(count)

    class_counts = np.array(class_counts, dtype=np.float32)
    total_samples = class_counts.sum()
    num_classes = len(class_counts)

    weights = total_samples / (num_classes * np.maximum(class_counts, 1.0))
    weights = weights / weights.mean()

    print("\n📊 类别样本统计与损失权重:")
    for name, count, weight in zip(class_names, class_counts, weights):
        print(f"  {name:<12} samples={int(count):<6} weight={weight:.4f}")

    return class_names, torch.tensor(weights, dtype=torch.float32)

class WeightedFocalClassificationLoss(nn.Module):
    """
    加权 Focal Loss for YOLOv8 分类任务

    普通 CrossEntropy：
        CE = -log(p_t)

    Focal Loss：
        FL = alpha * (1 - p_t)^gamma * CE

    这里同时支持：
    1. class_weights：类别权重
    2. gamma：难样本聚焦系数
    3. label_smoothing：标签平滑
    """

    def __init__(self, class_weights=None, gamma=1.5, label_smoothing=0.05):
        super().__init__()

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, preds, batch):
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        targets = batch["cls"].long().view(-1)

        class_weights = None
        if self.class_weights is not None:
            class_weights = self.class_weights.to(preds.device)

        ce_loss = F.cross_entropy(
            preds,
            targets,
            weight=class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing
        )

        pt = torch.exp(-ce_loss)
        focal_weight = (1.0 - pt).pow(self.gamma)

        loss = focal_weight * ce_loss
        loss = loss.mean()

        return loss, loss.detach()

def get_layer_output_channels(model_module, layer_idx, imgsz=320):
    """通过前向传播推断指定层输出通道数"""
    device = next(model_module.parameters()).device
    temp_input = torch.randn(1, 3, imgsz, imgsz, device=device)

    was_training = model_module.training
    model_module.eval()

    with torch.no_grad():
        for idx in range(layer_idx + 1):
            temp_input = model_module[idx](temp_input)

    if was_training:
        model_module.train()

    return temp_input.shape[1]

def inspect_cbam_modules(torch_model, stage_name):
    """检查模型中是否存在 CBAM"""
    print("\n" + "=" * 70)
    print(f"🔎 {stage_name}：检查 CBAM 是否存在")
    print("=" * 70)

    cbam_modules = []

    for name, module in torch_model.named_modules():
        is_cbam_module = isinstance(
            module,
            (CBAM, WrapperWithCBAM, ChannelAttention, SpatialAttention)
        )
        has_cbam_name = "cbam" in name.lower()

        if is_cbam_module or has_cbam_name:
            cbam_modules.append((name, module.__class__.__name__))

    if len(cbam_modules) == 0:
        print("❌ 未检测到 CBAM 相关模块")
    else:
        print(f"✅ 检测到 {len(cbam_modules)} 个 CBAM 相关模块")
        for name, module_type in cbam_modules[:10]:
            print(f"  - {name}: {module_type}")
        if len(cbam_modules) > 10:
            print(f"  ... 其余 {len(cbam_modules) - 10} 个省略")

    total_params, trainable_params = count_parameters(torch_model)

    print("\n📊 参数量统计:")
    print(f"  Total Params:     {total_params / 1e6:.6f} M")
    print(f"  Trainable Params: {trainable_params / 1e6:.6f} M")

    return {
        "stage": stage_name,
        "cbam_module_count": len(cbam_modules),
        "cbam_modules": cbam_modules,
        "total_params": total_params,
        "trainable_params": trainable_params
    }

def add_cbam_to_classification_model(torch_model, layer_indices=CBAM_LAYER_INDICES, imgsz=IMGSZ):
    """给 Ultralytics ClassificationModel 插入 CBAM"""
    if not hasattr(torch_model, "model"):
        raise AttributeError("当前模型没有 model 属性，无法插入 CBAM")

    model_module = torch_model.model

    print("\n🔍 YOLOv8-cls 内部模型结构:")
    for idx, layer in enumerate(model_module):
        print(f"Layer {idx}: {layer.__class__.__name__}")

    print("\n🔧 插入 Residual CBAM...")

    inserted_layers = []

    for layer_idx in layer_indices:
        if layer_idx >= len(model_module):
            print(f"⚠️ 跳过 Layer {layer_idx}，因为模型没有这一层")
            continue

        if isinstance(model_module[layer_idx], WrapperWithCBAM):
            print(f"⚠️ Layer {layer_idx} 已经是 WrapperWithCBAM，跳过")
            continue

        out_channels = get_layer_output_channels(
            model_module=model_module,
            layer_idx=layer_idx,
            imgsz=imgsz
        )

        original_layer = model_module[layer_idx]

        model_module[layer_idx] = WrapperWithCBAM(
            base_layer=original_layer,
            channels=out_channels
        )

        inserted_layers.append({
            "layer_idx": layer_idx,
            "channels": out_channels,
            "original_layer": original_layer.__class__.__name__
        })

        print(
            f"✅ Layer {layer_idx} 添加 Residual CBAM，"
            f"原始层: {original_layer.__class__.__name__}，"
            f"通道数: {out_channels}"
        )

    return inserted_layers

class CBAMFocalClassificationTrainer(ClassificationTrainer):
    """
    自定义分类训练器

    核心改动：
    1. 在 get_model() 中插入 CBAM
    2. 替换原始 CrossEntropy 为 Weighted Focal Loss
    """

    inserted_layers = []

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)

        print("\n" + "=" * 70)
        print("🧩 CBAMFocalClassificationTrainer.get_model() 已接管模型构建")
        print("=" * 70)

        original_total_params, original_trainable_params = count_parameters(model)

        print("\n📊 插入 CBAM 前参数量:")
        print(f"  Total Params:     {original_total_params / 1e6:.6f} M")
        print(f"  Trainable Params: {original_trainable_params / 1e6:.6f} M")

        self.inserted_layers = add_cbam_to_classification_model(
            torch_model=model,
            layer_indices=CBAM_LAYER_INDICES,
            imgsz=IMGSZ
        )

        class_names, class_weights = compute_class_weights(DATA_PATH)

        model.criterion = WeightedFocalClassificationLoss(
            class_weights=class_weights,
            gamma=1.5,
            label_smoothing=0.05
        )

        print("\n✅ 已替换分类损失函数:")
        print("  Loss: Weighted Focal Loss")
        print("  gamma: 1.5")
        print("  label_smoothing: 0.05")
        print(f"  class_weights: {[round(float(w), 4) for w in class_weights]}")

        inspect_cbam_modules(
            torch_model=model,
            stage_name="Trainer 内部插入 CBAM 后"
        )

        return model

def save_cbam_check_report(save_dir, report):
    """保存 CBAM 检查报告"""
    os.makedirs(save_dir, exist_ok=True)

    report_path = os.path.join(save_dir, "cbam_check_report.json")

    serializable_report = {}

    for key, value in report.items():
        serializable_report[key] = {
            "stage": value["stage"],
            "cbam_module_count": value["cbam_module_count"],
            "cbam_modules": [
                {
                    "name": item[0],
                    "type": item[1]
                }
                for item in value["cbam_modules"]
            ],
            "total_params": value["total_params"],
            "total_params_million": value["total_params"] / 1e6,
            "trainable_params": value["trainable_params"],
            "trainable_params_million": value["trainable_params"] / 1e6
        }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable_report, f, indent=2, ensure_ascii=False)

    print(f"\n💾 CBAM 检查报告保存到: {report_path}")

def main():
    set_seed(42)

    print("=" * 70)
    print("YOLOv8n-cls + Residual CBAM + Weighted Focal Loss 训练")
    print("=" * 70)

    data_path = DATA_PATH
    save_dir = os.path.join(CLASSIFY_PATH, EXPERIMENT_NAME)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集路径不存在: {data_path}")

    print(f"\n📂 数据集路径: {data_path}")
    print("🎯 目标类别数: 4")
    print(f"🧪 实验名称: {EXPERIMENT_NAME}")

    print("\n📥 加载 YOLOv8n-cls 预训练模型...")
    model = YOLO("yolov8n-cls.pt")

    print("\n🚀 开始训练 YOLOv8n-cls + Residual CBAM + Weighted Focal Loss...")

    results = model.train(
        data=data_path,
        epochs=120,
        imgsz=IMGSZ,
        batch=32,
        name=EXPERIMENT_NAME,
        device="cuda:0",
        amp=True,
        workers=8,

        optimizer="AdamW",
        lr0=0.0008,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0008,
        warmup_epochs=5.0,
        cos_lr=True,

        degrees=12.0,
        translate=0.08,
        scale=0.18,
        fliplr=0.5,
        mixup=0.10,
        erasing=0.25,

        patience=25,
        verbose=True,
        save=True,
        save_period=10,
        plots=True,

        trainer=CBAMFocalClassificationTrainer
    )

    print("\n" + "=" * 70)
    print("✅ YOLOv8n-cls + Residual CBAM + Weighted Focal Loss 训练完成！")
    print("=" * 70)

    cbam_check_report = {}

    cbam_check_report["after_train"] = inspect_cbam_modules(
        torch_model=model.model,
        stage_name="训练完成后"
    )

    best_path = os.path.join(save_dir, "weights", "best.pt")

    if os.path.exists(best_path):
        print("\n📥 重新加载训练得到的 best.pt...")
        loaded_model = YOLO(best_path)

        cbam_check_report["after_reload_best"] = inspect_cbam_modules(
            torch_model=loaded_model.model,
            stage_name="重新加载 best.pt 后"
        )
    else:
        print(f"\n⚠️ 未找到 best.pt，无法检查重新加载后的 CBAM: {best_path}")

    save_cbam_check_report(save_dir, cbam_check_report)

    inserted_layers = getattr(model.trainer, "inserted_layers", [])

    training_info = {
        "model_type": "YOLOv8n-cls + Residual CBAM + Weighted Focal Loss",
        "pretrained": "yolov8n-cls.pt",
        "attention": "Residual CBAM",
        "loss": "Weighted Focal Loss",
        "focal_gamma": 1.5,
        "loss_label_smoothing": 0.05,
        "cbam_layers": list(CBAM_LAYER_INDICES),
        "inserted_layers": inserted_layers,
        "seed": 42,
        "epochs": 120,
        "imgsz": IMGSZ,
        "batch_size": 32,
        "optimizer": "AdamW",
        "learning_rate": 0.0008,
        "lrf": 0.01,
        "cos_lr": True,
        "momentum": 0.937,
        "weight_decay": 0.0008,
        "warmup_epochs": 5.0,
        "data_augmentation": {
            "degrees": 12.0,
            "translate": 0.08,
            "scale": 0.18,
            "fliplr": 0.5,
            "mixup": 0.10,
            "erasing": 0.25
        },
        "cbam_check_summary": {
            "after_train_cbam_count": cbam_check_report.get("after_train", {}).get("cbam_module_count"),
            "after_reload_best_cbam_count": cbam_check_report.get("after_reload_best", {}).get("cbam_module_count")
        },
        "note": "在 Residual CBAM 基础上引入 Weighted Focal Loss、轻量 Label Smoothing 和 Cosine LR，用于提升垃圾四分类精度与难样本识别能力。"
    }

    os.makedirs(save_dir, exist_ok=True)

    info_path = os.path.join(save_dir, "training_info.json")

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)

    print(f"\n📁 最佳模型保存路径: {save_dir}/weights/best.pt")
    print(f"💾 训练信息保存到: {info_path}")

    print("\n" + "=" * 70)
    print("CBAM 检查结论参考")
    print("=" * 70)

    after_count = cbam_check_report.get("after_train", {}).get("cbam_module_count", 0)
    reload_count = cbam_check_report.get("after_reload_best", {}).get("cbam_module_count", 0)

    print(f"训练后 CBAM 模块数: {after_count}")
    print(f"重载后 CBAM 模块数: {reload_count}")

    if after_count > 0 and reload_count > 0:
        print("✅ CBAM 在训练后和重新加载后均存在。")
    elif after_count > 0 and reload_count == 0:
        print("❌ 训练后存在 CBAM，但重新加载后消失，请检查 cbam_modules.py 导入路径。")
    else:
        print("❌ 训练后未检测到 CBAM，请检查自定义 Trainer 是否生效。")

if __name__ == "__main__":
    main()