"""
YOLOv8 原始模型 vs SE-Block模型 性能对比脚本

评估两个模型在测试集上的性能，并生成详细的对比报告
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np
from ultralytics import YOLO
from SE_Block import SELayer


def calculate_metrics(y_true, y_pred, num_classes):
    """计算分类指标"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': precision.mean(),
        'macro_recall': recall.mean(),
        'macro_f1': f1.mean(),
        'overall_accuracy': np.diag(cm).sum() / cm.sum()
    }


class SEYOLOV8(nn.Module):
    """带SE注意力的YOLOv8分类模型 - 与train_se.py中完全一致"""

    def __init__(self, original_model=None, num_classes=4, reduction=16, backbone_ch=256):
        super(SEYOLOV8, self).__init__()

        if original_model is not None:
            self.backbone = original_model.model.model[:-1]
            temp_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                temp_output = self.backbone(temp_input)
            backbone_output_ch = temp_output.shape[1]
        else:
            base_model = YOLO('yolov8n-cls.pt')
            self.backbone = base_model.model.model[:-1]
            backbone_output_ch = backbone_ch

        self.se_layer = SELayer(channel=backbone_output_ch, reduction=reduction)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(backbone_output_ch, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.se_layer(features)
        output = self.classifier(features)
        return output


def load_original_model(model_path):
    """加载原始YOLOv8模型"""
    print(f"\n📥 加载原始模型: {model_path}")
    model = YOLO(model_path)
    return model


def load_se_model(model_path, num_classes=4, backbone_ch=256):
    """加载SE模型 - 与train_se.py保存的结构完全一致"""
    print(f"\n📥 加载SE模型: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model = SEYOLOV8(num_classes=num_classes, backbone_ch=backbone_ch)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    print(f"  ✅ SE模型加载成功")

    return model


def evaluate_yolo_model(model, data_path, class_names, model_type="模型"):
    """评估YOLO分类模型"""
    print(f"\n🔍 评估 {model_type}...")

    all_predictions = []
    all_true_labels = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = Path(data_path) / 'val' / class_name
        if not class_dir.exists():
            continue

        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))

        for img_path in images:
            try:
                results = model(str(img_path), verbose=False)
                pred_class = int(results[0].probs.top1)
                all_predictions.append(pred_class)
                all_true_labels.append(class_idx)
            except Exception as e:
                continue

    if len(all_predictions) == 0:
        print(f"  ❌ 没有找到测试图片")
        return None

    metrics = calculate_metrics(all_true_labels, all_predictions, len(class_names))

    print(f"  ✅ 评估完成，共 {len(all_predictions)} 张图片")
    print(f"  📊 准确率: {metrics['overall_accuracy']:.4f}")

    return metrics


def evaluate_pytorch_model(model, data_path, class_names, model_type="模型", device='cuda:0'):
    """评估PyTorch模型"""
    print(f"\n🔍 评估 {model_type} (PyTorch模式)...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_predictions = []
    all_true_labels = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for class_idx, class_name in enumerate(class_names):
            class_dir = Path(data_path) / 'val' / class_name
            if not class_dir.exists():
                continue

            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))

            for img_path in images:
                try:
                    image = Image.open(img_path).convert('RGB')
                    img_tensor = transform(image).unsqueeze(0).to(device)

                    outputs = model(img_tensor)
                    _, predicted = outputs.max(1)

                    all_predictions.append(predicted.item())
                    all_true_labels.append(class_idx)
                except Exception as e:
                    continue

    if len(all_predictions) == 0:
        print(f"  ❌ 没有找到测试图片")
        return None

    metrics = calculate_metrics(all_true_labels, all_predictions, len(class_names))

    print(f"  ✅ 评估完成，共 {len(all_predictions)} 张图片")
    print(f"  📊 准确率: {metrics['overall_accuracy']:.4f}")

    return metrics


def print_comparison(metrics_orig, metrics_se, class_names):
    """打印对比结果"""
    print("\n" + "="*80)
    print("📊 性能对比结果")
    print("="*80)

    print(f"\n{'指标':<20} {'原始模型':<15} {'SE模型':<15} {'差异':<15}")
    print("-" * 70)
    print(f"{'总体准确率':<20} {metrics_orig['overall_accuracy']:<15.4f} {metrics_se['overall_accuracy']:<15.4f} {metrics_se['overall_accuracy'] - metrics_orig['overall_accuracy']:<+15.4f}")
    print(f"{'宏平均精确率':<20} {metrics_orig['macro_precision']:<15.4f} {metrics_se['macro_precision']:<15.4f} {metrics_se['macro_precision'] - metrics_orig['macro_precision']:<+15.4f}")
    print(f"{'宏平均召回率':<20} {metrics_orig['macro_recall']:<15.4f} {metrics_se['macro_recall']:<15.4f} {metrics_se['macro_recall'] - metrics_orig['macro_recall']:<+15.4f}")
    print(f"{'宏平均F1分数':<20} {metrics_orig['macro_f1']:<15.4f} {metrics_se['macro_f1']:<15.4f} {metrics_se['macro_f1'] - metrics_orig['macro_f1']:<+15.4f}")

    print(f"\n{'类别':<15} {'指标':<12} {'原始模型':<12} {'SE模型':<12} {'差异':<12}")
    print("-" * 70)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {'精确率':<12} {metrics_orig['precision'][i]:<12.4f} {metrics_se['precision'][i]:<12.4f} {metrics_se['precision'][i] - metrics_orig['precision'][i]:<+12.4f}")
        print(f"{'':<15} {'召回率':<12} {metrics_orig['recall'][i]:<12.4f} {metrics_se['recall'][i]:<12.4f} {metrics_se['recall'][i] - metrics_orig['recall'][i]:<+12.4f}")
        print(f"{'':<15} {'F1分数':<12} {metrics_orig['f1'][i]:<12.4f} {metrics_se['f1'][i]:<12.4f} {metrics_se['f1'][i] - metrics_orig['f1'][i]:<+12.4f}")
        print()


def print_confusion_matrix(cm, class_names, title):
    """打印混淆矩阵"""
    print(f"\n{title}混淆矩阵:")
    print(f"{'':>15}", end='')
    for name in class_names:
        print(f"{name[:10]:>15}", end='')
    print()

    for i, name in enumerate(class_names):
        print(f"{name[:10]:>15}", end='')
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>15}", end='')
        print()


def main():
    print("="*80)
    print("🧪 YOLOv8 原始模型 vs SE-Block模型 性能对比")
    print("="*80)

    data_path = 'garbage_4cls'
    class_names = ['hazardous', 'kitchen', 'other', 'recyclable']

    orig_model_path = 'runs/classify/waste_cls_original/weights/best.pt'
    se_model_path = 'runs/classify/waste_cls_se/weights/best.pt'

    if not Path(orig_model_path).exists():
        print(f"\n❌ 原始模型不存在: {orig_model_path}")
        print("💡 请先运行 train_original.py 训练原始模型")
        return

    if not Path(se_model_path).exists():
        print(f"\n❌ SE模型不存在: {se_model_path}")
        print("💡 请先运行 train_se.py 训练SE模型")
        return

    print(f"📂 数据集: {data_path}")
    print(f"📂 原始模型: {orig_model_path}")
    print(f"📂 SE模型: {se_model_path}")

    model_orig = load_original_model(orig_model_path)
    model_se = load_se_model(se_model_path, num_classes=4, backbone_ch=256)

    metrics_orig = evaluate_yolo_model(model_orig, data_path, class_names, "原始模型")
    if metrics_orig is None:
        print("❌ 原始模型评估失败")
        return

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    metrics_se = evaluate_pytorch_model(model_se, data_path, class_names, "SE模型", device=device)
    if metrics_se is None:
        print("❌ SE模型评估失败")
        return

    print_comparison(metrics_orig, metrics_se, class_names)

    print("\n" + "="*80)
    print("📋 混淆矩阵对比")
    print("="*80)

    print_confusion_matrix(metrics_orig['confusion_matrix'], class_names, "原始模型")
    print_confusion_matrix(metrics_se['confusion_matrix'], class_names, "SE模型")

    print("\n" + "="*80)
    print("✅ 对比评估完成！")
    print("="*80)

    improvement = metrics_se['overall_accuracy'] - metrics_orig['overall_accuracy']
    if improvement > 0:
        print(f"\n📈 SE模型准确率提升: {improvement:.4f} (+{improvement*100:.2f}%)")
    elif improvement < 0:
        print(f"\n📉 SE模型准确率下降: {abs(improvement):.4f} ({improvement*100:.2f}%)")
        print("\n⚠️  警告：SE模型性能下降！")
        print("可能原因：")
        print("  1. 模型结构定义不一致")
        print("  2. 学习率设置不当")
        print("  3. 训练轮数不够")
        print("  4. 权重加载失败")
        print("\n💡 建议：重新训练SE模型")
    else:
        print(f"\n➖ 两个模型准确率相同")


if __name__ == "__main__":
    main()
