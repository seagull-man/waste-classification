"""
YOLOv8 优化模型 vs 原始模型 性能对比
"""

import os
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import numpy as np
from ultralytics import YOLO


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


def print_comparison(metrics_orig, metrics_opt, class_names):
    """打印对比结果"""
    print("\n" + "="*80)
    print("📊 性能对比结果")
    print("="*80)

    print(f"\n{'指标':<20} {'原始模型':<15} {'优化模型':<15} {'差异':<15}")
    print("-" * 70)
    print(f"{'总体准确率':<20} {metrics_orig['overall_accuracy']:<15.4f} {metrics_opt['overall_accuracy']:<15.4f} {metrics_opt['overall_accuracy'] - metrics_orig['overall_accuracy']:<+15.4f}")
    print(f"{'宏平均精确率':<20} {metrics_orig['macro_precision']:<15.4f} {metrics_opt['macro_precision']:<15.4f} {metrics_opt['macro_precision'] - metrics_orig['macro_precision']:<+15.4f}")
    print(f"{'宏平均召回率':<20} {metrics_orig['macro_recall']:<15.4f} {metrics_opt['macro_recall']:<15.4f} {metrics_opt['macro_recall'] - metrics_orig['macro_recall']:<+15.4f}")
    print(f"{'宏平均F1分数':<20} {metrics_orig['macro_f1']:<15.4f} {metrics_opt['macro_f1']:<15.4f} {metrics_opt['macro_f1'] - metrics_orig['macro_f1']:<+15.4f}")

    print(f"\n{'类别':<15} {'指标':<12} {'原始模型':<12} {'优化模型':<12} {'差异':<12}")
    print("-" * 70)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {'精确率':<12} {metrics_orig['precision'][i]:<12.4f} {metrics_opt['precision'][i]:<12.4f} {metrics_opt['precision'][i] - metrics_orig['precision'][i]:<+12.4f}")
        print(f"{'':<15} {'召回率':<12} {metrics_orig['recall'][i]:<12.4f} {metrics_opt['recall'][i]:<12.4f} {metrics_opt['recall'][i] - metrics_orig['recall'][i]:<+12.4f}")
        print(f"{'':<15} {'F1分数':<12} {metrics_orig['f1'][i]:<12.4f} {metrics_opt['f1'][i]:<12.4f} {metrics_opt['f1'][i] - metrics_orig['f1'][i]:<+12.4f}")
        print()


def print_confusion_matrix(cm, class_names, title):
    """打印混淆矩阵"""
    print(f"\n{title}:\n")
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
    print("🧪 YOLOv8 原始模型 vs 优化模型 性能对比")
    print("="*80)

    data_path = '/root/autodl-tmp/garbage_4cls'
    class_names = ['hazardous', 'kitchen', 'other', 'recyclable']

    orig_model_path = 'runs/classify/waste_cls_original-2/weights/best.pt'
    opt_model_path = 'runs/classify/waste_cls_optimized/weights/best.pt'

    if not Path(orig_model_path).exists():
        print(f"\n❌ 原始模型不存在: {orig_model_path}")
        print("💡 请先运行 train_original.py 训练原始模型")
        return

    if not Path(opt_model_path).exists():
        print(f"\n❌ 优化模型不存在: {opt_model_path}")
        print("💡 请先运行 betteryolo.py 训练优化模型")
        return

    print(f"📂 数据集: {data_path}")
    print(f"📂 原始模型: {orig_model_path}")
    print(f"📂 优化模型: {opt_model_path}")

    print(f"\n📥 加载原始模型...")
    model_orig = YOLO(orig_model_path)

    print(f"\n📥 加载优化模型...")
    model_opt = YOLO(opt_model_path)

    metrics_orig = evaluate_yolo_model(model_orig, data_path, class_names, "原始模型(YOLOv8n)")
    if metrics_orig is None:
        print("❌ 原始模型评估失败")
        return

    metrics_opt = evaluate_yolo_model(model_opt, data_path, class_names, "优化模型(YOLOv8n+)")
    if metrics_opt is None:
        print("❌ 优化模型评估失败")
        return

    print_comparison(metrics_orig, metrics_opt, class_names)

    print("\n" + "="*80)
    print("📋 混淆矩阵对比")
    print("="*80)

    print_confusion_matrix(metrics_orig['confusion_matrix'], class_names, "原始模型混淆矩阵")
    print_confusion_matrix(metrics_opt['confusion_matrix'], class_names, "优化模型混淆矩阵")

    print("\n" + "="*80)
    print("✅ 对比评估完成！")
    print("="*80)

    improvement = metrics_opt['overall_accuracy'] - metrics_orig['overall_accuracy']
    if improvement > 0:
        print(f"\n🎉 优化模型成功超越原始模型！")
        print(f"📈 准确率提升: {improvement:.4f} (+{improvement*100:.2f}%)")
    elif improvement < 0:
        print(f"\n📉 优化模型性能未超过原始模型")
        print(f"🔻 准确率下降: {abs(improvement):.4f} ({improvement*100:.2f}%)")
    else:
        print(f"\n➖ 优化模型与原始模型性能相同")

    # 保存结果
    result_text = f"""
================================================================================
🧪 YOLOv8 优化模型性能评估报告
================================================================================

数据集: {data_path}
模型: 原始YOLOv8n vs 优化YOLOv8n

--------------------------------------------------------------------------------
📊 性能对比
--------------------------------------------------------------------------------
{'指标':<20} {'原始模型':<15} {'优化模型':<15} {'差异':<15}
{'总体准确率':<20} {metrics_orig['overall_accuracy']:<15.4f} {metrics_opt['overall_accuracy']:<15.4f} {metrics_opt['overall_accuracy'] - metrics_orig['overall_accuracy']:<+15.4f}
{'宏平均精确率':<20} {metrics_orig['macro_precision']:<15.4f} {metrics_opt['macro_precision']:<15.4f} {metrics_opt['macro_precision'] - metrics_orig['macro_precision']:<+15.4f}
{'宏平均召回率':<20} {metrics_orig['macro_recall']:<15.4f} {metrics_opt['macro_recall']:<15.4f} {metrics_opt['macro_recall'] - metrics_orig['macro_recall']:<+15.4f}
{'宏平均F1分数':<20} {metrics_orig['macro_f1']:<15.4f} {metrics_opt['macro_f1']:<15.4f} {metrics_opt['macro_f1'] - metrics_orig['macro_f1']:<+15.4f}

--------------------------------------------------------------------------------
结论: {'优化模型超越原始模型！' if improvement > 0 else '优化模型未超过原始模型'}
准确率变化: {improvement:+.4f} ({improvement*100:+.2f}%)
================================================================================
"""
    report_path = '/root/autodl-tmp/runs/classify/optimization_report.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(result_text)
    print(f"\n📄 完整报告已保存到: {report_path}")


if __name__ == "__main__":
    main()