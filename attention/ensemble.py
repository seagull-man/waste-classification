"""
YOLOv8 模型融合（原始模型 + ECA模型）
通过投票机制集成两个模型的预测结果
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


def evaluate_ensemble(model_orig, model_eca, data_path, class_names):
    """评估融合模型（投票机制）"""
    print("\n🔍 评估融合模型（投票机制）...")

    all_predictions = []
    all_true_labels = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = Path(data_path) / 'val' / class_name
        if not class_dir.exists():
            continue

        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))

        for img_path in images:
            try:
                # 原始模型预测
                results_orig = model_orig(str(img_path), verbose=False)
                pred_orig = int(results_orig[0].probs.top1)
                prob_orig = float(results_orig[0].probs.top1conf)

                # ECA模型预测
                results_eca = model_eca(str(img_path), verbose=False)
                pred_eca = int(results_eca[0].probs.top1)
                prob_eca = float(results_eca[0].probs.top1conf)

                # 投票机制（带权重）
                if prob_orig > prob_eca:
                    final_pred = pred_orig
                elif prob_eca > prob_orig:
                    final_pred = pred_eca
                else:
                    # 概率相同时，选择原始模型的预测
                    final_pred = pred_orig

                all_predictions.append(final_pred)
                all_true_labels.append(class_idx)
            except Exception as e:
                continue

    if len(all_predictions) == 0:
        print("  ❌ 没有找到测试图片")
        return None

    metrics = calculate_metrics(all_true_labels, all_predictions, len(class_names))

    print(f"  ✅ 评估完成，共 {len(all_predictions)} 张图片")
    print(f"  📊 准确率: {metrics['overall_accuracy']:.4f}")

    return metrics


def print_comparison(metrics_orig, metrics_eca, metrics_ensemble, class_names):
    """打印对比结果"""
    print("\n" + "="*80)
    print("📊 性能对比结果")
    print("="*80)

    print(f"\n{'指标':<20} {'原始模型':<15} {'ECA模型':<15} {'融合模型':<15} {'融合-原始':<15}")
    print("-" * 90)
    print(f"{'总体准确率':<20} {metrics_orig['overall_accuracy']:<15.4f} {metrics_eca['overall_accuracy']:<15.4f} {metrics_ensemble['overall_accuracy']:<15.4f} {metrics_ensemble['overall_accuracy'] - metrics_orig['overall_accuracy']:<+15.4f}")
    print(f"{'宏平均精确率':<20} {metrics_orig['macro_precision']:<15.4f} {metrics_eca['macro_precision']:<15.4f} {metrics_ensemble['macro_precision']:<15.4f} {metrics_ensemble['macro_precision'] - metrics_orig['macro_precision']:<+15.4f}")
    print(f"{'宏平均召回率':<20} {metrics_orig['macro_recall']:<15.4f} {metrics_eca['macro_recall']:<15.4f} {metrics_ensemble['macro_recall']:<15.4f} {metrics_ensemble['macro_recall'] - metrics_orig['macro_recall']:<+15.4f}")
    print(f"{'宏平均F1分数':<20} {metrics_orig['macro_f1']:<15.4f} {metrics_eca['macro_f1']:<15.4f} {metrics_ensemble['macro_f1']:<15.4f} {metrics_ensemble['macro_f1'] - metrics_orig['macro_f1']:<+15.4f}")


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
    print("🧪 YOLOv8 模型融合（原始 + ECA）性能评估")
    print("="*80)

    data_path = '/root/autodl-tmp/garbage_4cls'
    class_names = ['hazardous', 'kitchen', 'other', 'recyclable']

    orig_model_path = 'runs/classify/waste_cls_original-2/weights/best.pt'
    eca_model_path = 'runs/classify/waste_cls_eca_frozen/weights/best.pt'

    if not Path(orig_model_path).exists():
        print(f"\n❌ 原始模型不存在: {orig_model_path}")
        print("💡 请先运行 train_original.py 训练原始模型")
        return

    if not Path(eca_model_path).exists():
        print(f"\n❌ ECA模型不存在: {eca_model_path}")
        print("💡 请先运行 train_eca.py 训练ECA模型")
        return

    print(f"📂 数据集: {data_path}")
    print(f"📂 原始模型: {orig_model_path}")
    print(f"📂 ECA模型: {eca_model_path}")

    print(f"\n📥 加载原始模型...")
    model_orig = YOLO(orig_model_path)

    print(f"\n📥 加载ECA模型...")
    model_eca = YOLO(eca_model_path)

    # 评估原始模型
    print(f"\n🔍 评估 原始模型...")
    all_predictions_orig = []
    all_true_labels = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = Path(data_path) / 'val' / class_name
        if not class_dir.exists():
            continue
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
        for img_path in images:
            try:
                results = model_orig(str(img_path), verbose=False)
                pred_class = int(results[0].probs.top1)
                all_predictions_orig.append(pred_class)
                all_true_labels.append(class_idx)
            except Exception:
                continue
    metrics_orig = calculate_metrics(all_true_labels, all_predictions_orig, len(class_names))
    print(f"  ✅ 评估完成，共 {len(all_predictions_orig)} 张图片")
    print(f"  📊 准确率: {metrics_orig['overall_accuracy']:.4f}")

    # 评估ECA模型
    print(f"\n🔍 评估 ECA模型...")
    all_predictions_eca = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = Path(data_path) / 'val' / class_name
        if not class_dir.exists():
            continue
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
        for img_path in images:
            try:
                results = model_eca(str(img_path), verbose=False)
                pred_class = int(results[0].probs.top1)
                all_predictions_eca.append(pred_class)
            except Exception:
                continue
    metrics_eca = calculate_metrics(all_true_labels, all_predictions_eca, len(class_names))
    print(f"  ✅ 评估完成，共 {len(all_predictions_eca)} 张图片")
    print(f"  📊 准确率: {metrics_eca['overall_accuracy']:.4f}")

    # 评估融合模型
    metrics_ensemble = evaluate_ensemble(model_orig, model_eca, data_path, class_names)
    if metrics_ensemble is None:
        print("❌ 融合模型评估失败")
        return

    # 打印对比结果
    print_comparison(metrics_orig, metrics_eca, metrics_ensemble, class_names)

    print("\n" + "="*80)
    print("📋 混淆矩阵对比")
    print("="*80)

    print_confusion_matrix(metrics_orig['confusion_matrix'], class_names, "原始模型混淆矩阵")
    print_confusion_matrix(metrics_eca['confusion_matrix'], class_names, "ECA模型混淆矩阵")
    print_confusion_matrix(metrics_ensemble['confusion_matrix'], class_names, "融合模型混淆矩阵")

    print("\n" + "="*80)
    print("✅ 对比评估完成！")
    print("="*80)

    improvement = metrics_ensemble['overall_accuracy'] - metrics_orig['overall_accuracy']
    if improvement > 0:
        print(f"\n🎉 融合模型成功超越原始模型！")
        print(f"📈 准确率提升: {improvement:.4f} (+{improvement*100:.2f}%)")
    elif improvement < 0:
        print(f"\n📉 融合模型性能未超过原始模型")
        print(f"🔻 准确率下降: {abs(improvement):.4f} ({improvement*100:.2f}%)")
    else:
        print(f"\n➖ 融合模型与原始模型性能相同")

    # 保存结果
    result_text = f"""
================================================================================
🧪 YOLOv8 模型融合性能评估报告
================================================================================

数据集: {data_path}
模型: 原始YOLOv8n + ECA模型
融合策略: 基于置信度的加权投票

--------------------------------------------------------------------------------
📊 性能对比
--------------------------------------------------------------------------------
{'指标':<20} {'原始模型':<15} {'ECA模型':<15} {'融合模型':<15} {'融合-原始':<15}
{'总体准确率':<20} {metrics_orig['overall_accuracy']:<15.4f} {metrics_eca['overall_accuracy']:<15.4f} {metrics_ensemble['overall_accuracy']:<15.4f} {metrics_ensemble['overall_accuracy'] - metrics_orig['overall_accuracy']:<+15.4f}
{'宏平均精确率':<20} {metrics_orig['macro_precision']:<15.4f} {metrics_eca['macro_precision']:<15.4f} {metrics_ensemble['macro_precision']:<15.4f} {metrics_ensemble['macro_precision'] - metrics_orig['macro_precision']:<+15.4f}
{'宏平均召回率':<20} {metrics_orig['macro_recall']:<15.4f} {metrics_eca['macro_recall']:<15.4f} {metrics_ensemble['macro_recall']:<15.4f} {metrics_ensemble['macro_recall'] - metrics_orig['macro_recall']:<+15.4f}
{'宏平均F1分数':<20} {metrics_orig['macro_f1']:<15.4f} {metrics_eca['macro_f1']:<15.4f} {metrics_ensemble['macro_f1']:<15.4f} {metrics_ensemble['macro_f1'] - metrics_orig['macro_f1']:<+15.4f}

--------------------------------------------------------------------------------
结论: {'融合模型超越原始模型！' if improvement > 0 else '融合模型未超过原始模型'}
准确率变化: {improvement:+.4f} ({improvement*100:+.2f}%)
================================================================================
"""
    report_path = '/root/autodl-tmp/runs/classify/ensemble_report.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(result_text)
    print(f"\n📄 完整报告已保存到: {report_path}")


if __name__ == "__main__":
    main()