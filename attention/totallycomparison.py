"""
多模型性能对比（测试集）
评估原始模型、SE、ECA、Residual SE 在测试集上的性能
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
        class_dir = Path(data_path) / class_name
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
    print("🧪 多模型性能对比（测试集）")
    print("="*80)

    # 测试集路径
    test_data_path = '/root/autodl-tmp/garbage_4cls/test'
    class_names = ['hazardous', 'kitchen', 'other', 'recyclable']

    # 模型路径
    models = {
        '原始模型 (YOLOv8n)': 'runs/classify/waste_cls_original-2/weights/best.pt',
        'SE模型 (ultralytics)': 'runs/classify/waste_cls_se_ultralytics/weights/best.pt',
        'ECA模型 (frozen)': 'runs/classify/waste_cls_eca_frozen/weights/best.pt',
        'Residual SE模型': 'runs/classify/waste_cls_residual_se-3/weights/best.pt',
    }

    # 检查测试集路径
    print(f"📂 测试集路径: {test_data_path}")
    print(f"📋 类别: {class_names}")

    # 检查测试集是否存在
    for class_name in class_names:
        class_dir = Path(test_data_path) / class_name
        if class_dir.exists():
            num_images = len(list(class_dir.glob('*.*')))
            print(f"  ✅ {class_name}: {num_images} 张图片")
        else:
            print(f"  ❌ {class_name}: 目录不存在")

    print("\n" + "="*80)

    # 评估所有模型
    results = {}
    loaded_models = {}

    for model_name, model_path in models.items():
        if not Path(model_path).exists():
            print(f"\n❌ 模型不存在: {model_path}")
            continue

        print(f"\n📥 加载{model_name}...")
        model = YOLO(model_path)
        loaded_models[model_name] = model

        metrics = evaluate_yolo_model(model, test_data_path, class_names, model_name)
        if metrics:
            results[model_name] = metrics

    if len(results) == 0:
        print("\n❌ 没有模型可以评估")
        return

    # 汇总对比
    print("\n" + "="*80)
    print("📊 多模型性能对比（测试集）")
    print("="*80)

    print(f"\n{'模型':<30} {'准确率':<12} {'宏精确率':<12} {'宏召回率':<12} {'宏F1':<12}")
    print("-" * 80)

    for model_name, metrics in results.items():
        print(f"{model_name:<30} {metrics['overall_accuracy']:<12.4f} {metrics['macro_precision']:<12.4f} {metrics['macro_recall']:<12.4f} {metrics['macro_f1']:<12.4f}")

    # 找出最佳模型
    best_model = max(results.items(), key=lambda x: x[1]['overall_accuracy'])
    baseline_model = list(results.values())[0]
    
    print("\n" + "="*80)
    print("📋 详细性能对比（相对于原始模型）")
    print("="*80)

    baseline_acc = baseline_model['overall_accuracy']
    
    print(f"\n{'模型':<30} {'准确率':<12} {'vs原始模型':<12}")
    print("-" * 60)
    for model_name, metrics in results.items():
        diff = metrics['overall_accuracy'] - baseline_acc
        symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"{model_name:<30} {metrics['overall_accuracy']:<12.4f} {diff:+.4f} {symbol}")

    # 各类别详细对比
    print("\n" + "="*80)
    print("📋 各类别F1分数对比")
    print("="*80)

    print(f"\n{'类别':<15}", end='')
    for model_name in results.keys():
        short_name = model_name.split('(')[0].strip()[:10]
        print(f"{short_name:<12}", end='')
    print()
    print("-" * 60)

    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15}", end='')
        for model_name, metrics in results.items():
            print(f"{metrics['f1'][i]:<12.4f}", end='')
        print()

    # 混淆矩阵
    print("\n" + "="*80)
    print("📋 混淆矩阵")
    print("="*80)

    for model_name, metrics in results.items():
        print_confusion_matrix(metrics['confusion_matrix'], class_names, f"{model_name}混淆矩阵")

    print("\n" + "="*80)
    print("✅ 测试评估完成！")
    print("="*80)

    print(f"\n🏆 最佳模型: {best_model[0]}")
    print(f"   准确率: {best_model[1]['overall_accuracy']:.4f}")

    # 保存结果
    result_text = f"""
================================================================================
🧪 多模型性能对比报告（测试集）
================================================================================

测试集路径: {test_data_path}
测试图片总数: {sum([len(list((Path(test_data_path)/c).glob('*.*'))) for c in class_names])}
评估模型数: {len(results)}

--------------------------------------------------------------------------------
📊 总体性能对比
--------------------------------------------------------------------------------
{'模型':<30} {'准确率':<12} {'宏精确率':<12} {'宏召回率':<12} {'宏F1':<12}
"""

    for model_name, metrics in results.items():
        result_text += f"{model_name:<30} {metrics['overall_accuracy']:<12.4f} {metrics['macro_precision']:<12.4f} {metrics['macro_recall']:<12.4f} {metrics['macro_f1']:<12.4f}\n"

    result_text += f"""
--------------------------------------------------------------------------------
🏆 最佳模型: {best_model[0]} (准确率: {best_model[1]['overall_accuracy']:.4f})
--------------------------------------------------------------------------------
"""

    report_path = '/root/autodl-tmp/runs/classify/test_comparison_report.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(result_text)
    print(f"\n📄 完整报告已保存到: {report_path}")


if __name__ == "__main__":
    main()