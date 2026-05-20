"""
YOLOv8 Original vs CBAM Model Detailed Evaluation
With Precision, Recall, F1-Score, Confusion Matrix
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Paths
orig_model_path = '/root/autodl-tmp/runs/classify/waste_cls_original/weights/best.pt'
cbam_model_path = '/root/autodl-tmp/runs/classify/waste_cls_cbam-4/weights/best.pt'
test_data_path = '/root/autodl-tmp/garbage_4cls/test'

# Classes
class_names = ['hazardous', 'kitchen', 'other', 'recyclable']
num_classes = len(class_names)

def evaluate_model(model, data_path, class_names):
    """Evaluate model on test set and return comprehensive metrics"""
    all_predictions = []
    all_true_labels = []
    all_probs = []
    
    print(f"\nEvaluating model...")
    print(f"Test directory: {data_path}")
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = Path(data_path) / class_name
        if not class_dir.exists():
            continue
        
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
        
        for img_path in image_files:
            try:
                results = model(str(img_path), verbose=False)
                pred_class = int(results[0].probs.top1)
                pred_probs = results[0].probs.data.cpu().numpy()
                
                all_predictions.append(pred_class)
                all_true_labels.append(class_idx)
                all_probs.append(pred_probs)
            except Exception as e:
                continue
    
    all_true_labels = np.array(all_true_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    
    # Calculate Confusion Matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(all_true_labels, all_predictions):
        confusion_matrix[true][pred] += 1
    
    # Calculate Per-Class Metrics
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        precision_per_class[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_per_class[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_per_class[i] = 2 * (precision_per_class[i] * recall_per_class[i]) / (precision_per_class[i] + recall_per_class[i]) if (precision_per_class[i] + recall_per_class[i]) > 0 else 0
    
    # Calculate Averages
    macro_precision = precision_per_class.mean()
    macro_recall = recall_per_class.mean()
    macro_f1 = f1_per_class.mean()
    
    # Micro-average
    total_tp = np.diag(confusion_matrix).sum()
    total_fp = confusion_matrix.sum() - np.diag(confusion_matrix).sum() - (confusion_matrix.sum(0) - np.diag(confusion_matrix)).sum() + total_tp
    total_fn = (confusion_matrix.sum(1) - np.diag(confusion_matrix)).sum()
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # Top-1 and Top-5 Accuracy
    correct_top1 = (all_true_labels == all_predictions).sum()
    accuracy_top1 = correct_top1 / len(all_true_labels)
    
    correct_top5 = 0
    for i, probs in enumerate(all_probs):
        top5_classes = np.argsort(-probs)[:5]
        if all_true_labels[i] in top5_classes:
            correct_top5 += 1
    accuracy_top5 = correct_top5 / len(all_true_labels)
    
    return {
        'confusion_matrix': confusion_matrix,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'accuracy_top1': accuracy_top1,
        'accuracy_top5': accuracy_top5,
        'num_samples': len(all_true_labels)
    }

def load_training_curves(results_path):
    """Load training curves from results.csv"""
    epochs = []
    train_loss = []
    val_loss = []
    acc_top1 = []
    
    if Path(results_path).exists():
        with open(results_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                train_loss.append(float(row['train/loss']))
                val_loss.append(float(row['val/loss']))
                acc_top1.append(float(row['metrics/accuracy_top1']))
    
    return epochs, train_loss, val_loss, acc_top1

def print_metrics_comparison(orig_metrics, cbam_metrics):
    """Print comprehensive metrics comparison"""
    print("\n" + "="*120)
    print("📊 Comprehensive Metrics Comparison".center(120))
    print("="*120)
    
    print(f"\n{'Metric':<30} {'Original':<25} {'CBAM':<25} {'Difference':<20} {'Improvement %':<15}")
    print("-"*120)
    
    # Accuracy metrics
    metrics_compare = [
        ('Top-1 Accuracy', 'accuracy_top1'),
        ('Top-5 Accuracy', 'accuracy_top5'),
        ('Macro Precision', 'macro_precision'),
        ('Macro Recall', 'macro_recall'),
        ('Macro F1-Score', 'macro_f1'),
        ('Micro Precision', 'micro_precision'),
        ('Micro Recall', 'micro_recall'),
        ('Micro F1-Score', 'micro_f1'),
    ]
    
    for name, key in metrics_compare:
        orig_val = orig_metrics[key]
        cbam_val = cbam_metrics[key]
        diff = cbam_val - orig_val
        improve_pct = (diff / orig_val) * 100 if orig_val != 0 else 0
        
        print(f"{name:<30} {orig_val:<25.6f} {cbam_val:<25.6f} {diff:<+20.6f} {improve_pct:+.2f}%")
    
    # Per-class metrics
    print(f"\n{'Per-Class Precision':<80}")
    print("-"*120)
    for i, class_name in enumerate(class_names):
        orig_prec = orig_metrics['precision_per_class'][i]
        cbam_prec = cbam_metrics['precision_per_class'][i]
        diff = cbam_prec - orig_prec
        print(f"  {class_name:<20}: {orig_prec:<15.4f} -> {cbam_prec:<15.4f} ({diff:+.4f})")
    
    print(f"\n{'Per-Class Recall':<80}")
    print("-"*120)
    for i, class_name in enumerate(class_names):
        orig_recall = orig_metrics['recall_per_class'][i]
        cbam_recall = cbam_metrics['recall_per_class'][i]
        diff = cbam_recall - orig_recall
        print(f"  {class_name:<20}: {orig_recall:<15.4f} -> {cbam_recall:<15.4f} ({diff:+.4f})")
    
    print(f"\n{'Per-Class F1-Score':<80}")
    print("-"*120)
    for i, class_name in enumerate(class_names):
        orig_f1 = orig_metrics['f1_per_class'][i]
        cbam_f1 = cbam_metrics['f1_per_class'][i]
        diff = cbam_f1 - orig_f1
        print(f"  {class_name:<20}: {orig_f1:<15.4f} -> {cbam_f1:<15.4f} ({diff:+.4f})")

def plot_comprehensive_comparison(orig_metrics, cbam_metrics, orig_curves, cbam_curves, save_dir):
    """Create comprehensive comparison plots"""
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('YOLOv8 Original vs CBAM - Comprehensive Evaluation', fontsize=16, fontweight='bold')
    
    # 1. Training Curves - Accuracy and Loss
    ax1 = plt.subplot(3, 3, 1)
    epochs_orig, tl_orig, vl_orig, acc_orig = orig_curves
    epochs_cbam, tl_cbam, vl_cbam, acc_cbam = cbam_curves
    
    ax1.plot(epochs_orig, acc_orig, 'b-', label='Original - Top-1', linewidth=2)
    ax1.plot(epochs_cbam, acc_cbam, 'r-', label='CBAM - Top-1', linewidth=2)
    ax1.set_title('Top-1 Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.6, 1.0])
    
    # 2. Training Loss
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(epochs_orig, tl_orig, 'b-', label='Original - Train', linewidth=2)
    ax2.plot(epochs_orig, vl_orig, 'b--', label='Original - Val', linewidth=2)
    ax2.plot(epochs_cbam, tl_cbam, 'r-', label='CBAM - Train', linewidth=2)
    ax2.plot(epochs_cbam, vl_cbam, 'r--', label='CBAM - Val', linewidth=2)
    ax2.set_title('Training Loss vs Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Overall Metrics Comparison
    ax3 = plt.subplot(3, 3, 3)
    metrics_names = ['Top-1', 'Top-5', 'Macro F1']
    orig_vals = [orig_metrics['accuracy_top1'], orig_metrics['accuracy_top5'], orig_metrics['macro_f1']]
    cbam_vals = [cbam_metrics['accuracy_top1'], cbam_metrics['accuracy_top5'], cbam_metrics['macro_f1']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    ax3.bar(x - width/2, orig_vals, width, label='Original', color='blue', alpha=0.7)
    ax3.bar(x + width/2, cbam_vals, width, label='CBAM', color='red', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_names)
    ax3.set_title('Key Metrics Comparison', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Original Confusion Matrix
    ax4 = plt.subplot(3, 3, 4)
    im4 = ax4.imshow(orig_metrics['confusion_matrix'], cmap='Blues', interpolation='nearest')
    ax4.set_title('Original Confusion Matrix', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(num_classes))
    ax4.set_yticks(range(num_classes))
    ax4.set_xticklabels(class_names, rotation=45, ha='right')
    ax4.set_yticklabels(class_names)
    
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax4.text(j, i, orig_metrics['confusion_matrix'][i, j],
                           ha="center", va="center", color="white" if orig_metrics['confusion_matrix'][i, j] > orig_metrics['confusion_matrix'].max()/2 else "black")
    plt.colorbar(im4, ax=ax4)
    
    # 5. CBAM Confusion Matrix
    ax5 = plt.subplot(3, 3, 5)
    im5 = ax5.imshow(cbam_metrics['confusion_matrix'], cmap='Reds', interpolation='nearest')
    ax5.set_title('CBAM Confusion Matrix', fontsize=12, fontweight='bold')
    ax5.set_xticks(range(num_classes))
    ax5.set_yticks(range(num_classes))
    ax5.set_xticklabels(class_names, rotation=45, ha='right')
    ax5.set_yticklabels(class_names)
    
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax5.text(j, i, cbam_metrics['confusion_matrix'][i, j],
                           ha="center", va="center", color="white" if cbam_metrics['confusion_matrix'][i, j] > cbam_metrics['confusion_matrix'].max()/2 else "black")
    plt.colorbar(im5, ax=ax5)
    
    # 6. Per-Class Precision Comparison
    ax6 = plt.subplot(3, 3, 6)
    x = np.arange(num_classes)
    width = 0.35
    ax6.bar(x - width/2, orig_metrics['precision_per_class'], width, label='Original Precision', color='blue', alpha=0.7)
    ax6.bar(x + width/2, cbam_metrics['precision_per_class'], width, label='CBAM Precision', color='red', alpha=0.7)
    ax6.set_xticks(x)
    ax6.set_xticklabels(class_names, rotation=45, ha='right')
    ax6.set_title('Per-Class Precision Comparison', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Precision')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Per-Class Recall Comparison
    ax7 = plt.subplot(3, 3, 7)
    ax7.bar(x - width/2, orig_metrics['recall_per_class'], width, label='Original Recall', color='blue', alpha=0.6)
    ax7.bar(x + width/2, cbam_metrics['recall_per_class'], width, label='CBAM Recall', color='red', alpha=0.6)
    ax7.set_xticks(x)
    ax7.set_xticklabels(class_names, rotation=45, ha='right')
    ax7.set_title('Per-Class Recall Comparison', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Recall')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Per-Class F1-Score Comparison
    ax8 = plt.subplot(3, 3, 8)
    ax8.bar(x - width/2, orig_metrics['f1_per_class'], width, label='Original F1', color='blue', alpha=0.7)
    ax8.bar(x + width/2, cbam_metrics['f1_per_class'], width, label='CBAM F1', color='red', alpha=0.7)
    ax8.set_xticks(x)
    ax8.set_xticklabels(class_names, rotation=45, ha='right')
    ax8.set_title('Per-Class F1-Score Comparison', fontsize=12, fontweight='bold')
    ax8.set_ylabel('F1-Score')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Metric Improvement Analysis
    ax9 = plt.subplot(3, 3, 9)
    metrics_list = ['Top-1', 'Top-5', 'Macro Prec', 'Macro Rec', 'Macro F1']
    improvements = []
    
    improvements.append((cbam_metrics['accuracy_top1'] - orig_metrics['accuracy_top1']) / orig_metrics['accuracy_top1'] * 100)
    improvements.append((cbam_metrics['accuracy_top5'] - orig_metrics['accuracy_top5']) / orig_metrics['accuracy_top5'] * 100)
    improvements.append((cbam_metrics['macro_precision'] - orig_metrics['macro_precision']) / orig_metrics['macro_precision'] * 100)
    improvements.append((cbam_metrics['macro_recall'] - orig_metrics['macro_recall']) / orig_metrics['macro_recall'] * 100)
    improvements.append((cbam_metrics['macro_f1'] - orig_metrics['macro_f1']) / orig_metrics['macro_f1'] * 100)
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax9.bar(metrics_list, improvements, color=colors, alpha=0.7)
    ax9.axhline(y=0, color='black', linewidth=1)
    ax9.set_title('Metrics Improvement (%)', fontsize=12, fontweight='bold')
    ax9.set_ylabel('Improvement (%)')
    ax9.tick_params(axis='x', rotation=45)
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = f"{save_dir}/comprehensive_evaluation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Comprehensive evaluation plot saved to: {plot_path}")
    plt.close()

def print_summary(orig_metrics, cbam_metrics):
    """Print summary and recommendations"""
    print("\n" + "="*120)
    print("📈 Summary and Recommendations".center(120))
    print("="*120)
    
    # Calculate improvements
    acc_improve = (cbam_metrics['accuracy_top1'] - orig_metrics['accuracy_top1']) / orig_metrics['accuracy_top1'] * 100
    f1_improve = (cbam_metrics['macro_f1'] - orig_metrics['macro_f1']) / orig_metrics['macro_f1'] * 100
    
    print(f"\n1. Overall Performance:")
    if acc_improve > 0 and f1_improve > 0:
        print(f"   ✓ Excellent! CBAM significantly improved both accuracy (+{acc_improve:.2f}%) and F1-score (+{f1_improve:.2f}%)")
    elif acc_improve > 0:
        print(f"   ✓ Good! CBAM improved accuracy (+{acc_improve:.2f}%), but F1-score needs attention")
    elif f1_improve > 0:
        print(f"   ✓ CBAM improved F1-score (+{f1_improve:.2f}%), but accuracy didn't increase as expected")
    else:
        print(f"   ✗ CBAM didn't bring significant improvements, consider further tuning")
    
    print(f"\n2. Class-wise Analysis:")
    for i, class_name in enumerate(class_names):
        f1_change = cbam_metrics['f1_per_class'][i] - orig_metrics['f1_per_class'][i]
        if f1_change > 0.02:
            print(f"   ✓ {class_name}: F1-score improved by +{f1_change:.4f}")
        elif f1_change < -0.02:
            print(f"   ✗ {class_name}: F1-score decreased by {f1_change:.4f}")
    
    print(f"\n3. Test Samples: {orig_metrics['num_samples']}")

def main():
    print("="*120)
    print("🔬 YOLOv8 Original vs CBAM - Comprehensive Evaluation".center(120))
    print("="*120)
    
    # Load models
    print("\n📥 Loading models...")
    model_orig = YOLO(orig_model_path)
    model_cbam = YOLO(cbam_model_path)
    
    # Evaluate on test set
    print("\n🔍 Evaluating models on test set...")
    orig_metrics = evaluate_model(model_orig, test_data_path, class_names)
    cbam_metrics = evaluate_model(model_cbam, test_data_path, class_names)
    
    # Load training curves
    orig_results_csv = '/root/autodl-tmp/runs/classify/waste_cls_original/results.csv'
    cbam_results_csv = '/root/autodl-tmp/runs/classify/waste_cls_cbam-4/results.csv'
    
    orig_curves = load_training_curves(orig_results_csv)
    cbam_curves = load_training_curves(cbam_results_csv)
    
    # Print metrics
    print_metrics_comparison(orig_metrics, cbam_metrics)
    
    # Create plots
    save_dir = '/root/autodl-tmp/runs/classify'
    plot_comprehensive_comparison(orig_metrics, cbam_metrics, orig_curves, cbam_curves, save_dir)
    
    # Print summary
    print_summary(orig_metrics, cbam_metrics)
    
    print("\n" + "="*120)
    print("✅ Evaluation Complete!".center(120))
    print("="*120)

if __name__ == "__main__":
    main()
