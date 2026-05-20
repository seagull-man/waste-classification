"""
单个模型训练分析与可视化
为原始模型和CBAM模型分别生成训练曲线图和参数报告
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ultralytics import YOLO

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置
ORIG_MODEL_PATH = '/root/autodl-tmp/runs/classify/waste_cls_original-6/weights/best.pt'
ORIG_RESULTS_PATH = '/root/autodl-tmp/runs/classify/waste_cls_original-6/results.csv'
CBAM_MODEL_PATH = '/root/autodl-tmp/runs/classify/waste_cls_cbam-4/weights/best.pt'
CBAM_RESULTS_PATH = '/root/autodl-tmp/runs/classify/waste_cls_cbam-4/results.csv'
TEST_DATA_PATH = '/root/autodl-tmp/garbage_4cls/test'
CLASS_NAMES = ['hazardous', 'kitchen', 'other', 'recyclable']
NUM_CLASSES = len(CLASS_NAMES)

def evaluate_model(model_path):
    """评估模型获取precision, recall, f1-score"""
    model = YOLO(model_path)
    
    all_preds = []
    all_labels = []
    
    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = Path(TEST_DATA_PATH) / class_name
        if not class_dir.exists():
            continue
        
        img_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
        for img_path in img_files:
            try:
                results = model(str(img_path), verbose=False)
                pred_class = int(results[0].probs.top1)
                all_preds.append(pred_class)
                all_labels.append(idx)
            except Exception as e:
                continue
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # 计算混淆矩阵
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        conf_matrix[true][pred] += 1
    
    # 计算每个类别的指标
    precision_per_class = np.zeros(NUM_CLASSES)
    recall_per_class = np.zeros(NUM_CLASSES)
    f1_per_class = np.zeros(NUM_CLASSES)
    
    for i in range(NUM_CLASSES):
        tp = conf_matrix[i][i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        
        precision_per_class[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_per_class[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_per_class[i] = 2 * (precision_per_class[i] * recall_per_class[i]) / (precision_per_class[i] + recall_per_class[i]) if (precision_per_class[i] + recall_per_class[i]) > 0 else 0
    
    # 计算宏平均
    macro_precision = precision_per_class.mean()
    macro_recall = recall_per_class.mean()
    macro_f1 = f1_per_class.mean()
    
    # 计算Top-1准确率
    accuracy = (all_labels == all_preds).mean()
    
    return {
        'confusion_matrix': conf_matrix,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'accuracy': accuracy
    }

def extract_key_metrics(df):
    """提取关键指标"""
    df.columns = df.columns.str.strip()
    
    best_val_loss_idx = df['val/loss'].idxmin()
    best_acc_idx = df['metrics/accuracy_top1'].idxmax()
    
    metrics = {
        'total_epochs': len(df),
        'final_train_loss': df['train/loss'].iloc[-1],
        'final_val_loss': df['val/loss'].iloc[-1],
        'final_acc': df['metrics/accuracy_top1'].iloc[-1],
        'final_acc_top5': df['metrics/accuracy_top5'].iloc[-1] if 'metrics/accuracy_top5' in df.columns else df['metrics/accuracy_top1'].iloc[-1],
        'best_val_loss': df['val/loss'].min(),
        'best_val_loss_epoch': best_val_loss_idx + 1,
        'best_acc': df['metrics/accuracy_top1'].max(),
        'best_acc_epoch': best_acc_idx + 1,
        'total_time': df['time'].sum(),
        'avg_epoch_time': df['time'].mean(),
        'time_list': df['time'].tolist(),
        'train_loss_list': df['train/loss'].tolist(),
        'val_loss_list': df['val/loss'].tolist(),
        'acc_list': df['metrics/accuracy_top1'].tolist(),
        'acc_top5_list': df['metrics/accuracy_top5'].tolist() if 'metrics/accuracy_top5' in df.columns else df['metrics/accuracy_top1'].tolist()
    }
    
    return metrics

def plot_single_model_results(metrics, model_name, save_path):
    """为单个模型绘制训练曲线图"""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{model_name} Training Analysis', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(metrics['acc_list']) + 1)
    
    # 1. Top-1 & Top-5 准确率曲线
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(epochs, metrics['acc_list'], 'b-', label='Top-1 Accuracy', linewidth=2, alpha=0.8)
    ax1.plot(epochs, metrics['acc_top5_list'], 'g-', label='Top-5 Accuracy', linewidth=2, alpha=0.8)
    ax1.axvline(x=metrics['best_acc_epoch'], color='r', linestyle='--', label=f'Best Acc Epoch ({metrics["best_acc_epoch"]})', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.6, 1.0])
    
    # 2. 训练与验证Loss曲线
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(epochs, metrics['train_loss_list'], 'b-', label='Train Loss', linewidth=2, alpha=0.8)
    ax2.plot(epochs, metrics['val_loss_list'], 'r--', label='Val Loss', linewidth=2, alpha=0.8)
    ax2.axvline(x=metrics['best_val_loss_epoch'], color='orange', linestyle='--', label=f'Best Val Loss Epoch ({metrics["best_val_loss_epoch"]})', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 训练时间分布
    ax3 = plt.subplot(2, 2, 3)
    ax3.bar(epochs, metrics['time_list'], color='blue', alpha=0.6)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title(f'Training Time per Epoch (Avg: {metrics["avg_epoch_time"]:.2f}s)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 累计训练时间
    ax4 = plt.subplot(2, 2, 4)
    time_cumsum = np.cumsum(metrics['time_list'])
    ax4.plot(epochs, time_cumsum / 60, 'purple', linewidth=2, alpha=0.8)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Cumulative Time (minutes)')
    ax4.set_title(f'Cumulative Training Time (Total: {time_cumsum[-1]/60:.2f} min)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练曲线图已保存到: {save_path}")

def plot_model_performance(eval_results, model_name, save_path):
    """为单个模型绘制性能指标图"""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{model_name} Performance Evaluation', fontsize=16, fontweight='bold')
    
    # 1. 混淆矩阵
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(eval_results['confusion_matrix'], cmap='Blues', interpolation='nearest')
    ax1.set_title('Confusion Matrix')
    ax1.set_xticks(range(NUM_CLASSES))
    ax1.set_yticks(range(NUM_CLASSES))
    ax1.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax1.set_yticklabels(CLASS_NAMES)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            text = ax1.text(j, i, eval_results['confusion_matrix'][i, j],
                           ha="center", va="center", color="white" if eval_results['confusion_matrix'][i, j] > eval_results['confusion_matrix'].max()/2 else "black")
    plt.colorbar(im1, ax=ax1)
    
    # 2. Precision, Recall, F1-Score 柱状图
    ax2 = plt.subplot(2, 2, 2)
    x = np.arange(NUM_CLASSES)
    width = 0.25
    
    ax2.bar(x - width, eval_results['precision_per_class'], width, label='Precision', color='blue', alpha=0.7)
    ax2.bar(x, eval_results['recall_per_class'], width, label='Recall', color='green', alpha=0.7)
    ax2.bar(x + width, eval_results['f1_per_class'], width, label='F1-Score', color='red', alpha=0.7)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax2.set_title('Precision, Recall, F1-Score per Class')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.0])
    
    # 3. 宏平均指标
    ax3 = plt.subplot(2, 2, 3)
    macro_metrics = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']
    macro_values = [eval_results['accuracy'], eval_results['macro_precision'], eval_results['macro_recall'], eval_results['macro_f1']]
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']
    
    bars = ax3.bar(macro_metrics, macro_values, color=colors, alpha=0.7)
    ax3.set_title('Overall Metrics')
    ax3.set_ylabel('Score')
    ax3.set_ylim([0, 1.0])
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, macro_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height, f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Radar Chart
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    angles = np.linspace(0, 2 * np.pi, NUM_CLASSES, endpoint=False).tolist()
    angles += angles[:1]
    
    precisions = eval_results['precision_per_class'].tolist()
    precisions += precisions[:1]
    recalls = eval_results['recall_per_class'].tolist()
    recalls += recalls[:1]
    f1s = eval_results['f1_per_class'].tolist()
    f1s += f1s[:1]
    
    ax4.plot(angles, precisions, 'b-', label='Precision', linewidth=2, alpha=0.7)
    ax4.fill(angles, precisions, 'b', alpha=0.3)
    ax4.plot(angles, recalls, 'g-', label='Recall', linewidth=2, alpha=0.7)
    ax4.fill(angles, recalls, 'g', alpha=0.3)
    ax4.plot(angles, f1s, 'r-', label='F1-Score', linewidth=2, alpha=0.7)
    ax4.fill(angles, f1s, 'r', alpha=0.3)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(CLASS_NAMES)
    ax4.set_title('Class-wise Performance Radar Chart')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 性能评估图已保存到: {save_path}")

def generate_model_report(metrics, eval_results, model_name, save_path):
    """生成单个模型的详细参数报告"""
    report = f"""
================================================================================
                    {model_name} 训练与评估报告
================================================================================

一、模型信息
--------------------------------------------------------------------------------
模型名称: {model_name}
数据集: garbage_4cls (4类生活垃圾分类)
类别: {', '.join(CLASS_NAMES)}

二、训练参数与结果
--------------------------------------------------------------------------------
训练轮数: {metrics['total_epochs']} epochs

最终结果:
- 训练损失: {metrics['final_train_loss']:.5f}
- 验证损失: {metrics['final_val_loss']:.5f}
- Top-1准确率: {metrics['final_acc']:.4f} ({metrics['final_acc']*100:.2f}%)
- Top-5准确率: {metrics['final_acc_top5']:.4f} ({metrics['final_acc_top5']*100:.2f}%)

最佳结果:
- 最佳验证损失: {metrics['best_val_loss']:.5f} (epoch {metrics['best_val_loss_epoch']})
- 最佳Top-1准确率: {metrics['best_acc']:.4f} ({metrics['best_acc']*100:.2f}%) (epoch {metrics['best_acc_epoch']})

训练时间:
- 总训练时间: {metrics['total_time']:.2f} 秒 ({metrics['total_time']/60:.2f} 分钟)
- 平均每轮时间: {metrics['avg_epoch_time']:.2f} 秒/epoch

三、测试集性能评估
--------------------------------------------------------------------------------
总体指标:
- Top-1准确率: {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)
- 宏平均Precision: {eval_results['macro_precision']:.4f}
- 宏平均Recall: {eval_results['macro_recall']:.4f}
- 宏平均F1-Score: {eval_results['macro_f1']:.4f}

各类别详细指标:
"""
    
    for i, name in enumerate(CLASS_NAMES):
        report += f"\n{name}:"
        report += f"\n  Precision: {eval_results['precision_per_class'][i]:.4f}"
        report += f"\n  Recall:    {eval_results['recall_per_class'][i]:.4f}"
        report += f"\n  F1-Score:  {eval_results['f1_per_class'][i]:.4f}"
    
    report += f"\n\n四、混淆矩阵\n--------------------------------------------------------------------------------\n"
    report += "           " + "  ".join([f"{name:<10}" for name in CLASS_NAMES]) + "\n"
    for i, true_name in enumerate(CLASS_NAMES):
        report += f"{true_name:<10}"
        for j, pred_name in enumerate(CLASS_NAMES):
            report += f"{eval_results['confusion_matrix'][i, j]:>10}"
        report += "\n"
    
    report += f"""
五、总结
--------------------------------------------------------------------------------
本报告展示了 {model_name} 在生活垃圾分类任务上的完整训练与评估结果。
- 模型在训练过程中共经历 {metrics['total_epochs']} 轮训练
- 最佳Top-1准确率达到 {metrics['best_acc']*100:.2f}%
- 在测试集上达到 {eval_results['accuracy']*100:.2f}% 的Top-1准确率
- 宏平均F1-Score为 {eval_results['macro_f1']:.4f}
- 总训练时间为 {metrics['total_time']/60:.2f} 分钟

================================================================================
报告生成时间: 2026-04-27
================================================================================
"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 详细报告已保存到: {save_path}")

def main():
    print("="*100)
    print("🔬 单个模型训练与性能分析")
    print("="*100)
    
    # ====== 处理原始模型 ======
    print(f"\n{'='*100}")
    print(f"📊 分析原始模型 (YOLOv8 Original)")
    print(f"{'='*100}")
    
    # 加载训练数据
    orig_df = pd.read_csv(ORIG_RESULTS_PATH)
    orig_metrics = extract_key_metrics(orig_df)
    
    # 评估模型
    print("\n🔍 评估原始模型在测试集上的性能...")
    orig_eval = evaluate_model(ORIG_MODEL_PATH)
    
    # 生成训练曲线图
    orig_train_chart_path = '/root/autodl-tmp/runs/classify/original_model_training_chart.png'
    plot_single_model_results(orig_metrics, 'YOLOv8 Original Model', orig_train_chart_path)
    
    # 生成性能评估图
    orig_perf_chart_path = '/root/autodl-tmp/runs/classify/original_model_performance_chart.png'
    plot_model_performance(orig_eval, 'YOLOv8 Original Model', orig_perf_chart_path)
    
    # 生成详细报告
    orig_report_path = '/root/autodl-tmp/runs/classify/original_model_report.txt'
    generate_model_report(orig_metrics, orig_eval, 'YOLOv8 Original', orig_report_path)
    
    # ====== 处理CBAM模型 ======
    print(f"\n{'='*100}")
    print(f"📊 分析CBAM模型 (YOLOv8 + CBAM)")
    print(f"{'='*100}")
    
    # 加载训练数据
    cbam_df = pd.read_csv(CBAM_RESULTS_PATH)
    cbam_metrics = extract_key_metrics(cbam_df)
    
    # 评估模型
    print("\n🔍 评估CBAM模型在测试集上的性能...")
    cbam_eval = evaluate_model(CBAM_MODEL_PATH)
    
    # 生成训练曲线图
    cbam_train_chart_path = '/root/autodl-tmp/runs/classify/cbam_model_training_chart.png'
    plot_single_model_results(cbam_metrics, 'YOLOv8 + CBAM Model', cbam_train_chart_path)
    
    # 生成性能评估图
    cbam_perf_chart_path = '/root/autodl-tmp/runs/classify/cbam_model_performance_chart.png'
    plot_model_performance(cbam_eval, 'YOLOv8 + CBAM Model', cbam_perf_chart_path)
    
    # 生成详细报告
    cbam_report_path = '/root/autodl-tmp/runs/classify/cbam_model_report.txt'
    generate_model_report(cbam_metrics, cbam_eval, 'YOLOv8 + CBAM', cbam_report_path)
    
    # 总结
    print(f"\n{'='*100}")
    print(f"✅ 所有分析完成！")
    print(f"{'='*100}")
    print(f"\n📁 生成的文件:")
    print(f"  原始模型训练曲线图: {orig_train_chart_path}")
    print(f"  原始模型性能评估图: {orig_perf_chart_path}")
    print(f"  原始模型详细报告: {orig_report_path}")
    print(f"  CBAM模型训练曲线图: {cbam_train_chart_path}")
    print(f"  CBAM模型性能评估图: {cbam_perf_chart_path}")
    print(f"  CBAM模型详细报告: {cbam_report_path}")
    print(f"\n📊 关键结果对比:")
    print(f"  原始模型准确率: {orig_eval['accuracy']:.4f} ({orig_eval['accuracy']*100:.2f}%)")
    print(f"  CBAM模型准确率: {cbam_eval['accuracy']:.4f} ({cbam_eval['accuracy']*100:.2f}%)")
    print(f"  准确率变化: {cbam_eval['accuracy'] - orig_eval['accuracy']:+.4f} ({(cbam_eval['accuracy'] - orig_eval['accuracy']) / orig_eval['accuracy'] * 100:+.2f}%)")

if __name__ == "__main__":
    main()
