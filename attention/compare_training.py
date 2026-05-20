"""
YOLOv8 原始模型 vs CBAM模型 训练参数对比与可视化
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ultralytics import YOLO

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 模型路径
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
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'accuracy': accuracy
    }

def load_training_data():
    """加载训练数据"""
    df_orig = pd.read_csv(ORIG_RESULTS_PATH)
    df_cbam = pd.read_csv(CBAM_RESULTS_PATH)

    return df_orig, df_cbam

def extract_key_metrics(df_orig, df_cbam):
    """提取关键指标"""
    metrics = {}

    df_orig.columns = df_orig.columns.str.strip()
    df_cbam.columns = df_cbam.columns.str.strip()

    orig_best_val_loss_idx = df_orig['val/loss'].idxmin()
    cbam_best_val_loss_idx = df_cbam['val/loss'].idxmin()

    orig_best_acc_idx = df_orig['metrics/accuracy_top1'].idxmax()
    cbam_best_acc_idx = df_cbam['metrics/accuracy_top1'].idxmax()

    metrics['original'] = {
        'total_epochs': len(df_orig),
        'final_train_loss': df_orig['train/loss'].iloc[-1],
        'final_val_loss': df_orig['val/loss'].iloc[-1],
        'final_acc': df_orig['metrics/accuracy_top1'].iloc[-1],
        'final_acc_top5': df_orig['metrics/accuracy_top5'].iloc[-1],
        'best_val_loss': df_orig['val/loss'].min(),
        'best_val_loss_epoch': orig_best_val_loss_idx + 1,
        'best_acc': df_orig['metrics/accuracy_top1'].max(),
        'best_acc_epoch': orig_best_acc_idx + 1,
        'total_time': df_orig['time'].sum(),
        'avg_epoch_time': df_orig['time'].mean(),
        'time_list': df_orig['time'].tolist(),
        'train_loss_list': df_orig['train/loss'].tolist(),
        'val_loss_list': df_orig['val/loss'].tolist(),
        'acc_list': df_orig['metrics/accuracy_top1'].tolist(),
    }

    metrics['cbam'] = {
        'total_epochs': len(df_cbam),
        'final_train_loss': df_cbam['train/loss'].iloc[-1],
        'final_val_loss': df_cbam['val/loss'].iloc[-1],
        'final_acc': df_cbam['metrics/accuracy_top1'].iloc[-1],
        'final_acc_top5': df_cbam['metrics/accuracy_top5'].iloc[-1],
        'best_val_loss': df_cbam['val/loss'].min(),
        'best_val_loss_epoch': cbam_best_val_loss_idx + 1,
        'best_acc': df_cbam['metrics/accuracy_top1'].max(),
        'best_acc_epoch': cbam_best_acc_idx + 1,
        'total_time': df_cbam['time'].sum(),
        'avg_epoch_time': df_cbam['time'].mean(),
        'time_list': df_cbam['time'].tolist(),
        'train_loss_list': df_cbam['train/loss'].tolist(),
        'val_loss_list': df_cbam['val/loss'].tolist(),
        'acc_list': df_cbam['metrics/accuracy_top1'].tolist(),
    }

    return metrics

def print_metrics_comparison(metrics, orig_eval, cbam_eval):
    """打印指标对比"""
    print("\n" + "="*100)
    print("📊 模型训练参数对比")
    print("="*100)

    print(f"\n{'指标':<25} {'原始模型':<20} {'CBAM模型':<20} {'差异':<15} {'提升率':<10}")
    print("-" * 100)

    m_orig = metrics['original']
    m_cbam = metrics['cbam']

    print(f"{'训练轮数':<25} {m_orig['total_epochs']:<20} {m_cbam['total_epochs']:<20} {m_cbam['total_epochs'] - m_orig['total_epochs']:<+15} {'':<10}")
    print(f"{'最终训练损失':<25} {m_orig['final_train_loss']:<20.5f} {m_cbam['final_train_loss']:<20.5f} {m_cbam['final_train_loss'] - m_orig['final_train_loss']:<+15.5f} {'':<10}")
    print(f"{'最终验证损失':<25} {m_orig['final_val_loss']:<20.5f} {m_cbam['final_val_loss']:<20.5f} {m_cbam['final_val_loss'] - m_orig['final_val_loss']:<+15.5f} {'':<10}")
    print(f"{'最佳验证损失':<25} {m_orig['best_val_loss']:<20.5f} {m_cbam['best_val_loss']:<20.5f} {m_cbam['best_val_loss'] - m_orig['best_val_loss']:<+15.5f} {'':<10}")
    print(f"{'最佳准确率':<25} {m_orig['best_acc']:<20.4f} {m_cbam['best_acc']:<20.4f} {m_cbam['best_acc'] - m_orig['best_acc']:<+15.4f} {'':<10}")
    
    print("\n" + "="*100)
    print("📈 测试集性能对比")
    print("="*100)
    print(f"\n{'指标':<25} {'原始模型':<20} {'CBAM模型':<20} {'差异':<15} {'提升率':<10}")
    print("-" * 100)
    
    print(f"{'测试Top-1准确率':<25} {orig_eval['accuracy']:<20.4f} {cbam_eval['accuracy']:<20.4f} {cbam_eval['accuracy'] - orig_eval['accuracy']:<+15.4f} {(cbam_eval['accuracy'] - orig_eval['accuracy']) / orig_eval['accuracy'] * 100 if orig_eval['accuracy'] > 0 else 0:>+8.2f}%")
    print(f"{'宏平均Precision':<25} {orig_eval['macro_precision']:<20.4f} {cbam_eval['macro_precision']:<20.4f} {cbam_eval['macro_precision'] - orig_eval['macro_precision']:<+15.4f} {(cbam_eval['macro_precision'] - orig_eval['macro_precision']) / orig_eval['macro_precision'] * 100 if orig_eval['macro_precision'] > 0 else 0:>+8.2f}%")
    print(f"{'宏平均Recall':<25} {orig_eval['macro_recall']:<20.4f} {cbam_eval['macro_recall']:<20.4f} {cbam_eval['macro_recall'] - orig_eval['macro_recall']:<+15.4f} {(cbam_eval['macro_recall'] - orig_eval['macro_recall']) / orig_eval['macro_recall'] * 100 if orig_eval['macro_recall'] > 0 else 0:>+8.2f}%")
    print(f"{'宏平均F1-Score':<25} {orig_eval['macro_f1']:<20.4f} {cbam_eval['macro_f1']:<20.4f} {cbam_eval['macro_f1'] - orig_eval['macro_f1']:<+15.4f} {(cbam_eval['macro_f1'] - orig_eval['macro_f1']) / orig_eval['macro_f1'] * 100 if orig_eval['macro_f1'] > 0 else 0:>+8.2f}%")
    
    print(f"\n{'类别':<20} {'指标':<15} {'原始模型':<15} {'CBAM模型':<15} {'差异':<15}")
    print("-"*80)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<20} {'Precision':<15} {orig_eval['precision_per_class'][i]:<15.4f} {cbam_eval['precision_per_class'][i]:<15.4f} {cbam_eval['precision_per_class'][i] - orig_eval['precision_per_class'][i]:<+15.4f}")
        print(f"{''.ljust(20)} {'Recall':<15} {orig_eval['recall_per_class'][i]:<15.4f} {cbam_eval['recall_per_class'][i]:<15.4f} {cbam_eval['recall_per_class'][i] - orig_eval['recall_per_class'][i]:<+15.4f}")
        print(f"{''.ljust(20)} {'F1-Score':<15} {orig_eval['f1_per_class'][i]:<15.4f} {cbam_eval['f1_per_class'][i]:<15.4f} {cbam_eval['f1_per_class'][i] - orig_eval['f1_per_class'][i]:<+15.4f}")

    print("\n" + "="*100)
    print("📈 测试集性能提升分析")
    print("="*100)

    acc_improvement = (cbam_eval['accuracy'] - orig_eval['accuracy']) / orig_eval['accuracy'] * 100
    f1_improvement = (cbam_eval['macro_f1'] - orig_eval['macro_f1']) / orig_eval['macro_f1'] * 100
    speed_diff = m_cbam['avg_epoch_time'] - m_orig['avg_epoch_time']

    print(f"\nTop-1准确率变化: {acc_improvement:+.2f}%")
    print(f"宏平均F1-Score变化: {f1_improvement:+.2f}%")
    print(f"训练速度差异: {speed_diff:+.2f} 秒/epoch")

    return m_orig, m_cbam

def plot_comparison_charts(m_orig, m_cbam, orig_eval, cbam_eval):
    """绘制对比图表"""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('YOLOv8 Original vs CBAM Model Comparison', fontsize=14, fontweight='bold')

    epochs_orig = range(1, len(m_orig['acc_list']) + 1)
    epochs_cbam = range(1, len(m_cbam['acc_list']) + 1)

    # 1. Top-1准确率曲线
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs_orig, m_orig['acc_list'], 'b-', label='Original', linewidth=2, alpha=0.8)
    ax1.plot(epochs_cbam, m_cbam['acc_list'], 'r-', label='CBAM', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Top-1 Accuracy Comparison (Train)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.7, 1.0])

    # 2. Loss曲线
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs_orig, m_orig['train_loss_list'], 'b-', label='Original Train', linewidth=2, alpha=0.8)
    ax2.plot(epochs_orig, m_orig['val_loss_list'], 'b--', label='Original Val', linewidth=2, alpha=0.8)
    ax2.plot(epochs_cbam, m_cbam['train_loss_list'], 'r-', label='CBAM Train', linewidth=2, alpha=0.8)
    ax2.plot(epochs_cbam, m_cbam['val_loss_list'], 'r--', label='CBAM Val', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training & Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Precision对比
    ax3 = plt.subplot(2, 3, 3)
    width = 0.35
    x = np.arange(NUM_CLASSES)
    ax3.bar(x - width/2, orig_eval['precision_per_class'], width, label='Original Precision', color='blue', alpha=0.7)
    ax3.bar(x + width/2, cbam_eval['precision_per_class'], width, label='CBAM Precision', color='red', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax3.set_title('Per-Class Precision Comparison')
    ax3.set_ylabel('Precision')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Recall对比
    ax4 = plt.subplot(2, 3, 4)
    ax4.bar(x - width/2, orig_eval['recall_per_class'], width, label='Original Recall', color='blue', alpha=0.7)
    ax4.bar(x + width/2, cbam_eval['recall_per_class'], width, label='CBAM Recall', color='red', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax4.set_title('Per-Class Recall Comparison')
    ax4.set_ylabel('Recall')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. F1-Score对比
    ax5 = plt.subplot(2, 3, 5)
    ax5.bar(x - width/2, orig_eval['f1_per_class'], width, label='Original F1', color='blue', alpha=0.7)
    ax5.bar(x + width/2, cbam_eval['f1_per_class'], width, label='CBAM F1', color='red', alpha=0.7)
    ax5.set_xticks(x)
    ax5.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax5.set_title('Per-Class F1-Score Comparison')
    ax5.set_ylabel('F1-Score')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. 总体指标对比
    ax6 = plt.subplot(2, 3, 6)
    overall_metrics = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']
    orig_values = [orig_eval['accuracy'], orig_eval['macro_precision'], orig_eval['macro_recall'], orig_eval['macro_f1']]
    cbam_values = [cbam_eval['accuracy'], cbam_eval['macro_precision'], cbam_eval['macro_recall'], cbam_eval['macro_f1']]
    x_overall = np.arange(len(overall_metrics))
    ax6.bar(x_overall - width/2, orig_values, width, label='Original', color='blue', alpha=0.7)
    ax6.bar(x_overall + width/2, cbam_values, width, label='CBAM', color='red', alpha=0.7)
    ax6.set_xticks(x_overall)
    ax6.set_xticklabels(overall_metrics, rotation=45, ha='right')
    ax6.set_title('Overall Metrics Comparison')
    ax6.set_ylabel('Score')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = '/root/autodl-tmp/runs/classify/model_comparison_charts.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 图表已保存到: {output_path}")

    plt.close()

def generate_report(metrics, m_orig, m_cbam, orig_eval, cbam_eval):
    """生成对比报告"""
    report = f"""
================================================================================
                    YOLOv8 + CBAM 模型对比评估报告
================================================================================

一、实验设置
--------------------------------------------------------------------------------
数据集: garbage_4cls (4类生活垃圾分类)
基础模型: YOLOv8n-cls
对比模型: YOLOv8n-cls + CBAM注意力机制
训练参数: batch=32, epochs=100, imgsz=320, optimizer=AdamW

二、训练结果对比
--------------------------------------------------------------------------------
{'指标':<25} {'原始模型':<20} {'CBAM模型':<20} {'差异':<15}
{'训练轮数':<25} {m_orig['total_epochs']:<20} {m_cbam['total_epochs']:<20} {m_cbam['total_epochs'] - m_orig['total_epochs']:<+15}
{'最终训练损失':<25} {m_orig['final_train_loss']:<20.5f} {m_cbam['final_train_loss']:<20.5f} {m_cbam['final_train_loss'] - m_orig['final_train_loss']:<+15.5f}
{'最终验证损失':<25} {m_orig['final_val_loss']:<20.5f} {m_cbam['final_val_loss']:<20.5f} {m_cbam['final_val_loss'] - m_orig['final_val_loss']:<+15.5f}
{'最佳验证损失':<25} {m_orig['best_val_loss']:<20.5f} {m_cbam['best_val_loss']:<20.5f} {m_cbam['best_val_loss'] - m_orig['best_val_loss']:<+15.5f}
{'最佳验证损失Epoch':<25} {m_orig['best_val_loss_epoch']:<20} {m_cbam['best_val_loss_epoch']:<20} {m_cbam['best_val_loss_epoch'] - m_orig['best_val_loss_epoch']:<+15}
{'最佳准确率':<25} {m_orig['best_acc']:<20.4f} {m_cbam['best_acc']:<20.4f} {m_cbam['best_acc'] - m_orig['best_acc']:<+15.4f}

三、测试集性能对比
--------------------------------------------------------------------------------
{'指标':<25} {'原始模型':<20} {'CBAM模型':<20} {'差异':<15} {'提升率':<10}
{'Top-1准确率':<25} {orig_eval['accuracy']:<20.4f} {cbam_eval['accuracy']:<20.4f} {cbam_eval['accuracy'] - orig_eval['accuracy']:<+15.4f} {(cbam_eval['accuracy'] - orig_eval['accuracy']) / orig_eval['accuracy'] * 100:>+8.2f}%
{'宏平均Precision':<25} {orig_eval['macro_precision']:<20.4f} {cbam_eval['macro_precision']:<20.4f} {cbam_eval['macro_precision'] - orig_eval['macro_precision']:<+15.4f} {(cbam_eval['macro_precision'] - orig_eval['macro_precision']) / orig_eval['macro_precision'] * 100:>+8.2f}%
{'宏平均Recall':<25} {orig_eval['macro_recall']:<20.4f} {cbam_eval['macro_recall']:<20.4f} {cbam_eval['macro_recall'] - orig_eval['macro_recall']:<+15.4f} {(cbam_eval['macro_recall'] - orig_eval['macro_recall']) / orig_eval['macro_recall'] * 100:>+8.2f}%
{'宏平均F1-Score':<25} {orig_eval['macro_f1']:<20.4f} {cbam_eval['macro_f1']:<20.4f} {cbam_eval['macro_f1'] - orig_eval['macro_f1']:<+15.4f} {(cbam_eval['macro_f1'] - orig_eval['macro_f1']) / orig_eval['macro_f1'] * 100:>+8.2f}%

四、各类别详细指标
--------------------------------------------------------------------------------
"""
    for i, name in enumerate(CLASS_NAMES):
        report += f"\n{name}:\n"
        report += f"  Precision: {orig_eval['precision_per_class'][i]:.4f} -> {cbam_eval['precision_per_class'][i]:.4f} ({cbam_eval['precision_per_class'][i] - orig_eval['precision_per_class'][i]:+.4f})\n"
        report += f"  Recall:    {orig_eval['recall_per_class'][i]:.4f} -> {cbam_eval['recall_per_class'][i]:.4f} ({cbam_eval['recall_per_class'][i] - orig_eval['recall_per_class'][i]:+.4f})\n"
        report += f"  F1-Score:  {orig_eval['f1_per_class'][i]:.4f} -> {cbam_eval['f1_per_class'][i]:.4f} ({cbam_eval['f1_per_class'][i] - orig_eval['f1_per_class'][i]:+.4f})\n"

    report += f"\n五、性能分析\n--------------------------------------------------------------------------------\n"

    acc_diff = cbam_eval['accuracy'] - orig_eval['accuracy']
    f1_diff = cbam_eval['macro_f1'] - orig_eval['macro_f1']

    if acc_diff > 0:
        report += f"✓ CBAM模型准确率优于原始模型，提升 {acc_diff:.4f} ({acc_diff/orig_eval['accuracy']*100:.2f}%)\n"
    else:
        report += f"✗ CBAM模型准确率低于原始模型，降低 {abs(acc_diff):.4f}\n"

    if f1_diff > 0:
        report += f"✓ CBAM模型F1-Score优于原始模型，提升 {f1_diff:.4f} ({f1_diff/orig_eval['macro_f1']*100:.2f}%)\n"
    else:
        report += f"✗ CBAM模型F1-Score低于原始模型，降低 {abs(f1_diff):.4f}\n"

    report += f"\n六、结论\n--------------------------------------------------------------------------------\n"

    if acc_diff > 0 and f1_diff > 0:
        report += "CBAM注意力机制有效提升了模型性能，在准确率和F1-Score两方面均有改善。\n"
    elif acc_diff > 0:
        report += "CBAM注意力机制提升了模型准确率，F1-Score变化需进一步分析。\n"
    elif f1_diff > 0:
        report += "CBAM注意力机制提升了F1-Score，但准确率未明显提升，可能存在类别不平衡问题。\n"
    else:
        report += "CBAM注意力机制对模型性能无明显改善作用。\n"

    report += f"""
================================================================================
报告生成时间: 2026-04-27
================================================================================
"""

    report_path = '/root/autodl-tmp/runs/classify/cbam_model_comparison_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"📄 报告已保存到: {report_path}")

def main():
    print("="*100)
    print("🔬 YOLOv8 原始模型 vs CBAM模型 全面对比分析")
    print("="*100)
    
    print("\n📥 正在加载训练数据...")
    df_orig, df_cbam = load_training_data()
    metrics = extract_key_metrics(df_orig, df_cbam)
    
    print("\n🔍 正在评估模型在测试集上的性能...")
    orig_eval = evaluate_model(ORIG_MODEL_PATH)
    cbam_eval = evaluate_model(CBAM_MODEL_PATH)
    
    m_orig = metrics['original']
    m_cbam = metrics['cbam']
    print_metrics_comparison(metrics, orig_eval, cbam_eval)
    
    print("\n📈 正在绘制对比图表...")
    plot_comparison_charts(m_orig, m_cbam, orig_eval, cbam_eval)
    
    print("\n📝 正在生成报告...")
    generate_report(metrics, m_orig, m_cbam, orig_eval, cbam_eval)

    print("\n" + "="*100)
    print("✅ 分析完成！")
    print("="*100)

if __name__ == "__main__":
    main()
