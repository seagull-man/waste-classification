"""
YOLOv8 原始模型 vs CBAM模型 全面对比
"""
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 配置中文字体
import platform
system = platform.system()

if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
elif system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
else:  # Linux
    # 尝试常见中文字体
    chinese_fonts = [
        'Noto Sans CJK SC',
        'Source Han Sans SC',
        'WenQuanYi Micro Hei',
        'SimHei',
        'Microsoft YaHei',
        'DejaVu Sans'
    ]
    
    # 查找可用字体
    from matplotlib.font_manager import findfont, FontProperties
    available_fonts = []
    for font_name in chinese_fonts:
        try:
            findfont(FontProperties(family=font_name))
            available_fonts.append(font_name)
        except:
            continue
    
    if available_fonts:
        plt.rcParams['font.sans-serif'] = available_fonts
    else:
        # 如果没有中文字体，使用英文
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False

# 模型文件路径
orig_model_path = '/root/autodl-tmp/runs/classify/waste_cls_original/results.csv'
cbam_model_path = '/root/autodl-tmp/runs/classify/waste_cls_cbam-4/results.csv'

def load_data(file_path):
    """加载训练结果数据"""
    epochs = []
    train_loss = []
    val_loss = []
    acc_top1 = []
    acc_top5 = []
    times = []
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train/loss']))
            val_loss.append(float(row['val/loss']))
            acc_top1.append(float(row['metrics/accuracy_top1']))
            acc_top5.append(float(row['metrics/accuracy_top5']))
            times.append(float(row['time']))
    
    return epochs, train_loss, val_loss, acc_top1, acc_top5, times

def load_hyperparams(file_path):
    """加载超参数"""
    hyperparam_path = file_path.replace('results.csv', 'args.yaml')
    try:
        import yaml
        with open(hyperparam_path, 'r') as f:
            return yaml.safe_load(f)
    except:
        return {}

def compare_hyperparams(orig_hyperparams, cbam_hyperparams):
    """对比超参数"""
    keys = [
        'epochs',
        'batch',
        'imgsz',
        'optimizer',
        'lr0',
        'lrf',
        'cos_lr',
        'weight_decay',
        'warmup_epochs',
        'degrees',
        'translate',
        'scale',
        'mosaic',
        'mixup',
        'copy_paste'
    ]
    
    print("\n" + "="*100)
    print("🔧 超参数对比".center(100))
    print("="*100)
    print(f"{'参数':<25} {'原始模型':<30} {'CBAM模型':<30} {'差异':<15}")
    print("-"*100)
    
    for key in keys:
        orig_val = orig_hyperparams.get(key, '-')
        cbam_val = cbam_hyperparams.get(key, '-')
        
        try:
            if isinstance(orig_val, str) and orig_val.replace('.', '', 1).isdigit():
                orig_val = float(orig_val)
            if isinstance(cbam_val, str) and cbam_val.replace('.', '', 1).isdigit():
                cbam_val = float(cbam_val)
            
            if isinstance(orig_val, (int, float)) and isinstance(cbam_val, (int, float)):
                diff = f"{cbam_val - orig_val:+.4f}"
            else:
                diff = '-'
        except:
            diff = '-'
        
        orig_str = str(orig_val) if orig_val else '-'
        cbam_str = str(cbam_val) if cbam_val else '-'
        
        print(f"{key:<25} {orig_str:<30} {cbam_str:<30} {diff:<15}")

def extract_training_metrics(epochs, train_loss, val_loss, acc_top1, acc_top5, times):
    """提取训练指标"""
    # 计算每轮的实际时间（相邻epoch的时间差）
    if len(times) > 1:
        epoch_times = [times[0]]
        for i in range(1, len(times)):
            epoch_times.append(times[i] - times[i-1])
    else:
        epoch_times = times
    
    metrics = {
        'total_epochs': len(epochs),
        'final_train_loss': train_loss[-1],
        'final_val_loss': val_loss[-1],
        'final_acc_top1': acc_top1[-1],
        'final_acc_top5': acc_top5[-1],
        'best_val_loss': min(val_loss),
        'best_val_loss_epoch': val_loss.index(min(val_loss)) + 1,
        'best_acc_top1': max(acc_top1),
        'best_acc_top1_epoch': acc_top1.index(max(acc_top1)) + 1,
        'best_acc_top5': max(acc_top5),
        'best_acc_top5_epoch': acc_top5.index(max(acc_top5)) + 1,
        'total_training_time': times[-1],
        'avg_epoch_time': np.mean(epoch_times),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'acc_top1': acc_top1,
        'acc_top5': acc_top5
    }
    return metrics

def print_training_metrics_comparison(orig_metrics, cbam_metrics):
    """打印训练指标对比"""
    print("\n" + "="*100)
    print("📊 训练过程指标对比".center(100))
    print("="*100)
    
    print(f"{'指标':<30} {'原始模型':<20} {'CBAM模型':<20} {'差异':<15} {'提升率':<15}")
    print("-"*100)
    
    metrics_list = [
        ('总训练轮数', 'total_epochs', 'epochs'),
        ('最终训练损失', 'final_train_loss', 'loss'),
        ('最终验证损失', 'final_val_loss', 'loss'),
        ('最佳验证损失', 'best_val_loss', 'loss'),
        ('最佳验证损失轮数', 'best_val_loss_epoch', 'epochs'),
        ('最终Top-1准确率', 'final_acc_top1', 'percent'),
        ('最佳Top-1准确率', 'best_acc_top1', 'percent'),
        ('最佳Top-1准确率轮数', 'best_acc_top1_epoch', 'epochs'),
        ('最终Top-5准确率', 'final_acc_top5', 'percent'),
        ('最佳Top-5准确率', 'best_acc_top5', 'percent'),
        ('最佳Top-5准确率轮数', 'best_acc_top5_epoch', 'epochs'),
        ('总训练时间(s)', 'total_training_time', 'time'),
        ('平均每轮时间(s)', 'avg_epoch_time', 'time'),
    ]
    
    for name, key, value_type in metrics_list:
        orig_val = orig_metrics[key]
        cbam_val = cbam_metrics[key]
        
        diff = cbam_val - orig_val
        
        if value_type in ['percent', 'loss'] and orig_val != 0:
            improve = (diff / orig_val) * 100
            improve_str = f"{improve:+.2f}%"
        else:
            improve_str = '-'
        
        print(f"{name:<30} {orig_val:<20.6f} {cbam_val:<20.6f} {diff:<+15.6f} {improve_str:<15}")

def create_comparison_plots(orig_metrics, cbam_metrics, save_dir):
    """创建对比图表"""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('YOLOv8 原始模型 vs CBAM模型 训练对比', fontsize=16, fontweight='bold')
    
    # 1. Top-1准确率对比
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(range(1, len(orig_metrics['acc_top1']) + 1), orig_metrics['acc_top1'], 'b-', label='原始模型', linewidth=2, alpha=0.8)
    ax1.plot(range(1, len(cbam_metrics['acc_top1']) + 1), cbam_metrics['acc_top1'], 'r-', label='CBAM模型', linewidth=2, alpha=0.8)
    ax1.set_title('Top-1准确率对比', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.6, 1.0])
    
    # 2. 训练损失对比
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(range(1, len(orig_metrics['train_loss']) + 1), orig_metrics['train_loss'], 'b-', label='原始模型 - 训练损失', linewidth=2, alpha=0.8)
    ax2.plot(range(1, len(orig_metrics['val_loss']) + 1), orig_metrics['val_loss'], 'b--', label='原始模型 - 验证损失', linewidth=2, alpha=0.8)
    ax2.plot(range(1, len(cbam_metrics['train_loss']) + 1), cbam_metrics['train_loss'], 'r-', label='CBAM模型 - 训练损失', linewidth=2, alpha=0.8)
    ax2.plot(range(1, len(cbam_metrics['val_loss']) + 1), cbam_metrics['val_loss'], 'r--', label='CBAM模型 - 验证损失', linewidth=2, alpha=0.8)
    ax2.set_title('损失函数对比', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. 关键指标柱状图
    ax3 = plt.subplot(2, 3, 3)
    metrics = ['最终Top-1准确率', '最佳Top-1准确率', '训练速度']
    orig_values = [orig_metrics['final_acc_top1'], orig_metrics['best_acc_top1'], 1/orig_metrics['avg_epoch_time']]
    cbam_values = [cbam_metrics['final_acc_top1'], cbam_metrics['best_acc_top1'], 1/cbam_metrics['avg_epoch_time']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, orig_values, width, label='原始模型', color='blue', alpha=0.7)
    bars2 = ax3.bar(x + width/2, cbam_values, width, label='CBAM模型', color='red', alpha=0.7)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, rotation=15)
    ax3.set_title('关键性能指标对比', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Top-5准确率对比
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(range(1, len(orig_metrics['acc_top5']) + 1), orig_metrics['acc_top5'], 'b-', label='原始模型', linewidth=2, alpha=0.8)
    ax4.plot(range(1, len(cbam_metrics['acc_top5']) + 1), cbam_metrics['acc_top5'], 'r-', label='CBAM模型', linewidth=2, alpha=0.8)
    ax4.set_title('Top-5准确率对比', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0.9, 1.0])
    
    # 5. 准确率提升分析
    ax5 = plt.subplot(2, 3, 5)
    categories = ['最终Top-1', '最佳Top-1', '最终Top-5', '最佳Top-5']
    improvements = []
    
    for cat in categories:
        if cat == '最终Top-1':
            improve = (cbam_metrics['final_acc_top1'] - orig_metrics['final_acc_top1']) / orig_metrics['final_acc_top1'] * 100
        elif cat == '最佳Top-1':
            improve = (cbam_metrics['best_acc_top1'] - orig_metrics['best_acc_top1']) / orig_metrics['best_acc_top1'] * 100
        elif cat == '最终Top-5':
            improve = (cbam_metrics['final_acc_top5'] - orig_metrics['final_acc_top5']) / orig_metrics['final_acc_top5'] * 100
        else:  # 最佳Top-5
            improve = (cbam_metrics['best_acc_top5'] - orig_metrics['best_acc_top5']) / orig_metrics['best_acc_top5'] * 100
        improvements.append(improve)
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax5.bar(categories, improvements, color=colors, alpha=0.7)
    ax5.set_title('准确率提升率 (%)', fontsize=12, fontweight='bold')
    ax5.axhline(y=0, color='black', linewidth=1)
    ax5.set_ylabel('提升率 (%)')
    ax5.tick_params(axis='x', rotation=15)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 训练速度对比
    ax6 = plt.subplot(2, 3, 6)
    
    orig_cum_time = np.cumsum([orig_metrics['avg_epoch_time']] * len(orig_metrics['acc_top1']))
    cbam_cum_time = np.cumsum([cbam_metrics['avg_epoch_time']] * len(cbam_metrics['acc_top1']))
    
    ax6.plot(range(1, len(orig_cum_time)+1), orig_cum_time, 'b-', label='原始模型', linewidth=2)
    ax6.plot(range(1, len(cbam_cum_time)+1), cbam_cum_time, 'r-', label='CBAM模型', linewidth=2)
    ax6.set_title('累计训练时间对比', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('累计时间 (s)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{save_dir}/training_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 对比图表已保存到: {plot_path}")
    plt.close()

def print_conclusion(orig_metrics, cbam_metrics):
    """打印总结分析"""
    print("\n" + "="*100)
    print("📈 总结与分析".center(100))
    print("="*100)
    
    final_acc_improve = (cbam_metrics['final_acc_top1'] - orig_metrics['final_acc_top1']) / orig_metrics['final_acc_top1'] * 100
    best_acc_improve = (cbam_metrics['best_acc_top1'] - orig_metrics['best_acc_top1']) / orig_metrics['best_acc_top1'] * 100
    
    val_loss_change = (cbam_metrics['final_val_loss'] - orig_metrics['final_val_loss']) / orig_metrics['final_val_loss'] * 100
    
    time_change = (cbam_metrics['avg_epoch_time'] - orig_metrics['avg_epoch_time']) / orig_metrics['avg_epoch_time'] * 100
    
    print(f"\n1. 准确率方面:")
    print(f"   - 最终Top-1准确率: {'提升' if final_acc_improve > 0 else '下降'} {abs(final_acc_improve):.2f}%")
    print(f"   - 最佳Top-1准确率: {'提升' if best_acc_improve > 0 else '下降'} {abs(best_acc_improve):.2f}%")
    
    print(f"\n2. 损失函数方面:")
    print(f"   - 最终验证损失: {'降低' if val_loss_change < 0 else '增加'} {abs(val_loss_change):.2f}%")
    
    print(f"\n3. 训练效率方面:")
    print(f"   - 平均每轮时间: {'缩短' if time_change < 0 else '延长'} {abs(time_change):.2f}%")
    
    print(f"\n4. 总体评价:")
    if final_acc_improve > 0 and val_loss_change < 0:
        print(f"   ✓ CBAM注意力机制显著提升了模型性能，同时验证损失降低！")
    elif final_acc_improve > 0:
        print(f"   ✓ CBAM注意力机制提升了模型准确率，但需要进一步优化训练策略")
    elif val_loss_change < 0:
        print(f"   ✓ CBAM注意力机制有助于模型泛化，但准确率提升不明显")
    else:
        print(f"   ✗ CBAM注意力机制未带来明显性能提升，建议调整超参数")

def main():
    print("="*100)
    print("🔬 YOLOv8 原始模型 vs CBAM模型 全面对比分析".center(100))
    print("="*100)
    
    # 加载数据
    orig_epochs, orig_train_loss, orig_val_loss, orig_acc_top1, orig_acc_top5, orig_times = load_data(orig_model_path)
    cbam_epochs, cbam_train_loss, cbam_val_loss, cbam_acc_top1, cbam_acc_top5, cbam_times = load_data(cbam_model_path)
    
    # 提取指标
    orig_metrics = extract_training_metrics(orig_epochs, orig_train_loss, orig_val_loss, orig_acc_top1, orig_acc_top5, orig_times)
    cbam_metrics = extract_training_metrics(cbam_epochs, cbam_train_loss, cbam_val_loss, cbam_acc_top1, cbam_acc_top5, cbam_times)
    
    # 加载超参数
    orig_hyperparams = load_hyperparams(orig_model_path)
    cbam_hyperparams = load_hyperparams(cbam_model_path)
    
    # 打印对比
    compare_hyperparams(orig_hyperparams, cbam_hyperparams)
    print_training_metrics_comparison(orig_metrics, cbam_metrics)
    
    # 创建图表
    save_dir = '/root/autodl-tmp/runs/classify'
    create_comparison_plots(orig_metrics, cbam_metrics, save_dir)
    
    # 打印结论
    print_conclusion(orig_metrics, cbam_metrics)
    
    print("\n" + "="*100)
    print("✅ 分析完成！".center(100))
    print("="*100)

if __name__ == "__main__":
    main()
