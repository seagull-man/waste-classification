"""
YOLOv8 Original vs CBAM Model Complete Comparison
"""
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# Model file paths
orig_model_path = '/root/autodl-tmp/runs/classify/waste_cls_original/results.csv'
cbam_model_path = '/root/autodl-tmp/runs/classify/waste_cls_cbam-4/results.csv'

def load_data(file_path):
    """Load training results data"""
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
    """Load hyperparameters"""
    hyperparam_path = file_path.replace('results.csv', 'args.yaml')
    try:
        import yaml
        with open(hyperparam_path, 'r') as f:
            return yaml.safe_load(f)
    except:
        return {}

def compare_hyperparams(orig_hyperparams, cbam_hyperparams):
    """Compare hyperparameters"""
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
    print("🔧 Hyperparameters Comparison".center(100))
    print("="*100)
    print(f"{'Parameter':<25} {'Original':<30} {'CBAM':<30} {'Difference':<15}")
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
    """Extract training metrics"""
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
    """Print training metrics comparison"""
    print("\n" + "="*100)
    print("📊 Training Process Metrics Comparison".center(100))
    print("="*100)
    
    print(f"{'Metric':<30} {'Original':<20} {'CBAM':<20} {'Difference':<15} {'Improvement':<15}")
    print("-"*100)
    
    metrics_list = [
        ('Total Epochs', 'total_epochs', 'epochs'),
        ('Final Train Loss', 'final_train_loss', 'loss'),
        ('Final Val Loss', 'final_val_loss', 'loss'),
        ('Best Val Loss', 'best_val_loss', 'loss'),
        ('Best Val Loss Epoch', 'best_val_loss_epoch', 'epochs'),
        ('Final Top-1 Accuracy', 'final_acc_top1', 'percent'),
        ('Best Top-1 Accuracy', 'best_acc_top1', 'percent'),
        ('Best Top-1 Acc Epoch', 'best_acc_top1_epoch', 'epochs'),
        ('Final Top-5 Accuracy', 'final_acc_top5', 'percent'),
        ('Best Top-5 Accuracy', 'best_acc_top5', 'percent'),
        ('Best Top-5 Acc Epoch', 'best_acc_top5_epoch', 'epochs'),
        ('Total Training Time (s)', 'total_training_time', 'time'),
        ('Avg Epoch Time (s)', 'avg_epoch_time', 'time'),
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
    """Create comparison plots"""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('YOLOv8 Original vs CBAM Model Training Comparison', fontsize=16, fontweight='bold')
    
    # 1. Top-1 Accuracy Comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(range(1, len(orig_metrics['acc_top1']) + 1), orig_metrics['acc_top1'], 'b-', label='Original', linewidth=2, alpha=0.8)
    ax1.plot(range(1, len(cbam_metrics['acc_top1']) + 1), cbam_metrics['acc_top1'], 'r-', label='CBAM', linewidth=2, alpha=0.8)
    ax1.set_title('Top-1 Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.6, 1.0])
    
    # 2. Loss Comparison
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(range(1, len(orig_metrics['train_loss']) + 1), orig_metrics['train_loss'], 'b-', label='Original Train', linewidth=2, alpha=0.8)
    ax2.plot(range(1, len(orig_metrics['val_loss']) + 1), orig_metrics['val_loss'], 'b--', label='Original Val', linewidth=2, alpha=0.8)
    ax2.plot(range(1, len(cbam_metrics['train_loss']) + 1), cbam_metrics['train_loss'], 'r-', label='CBAM Train', linewidth=2, alpha=0.8)
    ax2.plot(range(1, len(cbam_metrics['val_loss']) + 1), cbam_metrics['val_loss'], 'r--', label='CBAM Val', linewidth=2, alpha=0.8)
    ax2.set_title('Loss Function Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Key Metrics Bar Chart
    ax3 = plt.subplot(2, 3, 3)
    metrics = ['Final Top-1', 'Best Top-1', 'Training Speed']
    orig_values = [orig_metrics['final_acc_top1'], orig_metrics['best_acc_top1'], 1/orig_metrics['avg_epoch_time']]
    cbam_values = [cbam_metrics['final_acc_top1'], cbam_metrics['best_acc_top1'], 1/cbam_metrics['avg_epoch_time']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, orig_values, width, label='Original', color='blue', alpha=0.7)
    bars2 = ax3.bar(x + width/2, cbam_values, width, label='CBAM', color='red', alpha=0.7)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, rotation=15)
    ax3.set_title('Key Performance Metrics', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Top-5 Accuracy Comparison
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(range(1, len(orig_metrics['acc_top5']) + 1), orig_metrics['acc_top5'], 'b-', label='Original', linewidth=2, alpha=0.8)
    ax4.plot(range(1, len(cbam_metrics['acc_top5']) + 1), cbam_metrics['acc_top5'], 'r-', label='CBAM', linewidth=2, alpha=0.8)
    ax4.set_title('Top-5 Accuracy Comparison', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0.9, 1.0])
    
    # 5. Accuracy Improvement Analysis
    ax5 = plt.subplot(2, 3, 5)
    categories = ['Final Top-1', 'Best Top-1', 'Final Top-5', 'Best Top-5']
    improvements = []
    
    for cat in categories:
        if cat == 'Final Top-1':
            improve = (cbam_metrics['final_acc_top1'] - orig_metrics['final_acc_top1']) / orig_metrics['final_acc_top1'] * 100
        elif cat == 'Best Top-1':
            improve = (cbam_metrics['best_acc_top1'] - orig_metrics['best_acc_top1']) / orig_metrics['best_acc_top1'] * 100
        elif cat == 'Final Top-5':
            improve = (cbam_metrics['final_acc_top5'] - orig_metrics['final_acc_top5']) / orig_metrics['final_acc_top5'] * 100
        else:  # Best Top-5
            improve = (cbam_metrics['best_acc_top5'] - orig_metrics['best_acc_top5']) / orig_metrics['best_acc_top5'] * 100
        improvements.append(improve)
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax5.bar(categories, improvements, color=colors, alpha=0.7)
    ax5.set_title('Accuracy Improvement Rate (%)', fontsize=12, fontweight='bold')
    ax5.axhline(y=0, color='black', linewidth=1)
    ax5.set_ylabel('Improvement (%)')
    ax5.tick_params(axis='x', rotation=15)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Training Speed Comparison
    ax6 = plt.subplot(2, 3, 6)
    
    orig_cum_time = np.cumsum([orig_metrics['avg_epoch_time']] * len(orig_metrics['acc_top1']))
    cbam_cum_time = np.cumsum([cbam_metrics['avg_epoch_time']] * len(cbam_metrics['acc_top1']))
    
    ax6.plot(range(1, len(orig_cum_time)+1), orig_cum_time, 'b-', label='Original', linewidth=2)
    ax6.plot(range(1, len(cbam_cum_time)+1), cbam_cum_time, 'r-', label='CBAM', linewidth=2)
    ax6.set_title('Cumulative Training Time Comparison', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Cumulative Time (s)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{save_dir}/training_comparison_en.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: {plot_path}")
    plt.close()

def print_conclusion(orig_metrics, cbam_metrics):
    """Print summary and analysis"""
    print("\n" + "="*100)
    print("📈 Summary and Analysis".center(100))
    print("="*100)
    
    final_acc_improve = (cbam_metrics['final_acc_top1'] - orig_metrics['final_acc_top1']) / orig_metrics['final_acc_top1'] * 100
    best_acc_improve = (cbam_metrics['best_acc_top1'] - orig_metrics['best_acc_top1']) / orig_metrics['best_acc_top1'] * 100
    
    val_loss_change = (cbam_metrics['final_val_loss'] - orig_metrics['final_val_loss']) / orig_metrics['final_val_loss'] * 100
    
    time_change = (cbam_metrics['avg_epoch_time'] - orig_metrics['avg_epoch_time']) / orig_metrics['avg_epoch_time'] * 100
    
    print(f"\n1. Accuracy Aspect:")
    print(f"   - Final Top-1 Accuracy: {'Improved' if final_acc_improve > 0 else 'Decreased'} {abs(final_acc_improve):.2f}%")
    print(f"   - Best Top-1 Accuracy: {'Improved' if best_acc_improve > 0 else 'Decreased'} {abs(best_acc_improve):.2f}%")
    
    print(f"\n2. Loss Function Aspect:")
    print(f"   - Final Validation Loss: {'Reduced' if val_loss_change < 0 else 'Increased'} {abs(val_loss_change):.2f}%")
    
    print(f"\n3. Training Efficiency Aspect:")
    print(f"   - Average Epoch Time: {'Shortened' if time_change < 0 else 'Extended'} {abs(time_change):.2f}%")
    
    print(f"\n4. Overall Evaluation:")
    if final_acc_improve > 0 and val_loss_change < 0:
        print(f"   ✓ CBAM attention mechanism significantly improved model performance!")
    elif final_acc_improve > 0:
        print(f"   ✓ CBAM attention mechanism improved model accuracy, but training strategy needs optimization")
    elif val_loss_change < 0:
        print(f"   ✓ CBAM attention mechanism helps generalization, but accuracy improvement is not obvious")
    else:
        print(f"   ✗ CBAM attention mechanism did not bring significant performance improvement, adjust hyperparameters")

def main():
    print("="*100)
    print("🔬 YOLOv8 Original vs CBAM Model Complete Comparison Analysis".center(100))
    print("="*100)
    
    # Load data
    orig_epochs, orig_train_loss, orig_val_loss, orig_acc_top1, orig_acc_top5, orig_times = load_data(orig_model_path)
    cbam_epochs, cbam_train_loss, cbam_val_loss, cbam_acc_top1, cbam_acc_top5, cbam_times = load_data(cbam_model_path)
    
    # Extract metrics
    orig_metrics = extract_training_metrics(orig_epochs, orig_train_loss, orig_val_loss, orig_acc_top1, orig_acc_top5, orig_times)
    cbam_metrics = extract_training_metrics(cbam_epochs, cbam_train_loss, cbam_val_loss, cbam_acc_top1, cbam_acc_top5, cbam_times)
    
    # Load hyperparameters
    orig_hyperparams = load_hyperparams(orig_model_path)
    cbam_hyperparams = load_hyperparams(cbam_model_path)
    
    # Print comparisons
    compare_hyperparams(orig_hyperparams, cbam_hyperparams)
    print_training_metrics_comparison(orig_metrics, cbam_metrics)
    
    # Create plots
    save_dir = '/root/autodl-tmp/runs/classify'
    create_comparison_plots(orig_metrics, cbam_metrics, save_dir)
    
    # Print conclusion
    print_conclusion(orig_metrics, cbam_metrics)
    
    print("\n" + "="*100)
    print("✅ Analysis Complete!".center(100))
    print("="*100)

if __name__ == "__main__":
    main()
