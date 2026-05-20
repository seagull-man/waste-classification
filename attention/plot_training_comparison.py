import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 原始模型数据
orig_file = '/root/autodl-tmp/runs/classify/waste_cls_original/results.csv'
# CBAM模型数据
cbam_file = '/root/autodl-tmp/runs/classify/waste_cls_cbam-4/results.csv'

def read_csv(file_path):
    epochs = []
    train_loss = []
    val_loss = []
    acc_top1 = []
    times = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train/loss']))
            val_loss.append(float(row['val/loss']))
            acc_top1.append(float(row['metrics/accuracy_top1']))
            times.append(float(row['time']))
    return epochs, train_loss, val_loss, acc_top1, times

# 读取数据
orig_epochs, orig_train_loss, orig_val_loss, orig_acc, orig_times = read_csv(orig_file)
cbam_epochs, cbam_train_loss, cbam_val_loss, cbam_acc, cbam_times = read_csv(cbam_file)

# 生成图表
fig = plt.figure(figsize=(16, 10))
fig.suptitle('YOLOv8 Original vs CBAM Model Training Comparison', fontsize=16, fontweight='bold')

# 1. 准确率对比
ax1 = plt.subplot(2, 2, 1)
ax1.plot(orig_epochs, orig_acc, 'b-', linewidth=2, label='Original Model')
ax1.plot(cbam_epochs, cbam_acc, 'r-', linewidth=2, label='CBAM Model')
ax1.set_title('Top-1 Accuracy Comparison', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.5, 1.0])

# 2. 训练损失对比
ax2 = plt.subplot(2, 2, 2)
ax2.plot(orig_epochs, orig_train_loss, 'b-', linewidth=2, label='Original Train Loss')
ax2.plot(cbam_epochs, cbam_train_loss, 'r-', linewidth=2, label='CBAM Train Loss')
ax2.set_title('Training Loss Comparison', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 验证损失对比
ax3 = plt.subplot(2, 2, 3)
ax3.plot(orig_epochs, orig_val_loss, 'b-', linewidth=2, label='Original Val Loss')
ax3.plot(cbam_epochs, cbam_val_loss, 'r-', linewidth=2, label='CBAM Val Loss')
ax3.set_title('Validation Loss Comparison', fontsize=12, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 关键指标对比柱状图
ax4 = plt.subplot(2, 2, 4)
bar_width = 0.35
x = [0, 1, 2]
labels = ['Final Acc', 'Best Acc', 'Avg Time (s)']

orig_metrics = [orig_acc[-1], max(orig_acc), sum(orig_times) / len(orig_times)]
cbam_metrics = [cbam_acc[-1], max(cbam_acc), sum(cbam_times) / len(cbam_times)]

ax4.bar([i - bar_width/2 for i in x], orig_metrics, bar_width, label='Original', color='blue', alpha=0.7)
ax4.bar([i + bar_width/2 for i in x], cbam_metrics, bar_width, label='CBAM', color='red', alpha=0.7)
ax4.set_xticks(x)
ax4.set_xticklabels(labels)
ax4.set_title('Key Metrics Comparison', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_file = '/root/autodl-tmp/runs/classify/training_comparison_chart.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ 图表已保存到: {output_file}")
plt.close()

# 打印性能指标
print("\n" + "="*80)
print("📊 训练参数与性能对比报告")
print("="*80)
print(f"\n{'指标':<25} {'原始模型 (Original)':<25} {'CBAM模型 (cbam-2)':<25} {'差异':<15}")
print("-"*80)

# 总训练轮数
print(f"{'训练轮数':<25} {len(orig_epochs):<25} {len(cbam_epochs):<25} {len(cbam_epochs) - len(orig_epochs):<+15}")

# 最终准确率
print(f"{'最终Top-1准确率':<25} {orig_acc[-1]:<25.4f} {cbam_acc[-1]:<25.4f} {cbam_acc[-1] - orig_acc[-1]:<+15.4f}")

# 最佳准确率
print(f"{'最佳Top-1准确率':<25} {max(orig_acc):<25.4f} {max(cbam_acc):<25.4f} {max(cbam_acc) - max(orig_acc):<+15.4f}")

# 最佳准确率轮数
print(f"{'最佳准确率epoch':<25} {orig_acc.index(max(orig_acc)) + 1:<25} {cbam_acc.index(max(cbam_acc)) + 1:<25} {'':<15}")

# 最终训练损失
print(f"{'最终训练损失':<25} {orig_train_loss[-1]:<25.5f} {cbam_train_loss[-1]:<25.5f} {cbam_train_loss[-1] - orig_train_loss[-1]:<+15.5f}")

# 最终验证损失
print(f"{'最终验证损失':<25} {orig_val_loss[-1]:<25.5f} {cbam_val_loss[-1]:<25.5f} {cbam_val_loss[-1] - orig_val_loss[-1]:<+15.5f}")

# 最佳验证损失
print(f"{'最佳验证损失':<25} {min(orig_val_loss):<25.5f} {min(cbam_val_loss):<25.5f} {min(cbam_val_loss) - min(orig_val_loss):<+15.5f}")

# 总训练时间
orig_total_time = orig_times[-1]
cbam_total_time = cbam_times[-1]
print(f"{'总训练时间 (s)':<25} {orig_total_time:<25.1f} {cbam_total_time:<25.1f} {cbam_total_time - orig_total_time:<+15.1f}")

# 平均每轮时间
orig_avg_time = sum(orig_times) / len(orig_times)
cbam_avg_time = sum(cbam_times) / len(cbam_times)
print(f"{'平均每轮时间 (s)':<25} {orig_avg_time:<25.2f} {cbam_avg_time:<25.2f} {cbam_avg_time - orig_avg_time:<+15.2f}")

print("\n" + "="*80)
print("📈 性能分析")
print("="*80)

acc_diff = cbam_acc[-1] - orig_acc[-1]
best_acc_diff = max(cbam_acc) - max(orig_acc)
val_loss_diff = cbam_val_loss[-1] - orig_val_loss[-1]
speed_diff = cbam_avg_time - orig_avg_time

if acc_diff > 0:
    print(f"✅ 准确率提升: {acc_diff:.4f} (+{acc_diff/orig_acc[-1]*100:.2f}%)")
else:
    print(f"❌ 准确率下降: {abs(acc_diff):.4f}")

if best_acc_diff > 0:
    print(f"✅ 最佳准确率提升: {best_acc_diff:.4f}")
else:
    print(f"❌ 最佳准确率下降: {abs(best_acc_diff):.4f}")

if val_loss_diff < 0:
    print(f"✅ 验证损失降低: {abs(val_loss_diff):.5f}")
else:
    print(f"❌ 验证损失增加: {val_loss_diff:.5f}")

if speed_diff < 0:
    print(f"✅ 训练速度提升: {abs(speed_diff):.2f} 秒/轮")
else:
    print(f"❌ 训练速度减慢: {speed_diff:.2f} 秒/轮")

print("\n" + "="*80)
print("💡 结论")
print("="*80)

if acc_diff > 0 and val_loss_diff < 0:
    print("CBAM注意力机制成功提升了模型性能！")
elif acc_diff > 0:
    print("CBAM模型在准确率上有提升，但验证损失略有增加。")
else:
    print("需要进一步优化CBAM的训练参数。")
