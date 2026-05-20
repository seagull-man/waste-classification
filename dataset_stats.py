import os
from pathlib import Path

def count_images(folder_path):
    """统计文件夹中的图片数量"""
    count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        count += len(list(Path(folder_path).glob(ext)))
    return count

print("=" * 60)
print("垃圾分类数据集统计报告")
print("=" * 60)

# 原始40分类数据集
print("\n【原始数据集 (40分类)】")
print("来源: https://ai.gitcode.com/ai53_19/garbage_datasets")
print("-" * 60)

train_path = Path("garbage_datasets/images/train")
val_path = Path("garbage_datasets/images/val")

train_count = count_images(train_path)
val_count = count_images(val_path)
total = train_count + val_count

print(f"训练集图片数: {train_count}")
print(f"验证集图片数: {val_count}")
print(f"总图片数: {total}")
print(f"训练集比例: {train_count/total*100:.2f}%")
print(f"验证集比例: {val_count/total*100:.2f}%")

# 四分类数据集
print("\n【四分类数据集】")
print("-" * 60)

for split in ['train', 'val', 'test']:
    print(f"\n{split.upper()}集:")
    total_split = 0
    for cls in ['recyclable', 'hazardous', 'kitchen', 'other']:
        cls_path = Path(f"garbage_4cls/{split}/{cls}")
        if cls_path.exists():
            cnt = count_images(cls_path)
            print(f"  {cls:15s}: {cnt:5d}")
            total_split += cnt
    print(f"  {'总计':15s}: {total_split:5d}")

print("\n" + "=" * 60)
