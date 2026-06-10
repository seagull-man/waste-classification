#!/usr/bin/env python3
import os
import sys

def count_images(directory):
    """统计目录下的图片数量"""
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                count += 1
    return count

def main():
    data_dir = "/root/autodl-tmp/garbage_4cls"
    
    if not os.path.exists(data_dir):
        print(f"数据集目录不存在: {data_dir}")
        return
    
    print("=" * 70)
    print("数据集统计信息")
    print("=" * 70)
    
    splits = ['train', 'val', 'test']
    total_count = 0
    split_counts = {}
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            count = count_images(split_dir)
            split_counts[split] = count
            total_count += count
            print(f"{split}: {count} 张")
    
    print(f"\n总计: {total_count} 张")
    print("=" * 70)
    
    # 统计每个类别的数量
    print("\n类别分布:")
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            print(f"\n{split} 集:")
            class_dirs = [d for d in os.listdir(split_dir) 
                          if os.path.isdir(os.path.join(split_dir, d))]
            for cls_name in sorted(class_dirs):
                cls_dir = os.path.join(split_dir, cls_name)
                cls_count = count_images(cls_dir)
                print(f"  {cls_name}: {cls_count} 张")

if __name__ == "__main__":
    main()
