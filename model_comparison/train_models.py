"""
模型对比训练脚本
训练多个模型进行对比：YOLOv8、ResNet50、EfficientNet
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, efficientnet_b4
from ultralytics import YOLO
import time
import json

# 配置
DATA_PATH = 'c:/Users/11237/Desktop/final2/autodl-tmp/garbage_4cls'
OUTPUT_DIR = 'c:/Users/11237/Desktop/final2/autodl-tmp/model_comparison/results'
BATCH_SIZE = 32
EPOCHS = 50
IMG_SIZE = 320
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

from PIL import Image
from torchvision.datasets import VisionDataset

class SafeImageFolder(VisionDataset):
    """自定义数据集类，跳过损坏的图像文件"""
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.classes, self.class_to_idx = self.find_classes(root)
        self.samples = self.make_dataset(root, self.class_to_idx)
    
    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def make_dataset(self, directory, class_to_idx):
        instances = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            for entry in os.scandir(target_dir):
                if entry.is_file():
                    path = entry.path
                    try:
                        # 尝试打开图像文件
                        with Image.open(path) as img:
                            img.verify()
                        instances.append((path, class_index))
                    except (IOError, SyntaxError):
                        print(f"⚠️ 跳过损坏的图像: {path}")
        return instances
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    def __len__(self):
        return len(self.samples)

def train_torch_model(model_name, model, train_loader, val_loader, criterion, optimizer, train_dataset_len):
    """训练PyTorch模型（ResNet50、EfficientNet）"""
    results = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        total_samples = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
        
        train_loss /= total_samples if total_samples > 0 else 1
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        
        epoch_time = time.time() - start_time
        
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        results['epoch_time'].append(epoch_time)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'{model_name}_best.pth'))
        
        print(f"[{model_name}] Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.2f}s")
    
    return results, best_acc

def train_yolov8():
    """训练YOLOv8分类模型"""
    model = YOLO('yolov8n-cls.pt')
    
    results = model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        name='yolov8n_comparison',
        save=True,
        plots=True
    )
    
    # 保存结果
    results_path = '/root/autodl-tmp/runs/classify/yolov8n_comparison/results.csv'
    return results_path

def main():
    print("="*100)
    print("🔍 模型对比实验")
    print("="*100)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = SafeImageFolder(os.path.join(DATA_PATH, 'train'), transform=transform)
    val_dataset = SafeImageFolder(os.path.join(DATA_PATH, 'val'), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # 关闭多线程避免问题
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    print(f"\n📊 数据集统计:")
    print(f"   训练集: {len(train_dataset)} 张图片")
    print(f"   验证集: {len(val_dataset)} 张图片")
    print(f"   类别数: {num_classes}")
    
    # 存储所有模型结果
    all_results = {}
    
    # 1. 训练ResNet50
    print("\n" + "="*100)
    print("🚀 训练 ResNet50")
    print("="*100)
    resnet_model = resnet50(pretrained=True)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
    resnet_model = resnet_model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(resnet_model.parameters(), lr=0.001, weight_decay=0.0005)
    
    resnet_results, resnet_best_acc = train_torch_model('resnet50', resnet_model, train_loader, val_loader, criterion, optimizer, len(train_dataset))
    all_results['ResNet50'] = {
        'results': resnet_results,
        'best_acc': resnet_best_acc,
        'params': sum(p.numel() for p in resnet_model.parameters()) / 1e6
    }
    
    # 2. 训练EfficientNet-B4
    print("\n" + "="*100)
    print("🚀 训练 EfficientNet-B4")
    print("="*100)
    efficient_model = efficientnet_b4(pretrained=True)
    efficient_model.classifier[1] = nn.Linear(efficient_model.classifier[1].in_features, num_classes)
    efficient_model = efficient_model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(efficient_model.parameters(), lr=0.001, weight_decay=0.0005)
    
    efficient_results, efficient_best_acc = train_torch_model('efficientnet_b4', efficient_model, train_loader, val_loader, criterion, optimizer, len(train_dataset))
    all_results['EfficientNet-B4'] = {
        'results': efficient_results,
        'best_acc': efficient_best_acc,
        'params': sum(p.numel() for p in efficient_model.parameters()) / 1e6
    }
    
    # 3. 训练YOLOv8
    print("\n" + "="*100)
    print("🚀 训练 YOLOv8n-cls")
    print("="*100)
    yolov8_results_path = train_yolov8()
    
    # 保存所有结果
    with open(os.path.join(OUTPUT_DIR, 'comparison_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("\n" + "="*100)
    print("✅ 所有模型训练完成！")
    print("="*100)
    print(f"结果已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
