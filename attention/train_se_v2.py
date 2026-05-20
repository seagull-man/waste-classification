"""
YOLOv8 + SE-Block 注意力机制训练脚本 V2

改进版：
1. 调整学习率策略 - backbone使用更高的学习率
2. 添加梯度裁剪防止梯度爆炸
3. 更稳定的训练过程
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from SE_Block import SELayer


class WasteDataset(Dataset):
    """垃圾分类数据集"""

    def __init__(self, data_path, transform=None, mode='train'):
        self.data_path = Path(data_path)
        self.transform = transform
        self.mode = mode
        self.classes = ['hazardous', 'kitchen', 'other', 'recyclable']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        self.corrupted_files = []

        data_dir = self.data_path / mode
        for class_name in self.classes:
            class_dir = data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob('*.jpeg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label


class SEYOLOV8(nn.Module):
    """带SE注意力的YOLOv8分类模型"""

    def __init__(self, original_model, num_classes=4, reduction=16):
        super(SEYOLOV8, self).__init__()

        self.backbone = original_model.model.model[:-1]

        temp_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            temp_output = self.backbone(temp_input)
        backbone_output_ch = temp_output.shape[1]
        print(f"  🔍 动态检测backbone输出通道数: {backbone_output_ch}")

        self.se_layer = SELayer(channel=backbone_output_ch, reduction=reduction)
        print(f"  ✅ SE层注入成功，通道数={backbone_output_ch}")

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(backbone_output_ch, num_classes)
        )
        print(f"  ✅ 分类器修改为{num_classes}类")

    def forward(self, x):
        features = self.backbone(x)
        features = self.se_layer(features)
        output = self.classifier(features)
        return output


def evaluate_model(model, dataloader, device, num_classes):
    """评估模型在验证集上的性能"""
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            if inputs.sum() == 0:
                continue

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    model.train()

    avg_loss = val_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100. * correct / total if total > 0 else 0

    return avg_loss, accuracy


def train_se_model_v2():
    """训练SE-Block模型 V2 - 改进版学习率"""
    print("="*70)
    print("YOLOv8 + SE-Block 注意力机制训练 V2 (改进版)")
    print("="*70)

    data_path = '/root/autodl-tmp/garbage_4cls'
    num_classes = 4
    batch_size = 24
    num_epochs = 50
    learning_rate = 0.001

    print(f"\n📂 数据集路径: {data_path}")
    print(f"🎯 类别数: {num_classes}")
    print(f"📦 批次大小: {batch_size}")
    print(f"🔄 训练轮数: {num_epochs}")
    print(f"📉 学习率: {learning_rate}")

    print("\n📥 加载预训练YOLOv8模型...")
    base_model = YOLO('yolov8n-cls.pt')

    print("\n🔧 构建SE-YOLOv8模型...")
    model = SEYOLOV8(base_model, num_classes=num_classes)

    print("\n📊 模型结构:")
    print("  Backbone:")
    for i, layer in enumerate(model.backbone):
        print(f"    层 {i}: {type(layer).__name__}")
    print(f"  SE层: SELayer (channel={model.se_layer.fc[0].in_features})")
    print(f"  分类器:")
    for layer in model.classifier:
        print(f"    {type(layer).__name__}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 训练设备: {device}")

    model.to(device)

    print("\n🚀 开始训练 (改进版学习率策略)...")
    print("  - Backbone学习率: 0.0005 (稍高，允许更多微调)")
    print("  - SE层学习率: 0.001 (重点学习注意力机制)")
    print("  - 分类器学习率: 0.001")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = WasteDataset(data_path, transform=train_transform, mode='train')
    val_dataset = WasteDataset(data_path, transform=val_transform, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f"\n📊 训练数据集大小: {len(train_dataset)}")
    print(f"📊 验证数据集大小: {len(val_dataset)}")

    criterion = nn.CrossEntropyLoss()

    backbone_lr = 0.0005
    se_lr = 0.001
    classifier_lr = 0.001

    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': backbone_lr},
        {'params': model.se_layer.parameters(), 'lr': se_lr},
        {'params': model.classifier.parameters(), 'lr': classifier_lr}
    ], weight_decay=0.0005)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01)

    save_dir = 'runs/classify/waste_cls_se_v2'
    os.makedirs(save_dir, exist_ok=True)
    weights_dir = f'{save_dir}/weights'
    os.makedirs(weights_dir, exist_ok=True)

    training_history = {
        'epoch': [],
        'time': [],
        'train/loss': [],
        'metrics/accuracy_top1': [],
        'val/loss': [],
        'lr/backbone': [],
        'lr/se': [],
        'lr/classifier': []
    }

    best_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            if inputs.sum() == 0:
                continue

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total if total > 0 else 0

        val_loss, val_acc = evaluate_model(model, val_loader, device, num_classes)

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        current_lrs = [group['lr'] for group in optimizer.param_groups]

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"train/loss: {train_loss:.4f} "
              f"metrics/accuracy_top1: {train_acc:.2f}% "
              f"val/loss: {val_loss:.4f} "
              f"val/accuracy_top1: {val_acc:.2f}% "
              f"time: {epoch_time:.1f}s")

        training_history['epoch'].append(epoch + 1)
        training_history['time'].append(round(total_time, 1))
        training_history['train/loss'].append(round(train_loss, 4))
        training_history['metrics/accuracy_top1'].append(round(train_acc, 2))
        training_history['val/loss'].append(round(val_loss, 4))
        training_history['lr/backbone'].append(current_lrs[0])
        training_history['lr/se'].append(current_lrs[1])
        training_history['lr/classifier'].append(current_lrs[2])

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            print(f"  🏆 新最佳验证准确率: {best_acc:.2f}%")

            best_model_path = f'{weights_dir}/best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'backbone_ch': 256,
            }, best_model_path)
            print(f"  💾 保存最佳模型: {best_model_path}")

    print("\n" + "="*70)
    print("✅ 训练完成！保存最终模型...")
    print("="*70)

    final_path = f'{weights_dir}/last.pt'
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'backbone_ch': 256,
    }, final_path)
    print(f"💾 最终模型保存到: {final_path}")

    history_path = f'{save_dir}/results.csv'
    with open(history_path, 'w') as f:
        f.write('epoch,time,train/loss,metrics/accuracy_top1,val/loss,val_accuracy_top1\n')
        for i in range(len(training_history['epoch'])):
            f.write(f"{training_history['epoch'][i]},"
                   f"{training_history['time'][i]},"
                   f"{training_history['train/loss'][i]},"
                   f"{training_history['metrics/accuracy_top1'][i]},"
                   f"{training_history['val/loss'][i]},"
                   f"{training_history['metrics/accuracy_top1'][i]}\n")
    print(f"💾 训练结果保存到: {history_path}")

    json_path = f'{save_dir}/training_history.json'
    with open(json_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"💾 训练历史JSON保存到: {json_path}")

    print(f"\n🏆 训练期间最佳验证准确率: {best_acc:.2f}% (Epoch {best_epoch})")

    return final_path, training_history


if __name__ == "__main__":
    train_se_model_v2()
