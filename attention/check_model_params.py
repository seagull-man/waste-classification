#!/usr/bin/env python3
"""
检查模型参数量的简易脚本
可以检查单个模型或多个模型的参数量、可训练参数和模型大小
"""

import torch
from pathlib import Path
from ultralytics import YOLO


def check_model(model_path):
    """检查单个模型"""
    print(f"\n{'='*70}")
    print(f"检查模型: {model_path.name}")
    print(f"{'='*70}")
    
    try:
        model = YOLO(str(model_path))
        torch_model = model.model
        
        # 计算参数量
        total_params = sum(p.numel() for p in torch_model.parameters())
        trainable_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
        
        # 计算模型文件大小
        file_size_mb = model_path.stat().st_size / 1024 / 1024
        
        print(f"\n📊 参数统计:")
        print(f"  总参数量: {total_params:,} ({total_params/1e6:.3f} M)")
        print(f"  可训练参数: {trainable_params:,} ({trainable_params/1e6:.3f} M)")
        print(f"  模型大小: {file_size_mb:.2f} MB")
        
        # 统计各层参数
        print(f"\n🔍 各层参数:")
        layer_info = []
        for name, module in torch_model.named_modules():
            layer_params = sum(p.numel() for p in module.parameters())
            if layer_params > 0:  # 只显示有参数的层
                layer_info.append((name, layer_params, module.__class__.__name__))
        
        # 按参数数量排序
        layer_info.sort(key=lambda x: x[1], reverse=True)
        
        for name, params, module_type in layer_info[:10]:  # 只显示前10层
            print(f"  {name:30} ({module_type:20}) - {params:12,} params")
        
        if len(layer_info) > 10:
            print(f"  ... 还有 {len(layer_info) - 10} 层")
        
        return {
            "name": model_path.name,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "file_size_mb": file_size_mb
        }
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return None


def main():
    print("="*70)
    print("模型参数量检查工具")
    print("="*70)
    
    # 常见的模型路径
    common_paths = [
        # YOLOv8 原始模型
        Path("//root/autodl-tmp/waste-classification/attention/runs/classify/waste_cls_original/weights/best.pt"),
        
        
        # CBAM 模型
        Path("/root/autodl-tmp/waste-classification/attention/runs/classify/new_cbam_fixed/weights/best.pt"),
        
        
        # SE 模型
        Path("waste-classification/attention/runs/classify/waste_cls_yolov8n_se/weights/best.pt"),
       
        
        # ECA 模型
        Path("waste-classification/attention/runs/classify/waste_cls_yolov8n_eca/weights/best.pt"),
    ]
    
    existing_models = []
    for path in common_paths:
        if path.exists():
            existing_models.append(path)
    
    if len(existing_models) == 0:
        print("\n⚠️ 没有找到任何常见模型！")
        print("\n请输入你想检查的模型路径:")
        user_path = input("模型路径: ").strip()
        if user_path:
            user_path = Path(user_path)
            if user_path.exists():
                existing_models.append(user_path)
    
    if len(existing_models) == 0:
        print("\n❌ 没有找到任何模型！")
        return
    
    print(f"\n找到 {len(existing_models)} 个模型")
    
    all_results = []
    for model_path in existing_models:
        result = check_model(model_path)
        if result:
            all_results.append(result)
    
    if len(all_results) > 1:
        # 打印汇总对比
        print(f"\n{'='*100}")
        print(f"{'模型对比汇总':^100}")
        print(f"{'='*100}")
        print(f"{'模型名称':<40} {'总参数量(M)':<15} {'可训练参数(M)':<15} {'文件大小(MB)':<15}")
        print("-"*100)
        for result in all_results:
            print(f"{result['name']:<40} "
                  f"{result['total_params']/1e6:<15.3f} "
                  f"{result['trainable_params']/1e6:<15.3f} "
                  f"{result['file_size_mb']:<15.2f}")
        print("="*100)
    
    print(f"\n✅ 检查完成！")


if __name__ == "__main__":
    main()
