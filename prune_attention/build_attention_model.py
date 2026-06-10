"""
构建注意力增强的 YOLOv8-cls 模型 — YAML 方式

流程:
  1. YOLO(yaml_path) → 从 YAML 构建完整架构（注意力模块由 ultralytics 内置注册）
  2. model.load("yolov8n-cls.pt") → 迁移预训练 backbone 权重
  3. model.train(...) → 训练，保存的 best.pt 包含完整 YAML 结构
  4. YOLO("best.pt") → 重新加载时从 YAML 重建架构，注意力不丢失
"""
import os
import torch
import torch.nn as nn
from ultralytics import YOLO


def get_yaml_path(att_type: str) -> str:
    """根据注意力类型返回对应 YAML 路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_map = {
        "se": "yolov8n-cls-se.yaml",
        "cbam": "yolov8n-cls-cbam.yaml",
        "eca": "yolov8n-cls-eca.yaml",
    }
    yaml_file = yaml_map.get(att_type.lower())
    if yaml_file is None:
        raise ValueError(f"未知的注意力类型: {att_type}，可选: {list(yaml_map.keys())}")
    return os.path.join(current_dir, yaml_file)


def build_attention_model(att_type: str, pretrained_weights: str = "yolov8n-cls.pt") -> YOLO:
    """
    从 YAML 构建注意力模型，并联迁移预训练权重。

    参数:
        att_type: "se" | "cbam" | "eca"
        pretrained_weights: 预训练权重路径

    返回:
        YOLO 模型，model.model 中包含完整的注意力架构
    """
    yaml_path = get_yaml_path(att_type)
    print(f"[INFO] 从 YAML 构建模型: {yaml_path}")

    # Step 1: 从 YAML 构建模型架构
    model = YOLO(yaml_path)

    # Step 2: 加载预训练权重（strict=False，注意力层权重随机初始化）
    model.load(pretrained_weights)
    print(f"[INFO] 预训练权重已加载: {pretrained_weights}")

    # Step 3: 验证注意力模块是否在模型中
    verify_attention_modules(model, att_type)

    return model


def verify_attention_modules(model: YOLO, att_type: str):
    """验证注意力模块确实存在且有权重参数"""
    net = model.model
    att_class_name = {
        "se": "SEBlock",
        "cbam": "CBAM",
        "eca": "ECABlock",
    }.get(att_type.lower(), att_type.upper())

    found = []
    for name, module in net.named_modules():
        if module.__class__.__name__ == att_class_name:
            params = sum(p.numel() for p in module.parameters())
            found.append((name, params))

    if found:
        print(f"[CHECK] 检测到 {len(found)} 个 {att_class_name} 模块:")
        for name, params in found:
            print(f"         {name}: {params:,} params")
        print(f"[CHECK] ✓ 注意力模块已成功注入模型结构")
    else:
        print(f"[WARNING] ✗ 未检测到 {att_class_name} 模块!")
        print("[WARNING] 请确认 ultralytics 的 nn/modules 中已注册该模块")
        print_all_module_types(net)


def print_all_module_types(net: nn.Module):
    """打印模型中所有模块的类型"""
    print("\n模型所有层:")
    for i, layer in enumerate(net.model if hasattr(net, 'model') else net):
        name = layer.__class__.__name__
        params = sum(p.numel() for p in layer.parameters())
        print(f"  [{i}] {name:<20} params={params:,}")


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("测试: 从 YAML 构建三种注意力模型")
    print("=" * 60)

    for att in ["se", "cbam", "eca"]:
        print(f"\n{'─'*50}")
        model = build_attention_model(att, "yolov8n-cls.pt")
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"  总参数量: {total_params:,}")
        print(f"{'─'*50}")