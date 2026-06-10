"""
模型结构化剪枝 + 微调脚本

使用 torch-pruning 对 backbone 中的 Conv2d 层做 L1-norm 通道剪枝，
尝试多个剪枝比例，找到精度基本不降的最高比例。

剪枝策略:
  - 只剪 backbone 中的 Conv2d 层（排除注意力模块和分类头）
  - 保留注意力模块完整，因为其参数量极小
  - L1-norm 排序，按比例裁掉最不重要的通道
  - 每个剪枝比例后微调 FINE_TUNE_EPOCHS 轮恢复精度
  - best.pt 已包含 YAML 架构，YOLO(best.pt) 加载后注意力不丢失
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_PATH, RUNS_DIR, DEVICE, DEVICE_ID,
    BATCH_SIZE, IMGSZ, LR0, WEIGHT_DECAY,
    MOMENTUM, PATIENCE,
    PRUNE_RATIOS, FINE_TUNE_EPOCHS,
)

try:
    import torch_pruning as tp
    HAS_TP = True
except ImportError:
    HAS_TP = False
    print("[WARNING] torch_pruning 未安装")
    print("[TIP] pip install torch-pruning")


def get_model_info(model):
    """获取模型基本信息"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 检查注意力模块
    att_layers = []
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if cls in ("SEBlock", "CBAM", "ECABlock"):
            att_layers.append((name, cls, sum(p.numel() for p in module.parameters())))

    return {"total_params": total, "trainable_params": trainable, "attention_layers": att_layers}


def prune_model(model_path: str, prune_ratio: float, output_dir: str):
    """
    使用 torch-pruning 对模型进行结构化剪枝

    参数:
        model_path: 训练好的模型路径（best.pt，含 YAML 架构）
        prune_ratio: 剪枝比例 (0.0 ~ 1.0)
        output_dir: 输出目录
    """
    ratio_pct = int(prune_ratio * 100)
    print(f"\n{'='*60}")
    print(f"  剪枝: ratio={ratio_pct}%")
    print(f"  源模型: {model_path}")
    print(f"{'='*60}")

    # 加载模型（best.pt 含 YAML 架构，注意力模块不会丢失）
    yolo = YOLO(model_path)
    model = yolo.model
    model.eval()
    model.to(DEVICE)

    # 剪枝前信息
    info_before = get_model_info(model)
    print(f"\n[INFO] 剪枝前参数量: {info_before['total_params']:,}")
    if info_before['attention_layers']:
        print(f"[INFO] 注意力模块:")
        for name, cls, params in info_before['attention_layers']:
            print(f"         {name} ({cls}): {params:,} params")

    # 构建 torch-pruning 依赖图
    example_inputs = torch.randn(1, 3, IMGSZ, IMGSZ).to(DEVICE)
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=example_inputs)

    # 获取 backbone 中可剪枝的 Conv2d（排除注意力模块内部的和太小/太大的）
    pruneable_convs = []
    excluded_keywords = ["attention", "se", "cbam", "eca", "head", "classify", "linear"]

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        # 排除注意力模块内部
        if any(kw in name.lower() for kw in excluded_keywords):
            continue
        # 排除太小的层
        if module.out_channels <= 8:
            continue
        pruneable_convs.append((name, module))

    print(f"\n[INFO] 可剪枝 Conv2d 层: {len(pruneable_convs)}")
    for name, module in pruneable_convs[:5]:  # 只打印前 5 个
        print(f"  - {name}: out_channels={module.out_channels}")
    if len(pruneable_convs) > 5:
        print(f"  ... (共 {len(pruneable_convs)} 层)")

    # 逐层剪枝
    pruned_count = 0
    total_pruned_channels = 0

    for name, module in pruneable_convs:
        num_channels = module.out_channels
        num_prune = max(1, int(num_channels * prune_ratio))

        # 至少保留 4 个通道
        if num_prune >= num_channels - 4:
            num_prune = max(0, num_channels - 4)

        if num_prune < 1:
            continue

        # L1 范数重要性排序
        importance = torch.sum(torch.abs(module.weight), dim=(1, 2, 3))
        _, prune_indices = torch.topk(importance, num_prune, largest=False)

        try:
            plan = DG.get_pruning_plan(
                module,
                tp.prune_conv_out_channels,
                idxs=prune_indices.tolist()
            )
            plan.exec()
            pruned_count += 1
            total_pruned_channels += num_prune
        except Exception as e:
            print(f"  [SKIP] {name}: 无法剪枝 ({e})")
            continue

    print(f"\n[INFO] 已剪枝 {pruned_count} 层，共移除 {total_pruned_channels} 个通道")

    # 剪枝后信息
    info_after = get_model_info(model)
    reduction = (1 - info_after['total_params'] / info_before['total_params']) * 100
    print(f"[INFO] 剪枝后参数量: {info_after['total_params']:,} (减少 {reduction:.1f}%)")

    # 更新 yolo 对象
    yolo.model = model

    # 微调
    print(f"\n[INFO] 开始微调 ({FINE_TUNE_EPOCHS} epochs, lr={LR0 * 0.1:.6f})...")
    ft_dir = os.path.join(output_dir, f"finetune_{ratio_pct}")
    results = yolo.train(
        data=DATA_PATH,
        epochs=FINE_TUNE_EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE // 2,
        lr0=LR0 * 0.1,
        weight_decay=WEIGHT_DECAY,
        momentum=MOMENTUM,
        warmup_epochs=0,
        patience=PATIENCE,
        device=DEVICE_ID,
        project=ft_dir,
        name="ft",
        exist_ok=True,
        pretrained=False,
    )

    best_ft = os.path.join(ft_dir, "ft", "weights", "best.pt")
    print(f"\n[SUCCESS] 剪枝 {ratio_pct}% 微调完成: {best_ft}")

    return yolo, best_ft


def main():
    parser = argparse.ArgumentParser(description="YOLOv8n 模型剪枝 + 微调")
    parser.add_argument(
        "--model", type=str, default="yolov8n_se",
        help="模型名称 (在 runs/ 下的目录名)"
    )
    parser.add_argument(
        "--ratios", type=str, default=None,
        help="剪枝比例，逗号分隔，例如: 0.1,0.2,0.3。默认用 config 中的 PRUNE_RATIOS"
    )
    args = parser.parse_args()

    model_name = args.model
    model_path = os.path.join(RUNS_DIR, model_name, "train", "weights", "best.pt")

    if not os.path.exists(model_path):
        print(f"[ERROR] 模型不存在: {model_path}")
        print(f"[TIP] 可用的模型:")
        for d in sorted(os.listdir(RUNS_DIR)):
            pt = os.path.join(RUNS_DIR, d, "train", "weights", "best.pt")
            if os.path.exists(pt):
                print(f"  - {d}")
        return

    ratios = PRUNE_RATIOS
    if args.ratios:
        ratios = [float(r.strip()) for r in args.ratios.split(",")]

    prune_output_dir = os.path.join(RUNS_DIR, f"{model_name}_pruned")
    os.makedirs(prune_output_dir, exist_ok=True)

    print(f"\n[INFO] 源模型: {model_name}")
    print(f"[INFO] 模型路径: {model_path}")
    print(f"[INFO] 剪枝比例: {[f'{r*100:.0f}%' for r in ratios]}")

    if not HAS_TP:
        print("\n[ERROR] torch-pruning 未安装!")
        print("[FIX] pip install torch-pruning")
        return

    for ratio in ratios:
        try:
            prune_model(model_path, ratio, prune_output_dir)
        except Exception as e:
            print(f"[ERROR] ratio={ratio} 剪枝失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[SUMMARY] 所有剪枝实验完成!")
    print(f"[SUMMARY] 运行 evaluate_all.py 查看剪枝效果对比")


if __name__ == "__main__":
    main()