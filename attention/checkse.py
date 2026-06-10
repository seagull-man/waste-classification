import torch
import os
from pathlib import Path

# 尝试多个可能的路径
possible_paths = [
    "runs/classify/waste_cls_yolov8n_se/weights/best.pt",
    "/root/autodl-tmp/runs/classify/waste_cls_yolov8n_se/weights/best.pt",
    "/root/autodl-tmp/waste-classification/attention/runs/classify/waste_cls_yolov8n_se/weights/best.pt",
]

ckpt_path = None
for path in possible_paths:
    if Path(path).exists():
        ckpt_path = path
        print(f"✅ 找到模型: {path}")
        break

if ckpt_path is None:
    print("❌ 错误：找不到 SE 模型文件！")
    print("\n你需要先运行训练：")
    print("  cd /root/autodl-tmp/waste-classification/attention")
    print("  python waste_cls_yolov8n_se.py")
    exit(1)

print(f"\n加载模型: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")

print("\n检查点键值:")
print(ckpt.keys())

if "model" not in ckpt:
    print("\n❌ 错误：检查点中没有 'model' 键！")
    exit(1)

model = ckpt["model"]

se_keys = []

for name, param in model.state_dict().items():
    if "se" in name.lower():
        se_keys.append(name)

print("\n" + "=" * 80)
print(f"SE 参数数量: {len(se_keys)}")
print("=" * 80)

if len(se_keys) > 0:
    print("\n✅ best.pt 中确实保存了 SE-Block 参数！")
    print("\nSE 相关参数列表:")
    for key in se_keys[:20]:  # 只显示前20个
        print(f"  - {key}")
    if len(se_keys) > 20:
        print(f"  ... 还有 {len(se_keys) - 20} 个")
else:
    print("\n❌ best.pt 中没有 SE-Block 参数，说明 SE 没有真正保存")