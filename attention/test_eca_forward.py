import torch
from ultralytics import YOLO

yaml_path = "/root/autodl-tmp/waste-classification/attention/yolov8n-cls-eca.yaml"

model = YOLO(yaml_path)
model.load("yolov8n-cls.pt")

x = torch.randn(1, 3, 320, 320)

with torch.no_grad():
    y = model.model(x)

eca_keys = [
    k for k in model.model.state_dict().keys()
    if "conv.weight" in k and ("model.7" in k or "model.10" in k)
]

print("=" * 80)
print("前向传播成功")
print("输出类型:", type(y))
print("ECA参数数量:", len(eca_keys))

for k in eca_keys:
    print(k, model.model.state_dict()[k].shape)

print("=" * 80)

if len(eca_keys) > 0:
    print("✅ ECABlock 已经真正进入模型结构")
else:
    print("❌ 没检测到 ECABlock 参数")
