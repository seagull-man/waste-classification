import torch
from ultralytics import YOLO

yaml_path = "/root/autodl-tmp/waste-classification/attention/yolov8n-cls-se.yaml"

model = YOLO(yaml_path)
model.load("yolov8n-cls.pt")

x = torch.randn(1, 3, 320, 320)

with torch.no_grad():
    y = model.model(x)

se_keys = [
    k for k in model.model.state_dict().keys()
    if "fc.0.weight" in k or "fc.2.weight" in k
]

print("=" * 80)
print("前向传播成功")
print("输出类型:", type(y))
print("SE参数数量:", len(se_keys))

for k in se_keys:
    print(k)

print("=" * 80)

if len(se_keys) > 0:
    print("✅ SEBlock 已经真正进入模型结构")
else:
    print("❌ 没检测到 SEBlock 参数")
