"""
演示如何解决 KeyError: 'ECABlock' 问题
这是完整且可直接运行的示例
"""
import torch
import torch.nn as nn
from ultralytics import YOLO


# ================== 关键修复步骤 1: 定义 ECABlock ==================
class ECABlock(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._initialized = False
        self._gamma = 2
        self._b = 1

    def _initialize(self, channels):
        t = int(abs((torch.log2(torch.tensor(channels)) + self._b) / self._gamma))
        kernel_size = t if t % 2 == 1 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._initialized:
            self._initialize(x.size(1))
        
        b, c, _, _ = x.shape
        y = self.avg_pool(x).squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


# ================== 关键修复步骤 2: 注入到 ultralytics.nn.tasks ==================
import ultralytics.nn.tasks as tasks_module
# 直接修改模块的 __dict__，这样 parse_model 中的 globals() 就能找到了
tasks_module.__dict__['ECABlock'] = ECABlock
print("✅ ECABlock registered!")


# ================== 现在可以正常加载模型了 ==================
yaml_path = "/root/autodl-tmp/waste-classification/attention/yolov8n-cls-eca.yaml"
print(f"\n📦 Loading model from {yaml_path}...")

try:
    model = YOLO(yaml_path)
    print("✅ Model loaded successfully!")
    print("🎉 问题已解决！您现在可以正常运行训练脚本了！")
except Exception as e:
    print(f"❌ Error: {e}")
