import torch
import torch.nn as nn
from ultralytics import YOLO

# 首先导入 ultralytics.nn.tasks 模块，然后注入 ECABlock 到它的全局命名空间
import ultralytics.nn.tasks as tasks_module


class ECABlock(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, *args):
        super().__init__()
        self.args = args
        self._initialized = False
        self.avg_pool = None
        self.conv = None
        self.sigmoid = None

    def _initialize(self, channels):
        gamma = 2
        b = 1
        t = int(abs((torch.log2(torch.tensor(channels)) + b) / gamma))
        kernel_size = t if t % 2 == 1 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        if torch.cuda.is_available():
            self.avg_pool = self.avg_pool.cuda()
            self.conv = self.conv.cuda()
            self.sigmoid = self.sigmoid.cuda()
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


# 关键步骤：将 ECABlock 注入到 ultralytics.nn.tasks 的全局命名空间中
tasks_module.__dict__['ECABlock'] = ECABlock
print("Successfully registered ECABlock to ultralytics.nn.tasks!")

# 现在尝试加载模型
yaml_path = "/root/autodl-tmp/waste-classification/attention/yolov8n-cls-eca.yaml"
print(f"\nLoading model from {yaml_path}...")

try:
    model = YOLO(yaml_path)
    print("✓ Model loaded successfully!")
    
    # 尝试一次前向传播
    print("\nTesting forward pass...")
    test_input = torch.randn(1, 3, 224, 224)
    if torch.cuda.is_available():
        test_input = test_input.cuda()
        model = model.cuda()
    
    with torch.no_grad():
        output = model(test_input)
    print("✓ Forward pass successful!")
    print(f"Output type: {type(output)}")
    
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
