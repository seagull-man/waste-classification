import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """CBAM 通道注意力模块"""

    def __init__(self, in_planes, ratio=4):
        super().__init__()

        hidden_planes = max(in_planes // ratio, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_planes, in_planes, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    """CBAM 空间注意力模块"""

    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    """
    残差校正式 CBAM 模块

    out = x + gamma * (CBAM(x) - x)
    """

    def __init__(self, in_planes, ratio=4, kernel_size=7, gamma_init=0.1):
        super().__init__()

        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def forward(self, x):
        identity = x

        refined = self.channel_attention(x) * x
        refined = self.spatial_attention(refined) * refined

        return identity + self.gamma * (refined - identity)

class WrapperWithCBAM(nn.Module):
    """将 YOLOv8 原始层与 CBAM 包装在一起，并保留 Ultralytics 层属性"""

    def __init__(self, base_layer, channels):
        super().__init__()

        self.base = base_layer
        self.cbam = CBAM(channels)

        # 保留 Ultralytics 前向传播所需的层属性
        for attr in ["i", "f", "type", "np"]:
            if hasattr(base_layer, attr):
                setattr(self, attr, getattr(base_layer, attr))

        # 如果极端情况下 base_layer 没有 f，则默认来自上一层
        if not hasattr(self, "f"):
            self.f = -1

        if not hasattr(self, "i"):
            self.i = getattr(base_layer, "i", -1)

        if not hasattr(self, "type"):
            self.type = f"{self.__class__.__module__}.{self.__class__.__name__}"

        if not hasattr(self, "np"):
            self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        x = self.base(x)
        x = self.cbam(x)
        return x