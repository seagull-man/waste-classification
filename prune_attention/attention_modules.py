"""
注意力机制模块: SE (Squeeze-and-Excitation), CBAM, ECA
[参考文档] 实际使用的模块已内置在 ultralytics 的以下文件中:
  - /root/miniconda3/lib/python3.8/site-packages/ultralytics/nn/modules/block.py  (SEBlock, ECABlock)
  - /root/miniconda3/lib/python3.8/site-packages/ultralytics/nn/modules/conv.py   (CBAM)
本文件保留用于代码参考和理解，不被实际脚本引用。
"""
import torch
import torch.nn as nn


# ===================== SE Block =====================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block
    对通道维度进行注意力加权，参数增量极小。
    reduction: 压缩比例，默认 16
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weight = self.fc(x).view(b, c, 1, 1)
        return x * weight


# ===================== CBAM =====================
class ChannelAttention(nn.Module):
    """CBAM 的通道注意力子模块"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = nn.AdaptiveAvgPool2d(1)(x)
        max_pool = nn.AdaptiveMaxPool2d(1)(x)
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        out = torch.sigmoid(avg_out + max_out)
        b, c = out.shape[0], out.shape[1]
        return x * out.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM 的空间注意力子模块"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        weight = torch.sigmoid(self.conv(concat))
        return x * weight


class CBAM(nn.Module):
    """Convolutional Block Attention Module
    先通道注意力，再空间注意力。
    """
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ===================== ECA Block =====================
class ECABlock(nn.Module):
    """Efficient Channel Attention
    用一维卷积替代全连接层，实现局部跨通道交互，参数增量几乎为零。
    """
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # 自适应计算卷积核大小
        t = int(abs((torch.log2(torch.tensor(channels)) + b) / gamma))
        kernel_size = t if t % 2 == 1 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).squeeze(-1)           # (b, c, 1)
        y = y.transpose(-1, -2)                    # (b, 1, c)
        y = self.conv(y)                           # (b, 1, c)
        y = y.transpose(-1, -2).unsqueeze(-1)      # (b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y


# ===================== 工厂函数 =====================
def get_attention(att_type: str, channels: int, reduction: int = 16) -> nn.Module:
    """根据类型名称返回对应的注意力模块"""
    att_type = att_type.lower()
    if att_type == "se":
        return SEBlock(channels, reduction)
    elif att_type == "cbam":
        return CBAM(channels, reduction)
    elif att_type == "eca":
        return ECABlock(channels)
    else:
        raise ValueError(f"Unknown attention type: {att_type}. Choose from 'se', 'cbam', 'eca'.")