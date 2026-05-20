"""
SE-Block (Squeeze-and-Excitation Block) 实现
用于YOLOv8分类模型的注意力机制
"""

import torch
import torch.nn as nn

class SELayer(nn.Module):
    """SE注意力机制模块

    位置：在AdaptiveAvgPool2d之后使用，对全局池化后的特征进行通道注意力加权

    工作原理：
    1. Squeeze（压缩）：全局平均池化，将特征图从 (N, C, H, W) 压缩为 (N, C)
    2. Excitation（激励）：通过全连接层学习每个通道的权重
    3. Scale（缩放）：用学到的权重重新校准原始特征
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BottleneckWithSE(nn.Module):
    """带SE注意力机制的瓶颈块

    适用于在YOLOv8的backbone中集成SE模块
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super(BottleneckWithSE, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, k[0] // 2, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, k[1] // 2, groups=g, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.se = SELayer(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        identity = x
        out = self.cv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.cv2(out)
        out = self.bn2(out)

        if self.add:
            out += identity

        out = self.se(out)
        return out
