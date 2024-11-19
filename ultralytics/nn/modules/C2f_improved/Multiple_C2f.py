import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.Attention.ELA import ELA
from ultralytics.nn.modules.Attention.CAA import CAA



class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Multiple_C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1=(), c2=128, n=1, shortcut=False, g=1, e=0.5,is_upsample=False):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.is_upsample = is_upsample
        self.c = int(c2 * e)  # hidden channels
        self.cv1_1 = Conv(c1[0], 2 * self.c, 1, 1)
        self.cv1_2 = Conv(c1[1], 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1,1)  # optional act=FReLU(c2)
        self.act = nn.Sigmoid()
        self.relu = nn.ReLU6()
        self.m = nn.ModuleList(Block(self.c) for _ in range(n))
        if self.is_upsample:
            self.Upsample = nn.Upsample(None, 2, "nearest")


    def forward(self, x):

        if self.is_upsample:
            x[0] = self.Upsample(x[0])
        x1 = self.cv1_1(x[0])
        x2 = self.cv1_2(x[1])
        x = self.relu(x1*x2)
        y = list(x.chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        return y

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        x1 = self.cv1_1(x[0])
        x2 = self.cv1_2(x[1])
        x = self.act(x1*x2)
        y = list(x.split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        return y


if __name__ == '__main__':
    t1 = torch.rand(1, 256, 64, 64)
    t2 = torch.rand(1, 128, 64, 64)
    block = Multiple_C2f([256, 128],512)
    print(block([t1,t2]).shape)
