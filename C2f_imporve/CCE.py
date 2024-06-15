import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.Attention.CGAfusion import CGAFusion
from mmengine.model import BaseModule
from typing import Optional
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule


class CAA(BaseModule):
    """Context Anchor Attention"""

    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor

class ELA(nn.Module):
    def __init__(self, in_channels, phi='B'):
        super(ELA, self).__init__()
        Kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]
        groups = {'T': in_channels, 'B': in_channels, 'S': in_channels // 8, 'L': in_channels // 8}[phi]
        num_groups = {'T': 32, 'B': 16, 'S': 16, 'L': 16}[phi]
        pad = Kernel_size // 2
        self.con1 = nn.Conv1d(in_channels, in_channels, kernel_size=Kernel_size, padding=pad, groups=groups, bias=False)
        self.GN = nn.GroupNorm(num_groups, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        b, c, h, w = input.size()
        x_h = torch.mean(input, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(input, dim=2, keepdim=True).view(b, c, w)
        x_h = self.con1(x_h)  # [b,c,h]
        x_w = self.con1(x_w)  # [b,c,w]
        x_h = self.sigmoid(self.GN(x_h)).view(b, c, h, 1)  # [b, c, h, 1]
        x_w = self.sigmoid(self.GN(x_w)).view(b, c, 1, w)  # [b, c, 1, w]
        return x_h * x_w * input

class CAA_Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, k[0], 1, g=g)
        self.add = shortcut
        self.att = CAA(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.att(self.cv1(x)) if self.add else self.att(self.cv1(x))


class CCE_C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.cv3 = Conv(c1, c2, 1, 1)
        self.act = nn.Sigmoid()
        self.att = ELA(2 * self.c)
        self.m = nn.ModuleList(CAA_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.fusion = CGAFusion((c2,c2))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.att(self.cv1(x)).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        x = self.cv3(x)
        return self.act(self.fusion([x,y]))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.att(self.cv1(x)).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        x = self.cv3(x)
        return self.act(self.fusion([x,y]))


if __name__ == '__main__':
    t1 = torch.rand(1, 256, 64, 64)
    block = CCE_C2f(256, 512)
    print(block(t1).shape)
