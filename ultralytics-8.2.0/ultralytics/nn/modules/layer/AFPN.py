import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.tal import dist2bbox, make_anchors
import math

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs.
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391"""

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        # Two different softmax methods, choose one to use
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False, multip=1):
        super(ASFF, self).__init__()
        self.level = level
        # 特征金字塔从上到下三层的channel数
        # 对应特征图大小(以640*640输入为例)分别为20*20, 40*40, 80*80
        self.dim = [int(256*multip), int(128*multip), int(64*multip)]
        self.inter_dim = self.dim[self.level]
        if level == 0:  # 特征图最小的一层，channel数512
            self.stride_level_1 = Conv(int(128*multip), self.inter_dim, 3, 2)
            self.stride_level_2 = Conv(int(64*multip), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(256*multip), 3, 1)
        elif level == 1:  # 特征图大小适中的一层，channel数256
            self.compress_level_0 = Conv(int(256*multip), self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(int(64*multip), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(128*multip), 3, 1)
        elif level == 2:  # 特征图最大的一层，channel数128
            self.compress_level_0 = Conv(int(256*multip), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(int(128*multip), self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, int(64*multip), 3, 1)

        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x):
        """
        Forward pass of the ASFF module.

        Args:
        - x (tuple of tensors): Input feature maps at different scales (l, m, s).
        """
        x_level_0 = x[2]  # Smallest scale feature map
        x_level_1 = x[1]  # Medium scale feature map
        x_level_2 = x[0]  # Largest scale feature map

        # Resize and compress feature maps based on the current level
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        # Calculate attention weights for each level
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        # Fuse the feature maps weighted by the attention scores
        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] \
                            + level_1_resized * levels_weight[:, 1:2, :, :] \
                            + level_2_resized * levels_weight[:, 2:, :, :]
        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out