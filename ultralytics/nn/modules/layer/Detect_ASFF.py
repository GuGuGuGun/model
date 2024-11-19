import math

import torch
from torch import nn
from ultralytics.utils.tal import dist2bbox

from ultralytics.utils.tal import make_anchors
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import DFL


class ASFF_Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), rfb=False):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.l0_fusion = ASFF(level=0, rfb=rfb)
        self.l1_fusion = ASFF(level=1, rfb=rfb)
        self.l2_fusion = ASFF(level=2, rfb=rfb)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        x1 = self.l0_fusion(x)
        x2 = self.l1_fusion(x)
        x3 = self.l2_fusion(x)
        x = [x3, x2, x1]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)


    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage






class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False, multip=1):
        super(ASFF, self).__init__()
        self.level = level
        # 特征金字塔从上到下三层的channel数
        # 对应特征图大小(以640*640输入为例)分别为20*20, 40*40, 80*80
        self.dim = [int(256 * multip), int(128 * multip), int(64 * multip)]
        self.inter_dim = self.dim[self.level]
        if level == 0:  # 特征图最小的一层，channel数256
            self.stride_level_1 = Conv(int(128 * multip), self.inter_dim, 3, 2)
            self.stride_level_2 = Conv(int(64 * multip), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(256 * multip), 3, 1)
        elif level == 1:  # 特征图大小适中的一层，channel数128
            self.compress_level_0 = Conv(int(256 * multip), self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(int(64 * multip), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(128 * multip), 3, 1)
        elif level == 2:  # 特征图最大的一层，channel数64
            self.compress_level_0 = Conv(int(256 * multip), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(int(128 * multip), self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, int(64 * multip), 3, 1)

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


if __name__ == '__main__':
    t1 = torch.rand(1, 64, 32, 32)
    t2 = torch.rand(1, 128, 16, 16)
    t3 = torch.rand(1, 256, 8, 8)
    t = [t1,t2,t3]
    asff = ASFF(1)
    print(asff(t).shape)
