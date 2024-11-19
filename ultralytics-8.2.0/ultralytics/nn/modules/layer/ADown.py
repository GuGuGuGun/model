import torch
from torch import nn

from ultralytics.nn.modules import Conv

try:
    from mmcv.cnn import build_activation_layer, build_norm_layer
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
    from mmengine.model import constant_init, normal_init
except ImportError as e:
    pass

"""
论文地址：https://arxiv.org/pdf/2208.03641v1.pdf
"""


class HWD_ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        # self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv1 = HWD(c1 // 2, self.c, 3, 1, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class HWD(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        super(HWD, self).__init__()
        from pytorch_wavelets import DWTForward
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv = Conv(in_ch * 4, out_ch, k, s, p)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv(x)
        return x


if __name__ == '__main__':
    downsampling_layer = HWD_ADown(64, 128)
    input_data = torch.rand((1, 64, 64, 64))
    output_data = downsampling_layer(input_data)
    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)