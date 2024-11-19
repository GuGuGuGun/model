"""
Haar Wavelet-based Downsampling (HWD)
Original address of the paper: https://www.sciencedirect.com/science/article/abs/pii/S0031320323005174
Code reference: https://github.com/apple1986/HWD/tree/main
"""
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
from ultralytics.nn.modules import Conv


class HWDownsamp(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HWDownsamp, self).__init__()
        self.wt = DWTForward(J=1, wave='haar', mode='zero')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channel * 4, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x


class HWD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        from pytorch_wavelets import DWTForward
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv = Conv(in_ch * 4, out_ch, 1, 1)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv(x)

        return x


if __name__ == '__main__':
    downsampling_layer = HWD(3, 64)
    input_data = torch.rand((1, 3, 64, 64))
    output_data = downsampling_layer(input_data)
    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)