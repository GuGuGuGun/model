import torch
import torch.nn as nn
import torch.nn.functional as F




class ConvBNReLU6(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, groups=1, bias=False):
        super().__init__()
        self.act = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()
        )
    def forward(self, x):
        return self.act(x)


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, groups=1, bias=False):
        super().__init__()
        self.act = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=groups),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        return self.act(x)



class MutilHead(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0., dilation=None):
        super().__init__()
        if dilation is None:
            dilation = [3, 5, 7]
        self.cv0_0 = ConvBNReLU6(in_channels, 64, kernel_size=3, stride=1,dilation=dilation[0])
        self.cv0_1 = ConvBN(64, 128, kernel_size=7, stride=1, dilation=dilation[1])
        self.cv0_2 = ConvBNReLU6(128, out_channels, kernel_size=1, stride=1, dilation=dilation[2])
        self.cv1_0 = ConvBNReLU6(in_channels, 128, kernel_size=3, stride=1, dilation=dilation[0])
        self.cv1_1 = ConvBN(128, 256, kernel_size=7, stride=1, dilation=dilation[1])
        self.cv1_2 = ConvBNReLU6(256, out_channels, kernel_size=1, stride=1, dilation=dilation[2])
        self.w = nn.Parameter(torch.ones(2), requires_grad=True)
        self.act = nn.ReLU6()
        self.drop = nn.Dropout(drop_rate)
        self.eps = 1e-6


    def forward(self, x):
        weights = self.w
        weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x0 = self.cv0_0(x)
        x0 = self.cv0_1(x0)
        x0 = self.cv0_2(x0)
        x1 = self.cv1_0(x)
        x1 = self.cv1_1(x1)
        x1 = self.cv1_2(x1)
        x = self.act(weights[0] * x0 * weights[1] * x1)
        x = self.drop(x)
        return x

class MutilLayer(nn.Module):
    def __init__(self,input_channels, output_channels):
        super().__init__()
        self.MHeda = MutilHead(input_channels, output_channels)
        self.MHeda1 = MutilHead(input_channels, output_channels)
        self.cv1 = ConvBNReLU6(2*output_channels,output_channels)
        self.cv2 = ConvBNReLU6(input_channels,output_channels)
        self.w = nn.Parameter(torch.ones(2),requires_grad=True)
        self.act = nn.ReLU6()
        self.eps = 1e-6
        self.init = nn.Identity()


    def forward(self,x):
        x_clone = self.init(x)
        weights = self.w
        weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x0 = self.MHeda(x)
        x1 = self.MHeda1(x)
        x0 = weights[0] * x0
        x1 = weights[1] * x1
        x = self.act(torch.cat([x0,x1],1))
        x = self.cv1(x)
        x = self.act(self.cv2(x_clone)*x)
        return x

if __name__ == '__main__':
    model = MutilLayer(3, 64)
    x = torch.randn(1, 3, 640, 640)
    print(model(x).shape)