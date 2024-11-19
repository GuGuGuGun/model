import torch
import torch.nn as nn



class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)
class T_Layer(nn.Module):
    def __init__(self,in_channels,out_channels,ratio=2):
        super().__init__()
        self.conv1 = ConvBN(in_channels, in_channels*ratio, 3, 2, 1)
        self.conv2 = ConvBN(in_channels, in_channels*ratio, 3, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear_concat = nn.Linear(in_channels*ratio*2, in_channels)
        self.linear_multiple = nn.Linear(in_channels * ratio, in_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv3 = ConvBN(in_channels, out_channels, 3, 1, 1)



    def forward(self,x):
        b_x, c_x, h_x, w_x = x.size()
        x1 = self.conv1(x)
        b_x1, c_x1,_, _ = x1.size()
        x2 = self.conv2(x)
        b_x2, c_x2, _, _ = x1.size()
        x1 = self.avg_pool(x1).view(b_x1, c_x1)
        x2 = self.avg_pool(x2).view(b_x2, c_x2)
        z = self.relu(x1 * x2)
        z = self.linear_multiple(z).view(b_x, -1, 1, 1)
        y = torch.cat([x1,x2],dim=1)
        y = self.relu(self.linear_concat(y).view(b_x, -1, 1, 1))
        y = self.sigmoid(z*y)
        return self.conv3(x*y.expand_as(x))

if __name__ == '__main__':
    t1 = torch.rand(1,256,64,64)
    layer = Layer(256,512)
    print(layer(t1).shape)

