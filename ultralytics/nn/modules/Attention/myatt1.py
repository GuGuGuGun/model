import torch
import torch.nn as nn


class myatt(nn.Module):
    def __init__(self, in_channel, ratio):
        super().__init__()
        self.ratio = ratio
        self.conv = nn.Conv2d(in_channel, int(in_channel / ratio), 3, 1, 1)
        self.conv2 = nn.Conv2d(int(in_channel // ratio), in_channel, 3, 1, 1)
        self.activation = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1d = nn.Sequential(
            nn.Linear(int(in_channel / ratio), in_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel, in_channel, bias=False),
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = self.conv(x)
        x2 = self.avg_pool(self.conv(x)).view(b, int(c / self.ratio))
        x1 = self.activation(self.conv2(x1))
        x2 = self.fc1d(x2).view(b, c, 1, 1)
        x = x*x1*x2.expand_as(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


if __name__ == '__main__':
    t = torch.rand(3,256,64,64)
    att = myatt(256,8)
    t2 = att(t)
    print(t2.shape)