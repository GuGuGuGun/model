import torch
from torch import nn
from ultralytics.nn.modules import DWConv


class A_SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = DWConv(c1, c_, 1, 1)
        self.cv2 = DWConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.act = nn.ReLU()

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]+self.m(y[0])) for _ in range(3))
        return self.act(self.cv2(torch.cat(y, 1)))



if __name__ == '__main__':
    t1 = torch.rand(1,64,64,64)
    layer = SPPF(64,128)
    print(layer(t1).shape)
