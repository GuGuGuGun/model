import torch
from torch import nn

from ultralytics.nn.modules import Conv


class BiSPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.e = 0.0001
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.e)
        y = [self.cv1(x)]
        y.extend(w * self.m(y[-1]) for w in weight)
        return self.cv2(torch.cat(y, 1))

if __name__ == '__main__':
    t1 = torch.rand(1, 256, 64, 64)
    block = BiSPPF(256, 512)
    print(block(t1).shape)
