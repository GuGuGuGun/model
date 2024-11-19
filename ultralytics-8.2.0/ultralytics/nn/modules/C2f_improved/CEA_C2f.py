import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.Attention import SEAttention
from ultralytics.nn.modules.Attention.ELA import ELA
from ultralytics.nn.modules.Attention.CAA import CAA
from ultralytics.nn.modules.Attention.ShuffleAttention import ShuffleAttention


class ShuffleAttention_Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, k[0], 1, g=g)
        self.add = shortcut
        self.att = ShuffleAttention(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.att(self.cv1(x)) if self.add else self.att(self.cv1(x))


class CEA_C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv3 = Conv((2 + n) * self.c, c2, 1, 1)
        self.act = nn.Sigmoid()
        self.att = ShuffleAttention((2 + n) * self.c)
        self.att_in_y = SEAttention(self.c,8)
        self.m = nn.ModuleList(ShuffleAttention_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1])*self.att_in_y(y[-1]) for m in self.m)
        y = self.att(torch.cat(y, 1))
        return self.act(self.cv3(y))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1])*self.att_in_y(y[-1]) for m in self.m)
        y = self.att(torch.cat(y, 1))
        return self.act(self.cv3(y))


if __name__ == '__main__':
    t1 = torch.rand(1, 256, 64, 64)
    block = CEA_C2f(256, 512)
    print(block(t1).shape)
