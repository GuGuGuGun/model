from ultralytics.nn.modules.Net.CMTFNet import CMTFBlock, Fusion
import torch
from torch import nn
from timm.models.layers import DropPath, trunc_normal_
from ultralytics.nn.modules import Conv

class CMTFC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.cv3 = Conv(c1, c2, 1)
        self.m = nn.ModuleList(CMTFBlock(self.c,4) for _ in range(n))
        self.fusion = Fusion(c2)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        x = self.cv3(x)
        y = self.cv2(torch.cat(y, 1))
        return self.fusion(x,y)

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        x = self.cv3(x)
        y = self.cv2(torch.cat(y, 1))
        return self.fusion(x,y)




if __name__ == '__main__':
    t1 = torch.rand(1, 16, 256, 256)
    model = CMTFC2f(16, 32, n=1)
    print(model(t1).shape)