import torch
from torch import nn

from ultralytics.nn.modules import Conv


class GSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)  # g:gract：分组卷积
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)  # 分组为c_

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 1)

if __name__ == '__main__':
    t1 = torch.rand(1,3,64,64)
    block = GSConv(3,64,3,2)
    print(block(t1).shape)