import torch
from onnx.reference.ops.op_sigmoid import sigmoid
from torch import nn
from timm.models.layers import DropPath, trunc_normal_
from ultralytics.nn.modules import Conv


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class ConvBN_Relu(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,dilation=2, padding=None,groups=1, use_bn=True,study_bn=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, autopad(kernel_size, padding, dilation), dilation, groups)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channel)
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)
            if study_bn:
                self.w = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'bn'):
            if hasattr(self, 'w'):
                x = self.bn(x) * self.w
            else:
                x = self.bn(x)
        x = self.relu(x)
        return x



class ConvBN_Sigmoid(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,dilation=1, padding=0,groups=1, use_bn=True,study_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channel)
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)
            if study_bn:
                self.w = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'bn'):
            if hasattr(self, 'w'):
                weight = self.w
                x = self.bn(x) * weight
            x = self.sigmoid(x)
        else:
            x = self.sigmoid(x)
        return x
    
    
class LBlock(nn.Module):
    def __init__(self,in_channel,mlp_ratio=3,n=3,drop=0.,use_time_study_ratio=True):
        super().__init__()
        self.cv1_s = ConvBN_Relu(in_channel,in_channel*mlp_ratio,study_bn=False)
        last_input = sum(int(in_channel / 2 ** ratio if in_channel / 2 ** ratio >= 16 else 16) for ratio in range(n))
        self.m = nn.ModuleList(ConvBN_Relu(in_channel*mlp_ratio,int(in_channel/2**ratio if in_channel/2**ratio>=16 else 16),use_bn=False) for ratio in range(n))
        self.cv2_s = ConvBN_Relu(last_input,in_channel,study_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop) if drop > 0. else nn.Identity()
        if use_time_study_ratio:
            self.w = nn.Parameter(torch.ones(n,dtype=torch.float32),requires_grad=True)
            self.epsilon = 0.0001

    def forward(self, x):
        input = x
        x = self.cv1_s(x)
        if hasattr(self, 'w'):
            w = self.w
            weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 归一化
            xn = [m(x) for m in self.m]
            xn = [xi * weight[i] for i, xi in enumerate(xn)]
            x = self.act(torch.cat(xn, 1))
        else:
            xn = [m(x) for m in self.m]
            x = self.act(torch.cat(xn, 1))
        x = self.cv2_s(x)
        x = input + self.drop_path(x)
        return x

class LNet(nn.Module):
    def __init__(self,input):
        super().__init__()
        


class LBlock_C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(LBlock(self.c,2,2) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


if __name__ == '__main__':
    t1 = torch.rand(1,128,256,256)
    layer = LBlock(128)
    print(layer(t1).shape)

