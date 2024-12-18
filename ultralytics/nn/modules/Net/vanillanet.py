# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import weight_init
from timm.models.registry import register_model


class activation(nn.ReLU):
    # 初始化激活函数
    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        # 判断是否是部署模式
        self.deploy = deploy
        # 初始化权重
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num * 2 + 1, act_num * 2 + 1))
        # 初始化偏置
        self.bias = None
        # 初始化BatchNorm2d
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        # 设置维度
        self.dim = dim
        # 设置激活函数个数
        self.act_num = act_num
        # 初始化权重
        weight_init.trunc_normal_(self.weight, std=.02)

    def forward(self, x):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, self.bias, padding=(self.act_num * 2 + 1) // 2, groups=self.dim)
        else:
            return self.bn(torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, padding=(self.act_num * 2 + 1) // 2, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True


class vanillanetBlock(nn.Module):
    # 初始化vanillanetBlock类
    def __init__(self, dim, dim_out, act_num=3, stride=2, deploy=False, ada_pool=None):
        super().__init__()
        # 定义激活函数
        self.act_learn = 1
        self.deploy = deploy
        # 如果deploy为True，则将conv设置为Conv2d，否则设置为Conv2d + BatchNorm2d
        if self.deploy:
            self.conv = nn.Conv2d(dim, dim_out, kernel_size=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim, eps=1e-6),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out, eps=1e-6)
            )

        # 如果不是ada_pool，则将pool设置为Identity，否则设置为AdaptiveMaxPool2d
        if not ada_pool:
            self.pool = nn.Identity() if stride == 1 else nn.MaxPool2d(stride)
        else:
            self.pool = nn.Identity() if stride == 1 else nn.AdaptiveMaxPool2d((ada_pool, ada_pool))

        # 定义激活函数
        self.act = activation(dim_out, act_num)

    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            x = torch.nn.functional.leaky_relu(x, self.act_learn)
            x = self.conv2(x)

        x = self.pool(x)
        x = self.act(x)
        return x

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.conv1[0], self.conv1[1])
        self.conv1[0].weight.data = kernel
        self.conv1[0].bias.data = bias
        # kernel, bias = self.conv2[0].weight.data, self.conv2[0].bias.data
        kernel, bias = self._fuse_bn_tensor(self.conv2[0], self.conv2[1])
        self.conv = self.conv2[0]
        self.conv.weight.data = torch.matmul(kernel.transpose(1, 3),
                                             self.conv1[0].weight.data.squeeze(3).squeeze(2)).transpose(1, 3)
        self.conv.bias.data = bias + (self.conv1[0].bias.data.view(1, -1, 1, 1) * kernel).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.act.switch_to_deploy()
        self.deploy = True


class VanillaNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, dims=[96, 192, 384, 768],
                 drop_rate=0, act_num=3, strides=[2, 2, 2, 1], deploy=False, ada_pool=None, **kwargs):
        super().__init__()
        self.deploy = deploy
        if self.deploy:
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                activation(dims[0], act_num)
            )
        else:
            self.stem1 = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                nn.BatchNorm2d(dims[0], eps=1e-6),
            )
            self.stem2 = nn.Sequential(
                nn.Conv2d(dims[0], dims[0], kernel_size=1, stride=1),
                nn.BatchNorm2d(dims[0], eps=1e-6),
                activation(dims[0], act_num)
            )

        self.act_learn = 1

        self.stages = nn.ModuleList()
        for i in range(len(strides)):
            if not ada_pool:
                stage = vanillanetBlock(dim=dims[i], dim_out=dims[i + 1], act_num=act_num, stride=strides[i],
                                        deploy=deploy)
            else:
                stage = vanillanetBlock(dim=dims[i], dim_out=dims[i + 1], act_num=act_num, stride=strides[i],
                                        deploy=deploy, ada_pool=ada_pool[i])
            self.stages.append(stage)
        self.depth = len(strides)

        if self.deploy:
            self.cls = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Dropout(drop_rate),
                nn.Conv2d(dims[-1], num_classes, 1),
            )
        else:
            self.cls1 = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Dropout(drop_rate),
                nn.Conv2d(dims[-1], num_classes, 1),
                nn.BatchNorm2d(num_classes, eps=1e-6),
            )
            self.cls2 = nn.Sequential(
                nn.Conv2d(num_classes, num_classes, 1)
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            weight_init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def change_act(self, m):
        for i in range(self.depth):
            self.stages[i].act_learn = m
        self.act_learn = m

    def forward(self, x):
        if self.deploy:
            x = self.stem(x)
        else:
            x = self.stem1(x)
            x = torch.nn.functional.leaky_relu(x, self.act_learn)
            x = self.stem2(x)

        for i in range(self.depth):
            x = self.stages[i](x)

        if self.deploy:
            x = self.cls(x)
        else:
            x = self.cls1(x)
            x = torch.nn.functional.leaky_relu(x, self.act_learn)
            x = self.cls2(x)
        return x.view(x.size(0), -1)

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std

    def switch_to_deploy(self):
        self.stem2[2].switch_to_deploy()
        kernel, bias = self._fuse_bn_tensor(self.stem1[0], self.stem1[1])
        self.stem1[0].weight.data = kernel
        self.stem1[0].bias.data = bias
        kernel, bias = self._fuse_bn_tensor(self.stem2[0], self.stem2[1])
        self.stem1[0].weight.data = torch.einsum('oi,icjk->ocjk', kernel.squeeze(3).squeeze(2),
                                                 self.stem1[0].weight.data)
        self.stem1[0].bias.data = bias + (self.stem1[0].bias.data.view(1, -1, 1, 1) * kernel).sum(3).sum(2).sum(1)
        self.stem = torch.nn.Sequential(*[self.stem1[0], self.stem2[2]])
        self.__delattr__('stem1')
        self.__delattr__('stem2')

        for i in range(self.depth):
            self.stages[i].switch_to_deploy()

        kernel, bias = self._fuse_bn_tensor(self.cls1[2], self.cls1[3])
        self.cls1[2].weight.data = kernel
        self.cls1[2].bias.data = bias
        kernel, bias = self.cls2[0].weight.data, self.cls2[0].bias.data
        self.cls1[2].weight.data = torch.matmul(kernel.transpose(1, 3),
                                                self.cls1[2].weight.data.squeeze(3).squeeze(2)).transpose(1, 3)
        self.cls1[2].bias.data = bias + (self.cls1[2].bias.data.view(1, -1, 1, 1) * kernel).sum(3).sum(2).sum(1)
        self.cls = torch.nn.Sequential(*self.cls1[0:3])
        self.__delattr__('cls1')
        self.__delattr__('cls2')
        self.deploy = True


@register_model
def vanillanet_5(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(dims=[128 * 4, 256 * 4, 512 * 4, 1024 * 4], strides=[2, 2, 2], **kwargs)
    return model


@register_model
def vanillanet_6(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(dims=[128 * 4, 256 * 4, 512 * 4, 1024 * 4, 1024 * 4], strides=[2, 2, 2, 1], **kwargs)
    return model


@register_model
def vanillanet_7(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 1024 * 4, 1024 * 4], strides=[1, 2, 2, 2, 1], **kwargs)
    return model


@register_model
def vanillanet_8(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 512 * 4, 1024 * 4, 1024 * 4],
                       strides=[1, 2, 2, 1, 2, 1], **kwargs)
    return model


@register_model
def vanillanet_9(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 512 * 4, 512 * 4, 1024 * 4, 1024 * 4],
                       strides=[1, 2, 2, 1, 1, 2, 1], **kwargs)
    return model


@register_model
def vanillanet_10(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 1024 * 4, 1024 * 4],
        strides=[1, 2, 2, 1, 1, 1, 2, 1],
        **kwargs)
    return model


@register_model
def vanillanet_11(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 1024 * 4, 1024 * 4],
        strides=[1, 2, 2, 1, 1, 1, 1, 2, 1],
        **kwargs)
    return model


@register_model
def vanillanet_12(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 1024 * 4, 1024 * 4],
        strides=[1, 2, 2, 1, 1, 1, 1, 1, 2, 1],
        **kwargs)
    return model


@register_model
def vanillanet_13(pretrained=False, in_22k=False, **kwargs):
    '''
    VanillaNet-13
    :param pretrained: 是否使用预训练模型
    :param in_22k: 是否使用22k帧的图像
    :param kwargs: 其他参数
    :return: VanillaNet
    '''
    model = VanillaNet(
        dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 1024 * 4,
              1024 * 4],
        strides=[1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1],
        **kwargs)
    return model


@register_model
def vanillanet_13_x1_5(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128 * 6, 128 * 6, 256 * 6, 512 * 6, 512 * 6, 512 * 6, 512 * 6, 512 * 6, 512 * 6, 512 * 6, 1024 * 6,
              1024 * 6],
        strides=[1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1],
        **kwargs)
    return model


@register_model
def vanillanet_13_x1_5_ada_pool(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128 * 6, 128 * 6, 256 * 6, 512 * 6, 512 * 6, 512 * 6, 512 * 6, 512 * 6, 512 * 6, 512 * 6, 1024 * 6,
              1024 * 6],
        strides=[1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1],
        ada_pool=[0, 40, 20, 0, 0, 0, 0, 0, 0, 10, 0],
        **kwargs)
    return model

if __name__ == '__main__':
    t1 = torch.rand(1,3,16,16)
    block = vanillanetBlock(3,16)
    print(block(t1).shape)