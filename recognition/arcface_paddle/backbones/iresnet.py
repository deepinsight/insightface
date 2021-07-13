# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# reference: https://raw.githubusercontent.com/GuoQuanhao/arcface-Paddle/main/backbones/iresnet.py

import paddle
from paddle import nn

__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias_attr=False,
        dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2D(
        in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)


class IBasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2D(inplanes, epsilon=1e-05, momentum=0.1)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2D(planes, epsilon=1e-05, momentum=0.1)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2D(planes, epsilon=1e-05, momentum=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Layer):
    fc_scale = 7 * 7

    def __init__(self,
                 block,
                 layers,
                 dropout=0,
                 num_features=512,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2D(
            3,
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.bn1 = nn.BatchNorm2D(self.inplanes, epsilon=1e-05, momentum=0.1)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2D(
            512 * block.expansion, epsilon=1e-05, momentum=0.1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale,
                            num_features)
        self.features = nn.BatchNorm1D(
            num_features, momentum=0.1, epsilon=1e-05)
        self.features.weight = paddle.create_parameter(
            shape=self.features.weight.shape,
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=1.0))
        # nn.init.constant_(self.features.weight, 1.0)
        # 修改了stop_gradient，将True设为False
        self.features.weight.stop_gradient = False
        #self.features.weight.requires_grad = False

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                m.weight = paddle.create_parameter(
                    shape=m.weight.shape,
                    dtype='float32',
                    default_initializer=nn.initializer.Normal(
                        mean=0.0, std=0.1))
                # nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
                m.weight = paddle.create_parameter(
                    shape=m.weight.shape,
                    dtype='float32',
                    default_initializer=nn.initializer.Constant(value=1.0))
                m.bias = paddle.create_parameter(
                    shape=m.bias.shape,
                    dtype='float32',
                    default_initializer=nn.initializer.Constant(value=0.0))
                # nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.sublayers():
                if isinstance(m, IBasicBlock):
                    m.bn2.weight = paddle.create_parameter(
                        shape=m.bn2.weight.shape,
                        dtype='float32',
                        default_initializer=nn.initializer.Constant(value=0.0))
                    # nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2D(
                    planes * block.expansion, epsilon=1e-05, momentum=0.1), )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        with paddle.amp.auto_cast():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = paddle.cast(x, dtype='float32')
            x = paddle.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(paddle.cast(x, dtype='float16') if self.fp16 else x)
        x = self.features(x)
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)
