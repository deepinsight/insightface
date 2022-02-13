# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout, PReLU
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import XavierNormal, Constant

import math

__all__ = ["FresResNet50", "FresResNet100"]


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None,
                 data_format="NCHW"):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            data_format=data_format)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            epsilon=1e-05,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(bn_name + "_offset"),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + "_variance",
            data_layout=data_format)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 name=None,
                 data_format="NCHW"):
        super(BasicBlock, self).__init__()
        self.stride = stride
        bn_name = "bn_" + name[3:] + "_before"
        self._batch_norm = BatchNorm(
            num_channels,
            act=None,
            epsilon=1e-05,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(bn_name + "_offset"),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + "_variance",
            data_layout=data_format)

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            stride=1,
            act=None,
            name=name + "_branch2a",
            data_format=data_format)
        self.prelu = PReLU(num_parameters=1, name=name + "_branch2a_prelu")
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act=None,
            name=name + "_branch2b",
            data_format=data_format)

        if shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters,
                filter_size=1,
                stride=stride,
                act=None,
                name=name + "_branch1",
                data_format=data_format)

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self._batch_norm(inputs)
        y = self.conv0(y)
        y = self.prelu(y)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = self.short(inputs)
        else:
            short = inputs
        y = paddle.add(x=short, y=conv1)
        return y


class FC(nn.Layer):
    def __init__(self,
                 bn_channels,
                 num_channels,
                 num_classes,
                 fc_type,
                 dropout=0.4,
                 name=None,
                 data_format="NCHW"):
        super(FC, self).__init__()
        self.p = dropout
        self.fc_type = fc_type
        self.num_channels = num_channels

        bn_name = "bn_" + name
        if fc_type == "Z":
            self._batch_norm_1 = BatchNorm(
                bn_channels,
                act=None,
                epsilon=1e-05,
                param_attr=ParamAttr(name=bn_name + "_1_scale"),
                bias_attr=ParamAttr(bn_name + "_1_offset"),
                moving_mean_name=bn_name + "_1_mean",
                moving_variance_name=bn_name + "_1_variance",
                data_layout=data_format)
            if self.p > 0:
                self.dropout = Dropout(p=self.p, name=name + '_dropout')

        elif fc_type == "E":
            self._batch_norm_1 = BatchNorm(
                bn_channels,
                act=None,
                epsilon=1e-05,
                param_attr=ParamAttr(name=bn_name + "_1_scale"),
                bias_attr=ParamAttr(bn_name + "_1_offset"),
                moving_mean_name=bn_name + "_1_mean",
                moving_variance_name=bn_name + "_1_variance",
                data_layout=data_format)
            if self.p > 0:
                self.dropout = Dropout(p=self.p, name=name + '_dropout')
            self.fc = Linear(
                num_channels,
                num_classes,
                weight_attr=ParamAttr(
                    initializer=XavierNormal(fan_in=0.0), name=name + ".w_0"),
                bias_attr=ParamAttr(
                    initializer=Constant(), name=name + ".b_0"))
            self._batch_norm_2 = BatchNorm(
                num_classes,
                act=None,
                epsilon=1e-05,
                param_attr=ParamAttr(name=bn_name + "_2_scale"),
                bias_attr=ParamAttr(bn_name + "_2_offset"),
                moving_mean_name=bn_name + "_2_mean",
                moving_variance_name=bn_name + "_2_variance",
                data_layout=data_format)

        elif fc_type == "FC":
            self._batch_norm_1 = BatchNorm(
                bn_channels,
                act=None,
                epsilon=1e-05,
                param_attr=ParamAttr(name=bn_name + "_1_scale"),
                bias_attr=ParamAttr(bn_name + "_1_offset"),
                moving_mean_name=bn_name + "_1_mean",
                moving_variance_name=bn_name + "_1_variance",
                data_layout=data_format)
            self.fc = Linear(
                num_channels,
                num_classes,
                weight_attr=ParamAttr(
                    initializer=XavierNormal(fan_in=0.0), name=name + ".w_0"),
                bias_attr=ParamAttr(
                    initializer=Constant(), name=name + ".b_0"))
            self._batch_norm_2 = BatchNorm(
                num_classes,
                act=None,
                epsilon=1e-05,
                param_attr=ParamAttr(name=bn_name + "_2_scale"),
                bias_attr=ParamAttr(bn_name + "_2_offset"),
                moving_mean_name=bn_name + "_2_mean",
                moving_variance_name=bn_name + "_2_variance",
                data_layout=data_format)

    def forward(self, inputs):
        if self.fc_type == "Z":
            y = self._batch_norm_1(inputs)
            y = paddle.reshape(y, shape=[-1, self.num_channels])
            if self.p > 0:
                y = self.dropout(y)

        elif self.fc_type == "E":
            y = self._batch_norm_1(inputs)
            y = paddle.reshape(y, shape=[-1, self.num_channels])
            if self.p > 0:
                y = self.dropout(y)
            y = self.fc(y)
            y = self._batch_norm_2(y)

        elif self.fc_type == "FC":
            y = self._batch_norm_1(inputs)
            y = paddle.reshape(y, shape=[-1, self.num_channels])
            y = self.fc(y)
            y = self._batch_norm_2(y)

        return y


class FresResNet(nn.Layer):
    def __init__(self,
                 layers=50,
                 num_features=512,
                 fc_type='E',
                 dropout=0.4,
                 input_image_channel=3,
                 input_image_width=112,
                 input_image_height=112,
                 data_format="NCHW"):

        super(FresResNet, self).__init__()

        self.layers = layers
        self.data_format = data_format
        self.input_image_channel = input_image_channel

        supported_layers = [50, 100]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 50:
            units = [3, 4, 14, 3]
        elif layers == 100:
            units = [3, 13, 30, 3]

        num_channels = [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            num_channels=self.input_image_channel,
            num_filters=64,
            filter_size=3,
            stride=1,
            act=None,
            name="conv1",
            data_format=self.data_format)
        self.prelu = PReLU(num_parameters=1, name="prelu1")

        self.block_list = paddle.nn.LayerList()
        for block in range(len(units)):
            shortcut = True
            for i in range(units[block]):
                conv_name = "res" + str(block + 2) + chr(97 + i)
                basic_block = self.add_sublayer(
                    conv_name,
                    BasicBlock(
                        num_channels=num_channels[block]
                        if i == 0 else num_filters[block],
                        num_filters=num_filters[block],
                        stride=2 if shortcut else 1,
                        shortcut=shortcut,
                        name=conv_name,
                        data_format=self.data_format))
                self.block_list.append(basic_block)
                shortcut = False

        assert input_image_width % 16 == 0
        assert input_image_height % 16 == 0
        feat_w = input_image_width // 16
        feat_h = input_image_height // 16
        self.fc_channels = num_filters[-1] * feat_w * feat_h
        self.fc = FC(num_filters[-1],
                     self.fc_channels,
                     num_features,
                     fc_type,
                     dropout,
                     name='fc')

    def forward(self, inputs):
        if self.data_format == "NHWC":
            inputs = paddle.tensor.transpose(inputs, [0, 2, 3, 1])
            inputs.stop_gradient = True
        y = self.conv(inputs)
        y = self.prelu(y)
        for block in self.block_list:
            y = block(y)
        y = self.fc(y)
        return y


def FresResNet50(**args):
    model = FresResNet(layers=50, **args)
    return model


def FresResNet100(**args):
    model = FresResNet(layers=100, **args)
    return model
