# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ
"""ResNets, implemented in Gluon."""
from __future__ import division

#__all__ = ['ResNetV1', 'ResNetV2',
#           'BasicBlockV1', 'BasicBlockV2',
#           'BottleneckV1', 'BottleneckV2',
#           'resnet18_v1', 'resnet34_v1', 'resnet50_v1', 'resnet101_v1', 'resnet152_v1',
#           'resnet18_v2', 'resnet34_v2', 'resnet50_v2', 'resnet101_v2', 'resnet152_v2',
#           'get_resnet']

import os

#from ....context import cpu
from mxnet import gluon
from mxnet import profiler
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock

# Helpers
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)

def _act(act_type):
  if act_type=='prelu':
    return nn.PReLU()
  else:
    return nn.Activation(act_type)

# Blocks
class BasicBlockV1(HybridBlock):
    r"""BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, act_type = 'relu', **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        self.act_type = act_type
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, 1, in_channels))
        self.body.add(nn.BatchNorm(epsilon=2e-5))
        self.body.add(_act(act_type))
        self.body.add(_conv3x3(channels, stride, channels))
        self.body.add(nn.BatchNorm(epsilon=2e-5))
        if self.act_type=='prelu':
          self.prelu = nn.PReLU()
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm(epsilon=2e-5))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        if self.act_type=='prelu':
          x = self.prelu(x+residual)
          #x = F.LeakyReLU(residual+x, act_type = self.act_type)
        else:
          x = F.Activation(x+residual, act_type=self.act_type)

        return x


class BasicBlockV2(HybridBlock):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BasicBlockV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = _conv3x3(channels, stride, in_channels)
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv3x3(channels, 1, channels)
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        return x + residual


class ResNet(HybridBlock):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    """
    def __init__(self, layers, channels, **kwargs):
        version_unit = kwargs.get('version_unit', 1)
        act_type = kwargs.get('version_act', 'prelu')
        self.act_type = act_type
        del kwargs['version_unit']
        del kwargs['version_act']
        super(ResNet, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        print(version_unit, act_type)
        if version_unit==1:
          block = BasicBlockV1
        elif version_unit==2:
          block = BasicBlockV2
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            #self.features.add(nn.BatchNorm(scale=False, center=False))
            #self.features.add(nn.BatchNorm())
            self.features.add(_conv3x3(channels[0], 1, 0))
            self.features.add(nn.BatchNorm(epsilon=2e-5))
            self.features.add(_act(act_type))

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                #stride = 1 if i == 0 else 2
                stride = 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=in_channels))
                in_channels = channels[i+1]
            #self.features.add(nn.BatchNorm())
            #self.features.add(nn.Activation('relu'))
            #self.features.add(nn.GlobalAvgPool2D())
            #self.features.add(nn.Flatten())

            #self.output = nn.Dense(classes, in_units=in_channels)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            #print(channels, in_channels)
            layer.add(block(channels, stride, True, in_channels=in_channels, act_type = self.act_type,
                            prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, act_type = self.act_type, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = x-127.5
        x = x*0.0078125
        x = self.features(x)
        return x


# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('basic_block', [3, 4, 14, 3], [64, 64, 128, 256, 512]),
               100: ('basic_block', [3, 13, 30, 3], [64, 64, 128, 256, 512]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}


# Constructor
def get(num_layers, **kwargs):
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    net = ResNet(layers, channels, **kwargs)
    return net

