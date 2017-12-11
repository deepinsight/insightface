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

"""
Contains the definition of the Inception Resnet V2 architecture.
As described in http://arxiv.org/abs/1602.07261.
Inception-v4, Inception-ResNet and the Impact of Residual Connections
on Learning
Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
import mxnet as mx
import symbol_utils


def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type="relu", mirror_attr={}, with_act=True):
    conv = mx.symbol.Convolution(
        data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    if with_act:
        act = mx.symbol.Activation(
            data=bn, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return bn


def block35(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}):
    tower_conv = ConvFactory(net, 32, (1, 1))
    tower_conv1_0 = ConvFactory(net, 32, (1, 1))
    tower_conv1_1 = ConvFactory(tower_conv1_0, 32, (3, 3), pad=(1, 1))
    tower_conv2_0 = ConvFactory(net, 32, (1, 1))
    tower_conv2_1 = ConvFactory(tower_conv2_0, 48, (3, 3), pad=(1, 1))
    tower_conv2_2 = ConvFactory(tower_conv2_1, 64, (3, 3), pad=(1, 1))
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_1, tower_conv2_2])
    tower_out = ConvFactory(
        tower_mixed, input_num_channels, (1, 1), with_act=False)

    net = net+scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def block17(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}):
    tower_conv = ConvFactory(net, 192, (1, 1))
    tower_conv1_0 = ConvFactory(net, 129, (1, 1))
    tower_conv1_1 = ConvFactory(tower_conv1_0, 160, (1, 7), pad=(1, 2))
    tower_conv1_2 = ConvFactory(tower_conv1_1, 192, (7, 1), pad=(2, 1))
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
    tower_out = ConvFactory(
        tower_mixed, input_num_channels, (1, 1), with_act=False)
    net = net+scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def block8(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}):
    tower_conv = ConvFactory(net, 192, (1, 1))
    tower_conv1_0 = ConvFactory(net, 192, (1, 1))
    tower_conv1_1 = ConvFactory(tower_conv1_0, 224, (1, 3), pad=(0, 1))
    tower_conv1_2 = ConvFactory(tower_conv1_1, 256, (3, 1), pad=(1, 0))
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
    tower_out = ConvFactory(
        tower_mixed, input_num_channels, (1, 1), with_act=False)
    net = net+scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def repeat(inputs, repetitions, layer, *args, **kwargs):
    outputs = inputs
    for i in range(repetitions):
        outputs = layer(outputs, *args, **kwargs)
    return outputs


def get_symbol(num_classes=1000, **kwargs):
    data = mx.symbol.Variable(name='data')
    data = data-127.5
    data = data*0.0078125
    version_input = kwargs.get('version_input', 1)
    assert version_input>=0
    version_output = kwargs.get('version_output', 'E')
    fc_type = version_output
    version_unit = kwargs.get('version_unit', 3)
    print(version_input, version_output, version_unit)

    if version_input==0:
      conv1a_3_3 = ConvFactory(data=data, num_filter=32,
                               kernel=(3, 3), stride=(2, 2))
    else:
      conv1a_3_3 = ConvFactory(data=data, num_filter=32,
                               kernel=(3, 3), stride=(1, 1))
    conv2a_3_3 = ConvFactory(conv1a_3_3, 32, (3, 3))
    conv2b_3_3 = ConvFactory(conv2a_3_3, 64, (3, 3), pad=(1, 1))
    maxpool3a_3_3 = mx.symbol.Pooling(
        data=conv2b_3_3, kernel=(3, 3), stride=(2, 2), pool_type='max')
    conv3b_1_1 = ConvFactory(conv2b_3_3, 80, (1, 1))
    conv4a_3_3 = ConvFactory(conv3b_1_1, 192, (3, 3))
    maxpool5a_3_3 = mx.symbol.Pooling(
        data=conv4a_3_3, kernel=(3, 3), stride=(2, 2), pool_type='max')

    tower_conv = ConvFactory(maxpool5a_3_3, 96, (1, 1))
    tower_conv1_0 = ConvFactory(maxpool5a_3_3, 48, (1, 1))
    tower_conv1_1 = ConvFactory(tower_conv1_0, 64, (5, 5), pad=(2, 2))

    tower_conv2_0 = ConvFactory(maxpool5a_3_3, 64, (1, 1))
    tower_conv2_1 = ConvFactory(tower_conv2_0, 96, (3, 3), pad=(1, 1))
    tower_conv2_2 = ConvFactory(tower_conv2_1, 96, (3, 3), pad=(1, 1))

    tower_pool3_0 = mx.symbol.Pooling(data=maxpool5a_3_3, kernel=(
        3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg')
    tower_conv3_1 = ConvFactory(tower_pool3_0, 64, (1, 1))
    tower_5b_out = mx.symbol.Concat(
        *[tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1])
    net = repeat(tower_5b_out, 10, block35, scale=0.17, input_num_channels=320)
    tower_conv = ConvFactory(net, 384, (3, 3), stride=(2, 2))
    tower_conv1_0 = ConvFactory(net, 256, (1, 1))
    tower_conv1_1 = ConvFactory(tower_conv1_0, 256, (3, 3), pad=(1, 1))
    tower_conv1_2 = ConvFactory(tower_conv1_1, 384, (3, 3), stride=(2, 2))
    tower_pool = mx.symbol.Pooling(net, kernel=(
        3, 3), stride=(2, 2), pool_type='max')
    net = mx.symbol.Concat(*[tower_conv, tower_conv1_2, tower_pool])
    net = repeat(net, 20, block17, scale=0.1, input_num_channels=1088)
    tower_conv = ConvFactory(net, 256, (1, 1))
    tower_conv0_1 = ConvFactory(tower_conv, 384, (3, 3), stride=(2, 2))
    tower_conv1 = ConvFactory(net, 256, (1, 1))
    tower_conv1_1 = ConvFactory(tower_conv1, 288, (3, 3), stride=(2, 2))
    tower_conv2 = ConvFactory(net, 256, (1, 1))
    tower_conv2_1 = ConvFactory(tower_conv2, 288, (3, 3), pad=(1, 1))
    tower_conv2_2 = ConvFactory(tower_conv2_1, 320, (3, 3),  stride=(2, 2))
    tower_pool = mx.symbol.Pooling(net, kernel=(
        3, 3), stride=(2, 2), pool_type='max')
    net = mx.symbol.Concat(
        *[tower_conv0_1, tower_conv1_1, tower_conv2_2, tower_pool])

    net = repeat(net, 9, block8, scale=0.2, input_num_channels=2080)
    net = block8(net, with_act=False, input_num_channels=2080)

    net = ConvFactory(data = net, num_filter=1536, kernel=(1, 1))
    body = net
    fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
    return fc1
