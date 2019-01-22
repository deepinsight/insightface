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
"""DenseNet, implemented in Gluon."""

import sys
import os
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag
import symbol_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config

def Act():
    if config.net_act=='prelu':
      return nn.PReLU()
    else:
      return nn.Activation(config.net_act)
# Helpers
def _make_dense_block(num_layers, bn_size, growth_rate, dropout, stage_index):
    out = nn.HybridSequential(prefix='stage%d_'%stage_index)
    with out.name_scope():
        for _ in range(num_layers):
            out.add(_make_dense_layer(growth_rate, bn_size, dropout))
    return out

def _make_dense_layer(growth_rate, bn_size, dropout):
    new_features = nn.HybridSequential(prefix='')
    new_features.add(nn.BatchNorm())
    #new_features.add(nn.Activation('relu'))
    new_features.add(Act())
    new_features.add(nn.Conv2D(bn_size * growth_rate, kernel_size=1, use_bias=False))
    new_features.add(nn.BatchNorm())
    #new_features.add(nn.Activation('relu'))
    new_features.add(Act())
    new_features.add(nn.Conv2D(growth_rate, kernel_size=3, padding=1, use_bias=False))
    if dropout:
        new_features.add(nn.Dropout(dropout))

    out = gluon.contrib.nn.HybridConcurrent(axis=1, prefix='')
    out.add(gluon.contrib.nn.Identity())
    out.add(new_features)

    return out

def _make_transition(num_output_features):
    out = nn.HybridSequential(prefix='')
    out.add(nn.BatchNorm())
    #out.add(nn.Activation('relu'))
    out.add(Act())
    out.add(nn.Conv2D(num_output_features, kernel_size=1, use_bias=False))
    out.add(nn.AvgPool2D(pool_size=2, strides=2))
    return out

# Net
class DenseNet(nn.HybridBlock):
    r"""Densenet-BC model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    num_init_features : int
        Number of filters to learn in the first convolution layer.
    growth_rate : int
        Number of filters to add each layer (`k` in the paper).
    block_config : list of int
        List of integers for numbers of layers in each pooling block.
    bn_size : int, default 4
        Multiplicative factor for number of bottle neck layers.
        (i.e. bn_size * k features in the bottleneck layer)
    dropout : float, default 0
        Rate of dropout after each dense layer.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self, num_init_features, growth_rate, block_config,
                 bn_size=4, dropout=0, classes=1000, **kwargs):

        super(DenseNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(num_init_features, kernel_size=3,
                                        strides=1, padding=1, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            # Add dense blocks
            num_features = num_init_features
            for i, num_layers in enumerate(block_config):
                self.features.add(_make_dense_block(num_layers, bn_size, growth_rate, dropout, i+1))
                num_features = num_features + num_layers * growth_rate
                if i != len(block_config) - 1:
                    self.features.add(_make_transition(num_features // 2))
                    num_features = num_features // 2
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            #self.features.add(nn.AvgPool2D(pool_size=7))
            #self.features.add(nn.Flatten())

            #self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        #x = self.output(x)
        return x


# Specification
densenet_spec = {121: (64, 32, [6, 12, 24, 16]),
                 161: (96, 48, [6, 12, 36, 24]),
                 169: (64, 32, [6, 12, 32, 32]),
                 201: (64, 32, [6, 12, 48, 32])}


# Constructor
def get_symbol():
    num_layers = config.num_layers
    num_init_features, growth_rate, block_config = densenet_spec[num_layers]
    net = DenseNet(num_init_features, growth_rate, block_config, dropout = config.densenet_dropout)
    data = mx.sym.Variable(name='data')
    data = data-127.5
    data = data*0.0078125
    body = net(data)
    fc1 = symbol_utils.get_fc1(body, config.emb_size, config.net_output)
    return fc1

