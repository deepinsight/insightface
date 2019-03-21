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

import sys
import os
import mxnet as mx
import symbol_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config

def Act(data, act_type, name):
    #ignore param act_type, set it in this function 
    if act_type=='prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    else:
      body = mx.sym.Activation(data=data, act_type=act_type, name=name)
    return body

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    act = Act(data=bn, act_type=config.net_act, name='%s%s_relu' %(name, suffix))
    return act

def ConvOnly(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    return conv

def get_symbol():
    num_classes = config.emb_size
    bn_mom = config.bn_mom
    workspace = config.workspace
    data = mx.symbol.Variable(name="data") # 224
    data = data-127.5
    data = data*0.0078125
    fc_type = config.net_output
    bf = int(32*config.net_multiplier)
    if config.net_input==0:
      conv_1 = Conv(data, num_filter=bf, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_1") # 224/112
    else:
      conv_1 = Conv(data, num_filter=bf, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_1") # 224/112
    conv_2_dw = Conv(conv_1, num_group=bf, num_filter=bf, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_2_dw") # 112/112
    conv_2 = Conv(conv_2_dw, num_filter=bf*2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_2") # 112/112
    conv_3_dw = Conv(conv_2, num_group=bf*2, num_filter=bf*2, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_3_dw") # 112/56
    conv_3 = Conv(conv_3_dw, num_filter=bf*4, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_3") # 56/56
    conv_4_dw = Conv(conv_3, num_group=bf*4, num_filter=bf*4, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_4_dw") # 56/56
    conv_4 = Conv(conv_4_dw, num_filter=bf*4, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_4") # 56/56
    conv_5_dw = Conv(conv_4, num_group=bf*4, num_filter=bf*4, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_5_dw") # 56/28
    conv_5 = Conv(conv_5_dw, num_filter=bf*8, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_5") # 28/28
    conv_6_dw = Conv(conv_5, num_group=bf*8, num_filter=bf*8, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_6_dw") # 28/28
    conv_6 = Conv(conv_6_dw, num_filter=bf*8, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6") # 28/28
    conv_7_dw = Conv(conv_6, num_group=bf*8, num_filter=bf*8, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_7_dw") # 28/14
    conv_7 = Conv(conv_7_dw, num_filter=bf*16, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_7") # 14/14

    conv_8_dw = Conv(conv_7, num_group=bf*16, num_filter=bf*16, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_8_dw") # 14/14
    conv_8 = Conv(conv_8_dw, num_filter=bf*16, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_8") # 14/14
    conv_9_dw = Conv(conv_8, num_group=bf*16, num_filter=bf*16, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_9_dw") # 14/14
    conv_9 = Conv(conv_9_dw, num_filter=bf*16, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_9") # 14/14
    conv_10_dw = Conv(conv_9, num_group=bf*16, num_filter=bf*16, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_10_dw") # 14/14
    conv_10 = Conv(conv_10_dw, num_filter=bf*16, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_10") # 14/14
    conv_11_dw = Conv(conv_10, num_group=bf*16, num_filter=bf*16, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_11_dw") # 14/14
    conv_11 = Conv(conv_11_dw, num_filter=bf*16, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_11") # 14/14
    conv_12_dw = Conv(conv_11, num_group=bf*16, num_filter=bf*16, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_12_dw") # 14/14
    conv_12 = Conv(conv_12_dw, num_filter=bf*16, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_12") # 14/14

    conv_13_dw = Conv(conv_12, num_group=bf*16, num_filter=bf*16, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_13_dw") # 14/7
    conv_13 = Conv(conv_13_dw, num_filter=bf*32, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_13") # 7/7
    conv_14_dw = Conv(conv_13, num_group=bf*32, num_filter=bf*32, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_14_dw") # 7/7
    conv_14 = Conv(conv_14_dw, num_filter=bf*32, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_14") # 7/7
    body = conv_14
    fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
    return fc1

