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

import mxnet as mx

def Act(data, act_type, name):
    #ignore param act_type, set it in this function 
    body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    #act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
    return body

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    act = Act(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act

def get_symbol(num_classes, **kwargs):
    data = mx.symbol.Variable(name="data") # 224
    data = data-127.5
    data = data*0.0078125
    version_input = kwargs.get('version_input', 0)
    assert version_input>=0
    version_output = kwargs.get('version_output', 'A')
    fc_type = version_output
    version_unit = kwargs.get('version_unit', 1)
    print(version_input, version_output, version_unit)
    if version_input==0:
      conv_1 = Conv(data, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_1") # 224/112
    else:
      conv_1 = Conv(data, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_1") # 224/112
    conv_2_dw = Conv(conv_1, num_group=32, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_2_dw") # 112/112
    conv_2 = Conv(conv_2_dw, num_filter=64, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_2") # 112/112
    conv_3_dw = Conv(conv_2, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_3_dw") # 112/56
    conv_3 = Conv(conv_3_dw, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_3") # 56/56
    conv_4_dw = Conv(conv_3, num_group=128, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_4_dw") # 56/56
    conv_4 = Conv(conv_4_dw, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_4") # 56/56
    conv_5_dw = Conv(conv_4, num_group=128, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_5_dw") # 56/28
    conv_5 = Conv(conv_5_dw, num_filter=256, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_5") # 28/28
    conv_6_dw = Conv(conv_5, num_group=256, num_filter=256, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_6_dw") # 28/28
    conv_6 = Conv(conv_6_dw, num_filter=256, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6") # 28/28
    conv_7_dw = Conv(conv_6, num_group=256, num_filter=256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_7_dw") # 28/14
    conv_7 = Conv(conv_7_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_7") # 14/14

    conv_8_dw = Conv(conv_7, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_8_dw") # 14/14
    conv_8 = Conv(conv_8_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_8") # 14/14
    conv_9_dw = Conv(conv_8, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_9_dw") # 14/14
    conv_9 = Conv(conv_9_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_9") # 14/14
    conv_10_dw = Conv(conv_9, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_10_dw") # 14/14
    conv_10 = Conv(conv_10_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_10") # 14/14
    conv_11_dw = Conv(conv_10, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_11_dw") # 14/14
    conv_11 = Conv(conv_11_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_11") # 14/14
    conv_12_dw = Conv(conv_11, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_12_dw") # 14/14
    conv_12 = Conv(conv_12_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_12") # 14/14

    conv_13_dw = Conv(conv_12, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_13_dw") # 14/7
    conv_13 = Conv(conv_13_dw, num_filter=1024, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_13") # 7/7
    conv_14_dw = Conv(conv_13, num_group=1024, num_filter=1024, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_14_dw") # 7/7
    conv_14 = Conv(conv_14_dw, num_filter=1024, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_14") # 7/7
    body = conv_14

    if fc_type=='E':
      body = mx.symbol.Dropout(data=body, p=0.4)
      fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
      fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, name='fc1')
    elif fc_type=='F':
      body = mx.symbol.Dropout(data=body, p=0.4)
      fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
      fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, name='fc1')
      fc1 = Act(data=fc1, act_type='relu', name='fc1_relu')
    else:
      pool = mx.sym.Pooling(data=conv_14, global_pool=True, kernel=(7, 7), stride=(1, 1), pool_type="avg", name="global_pool")
      flat = mx.sym.Flatten(data=pool, name="flatten")
      if fc_type=='A':
        fc1 = flat
      else:
        if fc_type=='G' or fc_type=='H':
          fc1 = mx.symbol.Dropout(data=flat, p=0.2)
          fc1 = mx.sym.FullyConnected(data=fc1, num_hidden=num_classes, name='pre_fc1')
          if fc_type=='G':
            return fc1
          else:
            fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, name='fc1')
            return fc1
        else:
          #B-D
          #B
          fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_classes, name='pre_fc1')
          if fc_type=='C':
            fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, name='fc1')
          elif fc_type=='D':
            fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, name='fc1')
            fc1 = Act(data=fc1, act_type='relu', name='fc1_relu')
    return fc1

