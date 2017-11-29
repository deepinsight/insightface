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
import numpy as np
import math
#from mxnet.base import _Null


def activate(data, act_type, name):
  if act_type=='prelu':
    body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
  else:
    body = mx.sym.Activation(data=data, act_type='relu', name=name)
  return body

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False, dilate=(1,1), use_deformable=False, act='relu'):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = activate(data=bn1, act_type=act, name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = activate(data=bn2, act_type=act, name=name + '_relu2')
        if not use_deformable:
          conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=dilate,
                                     dilate = dilate, no_bias=True, workspace=workspace, name=name + '_conv2')
        else:
          conv2_offset_weight = mx.symbol.Variable(name+'_conv2_offset_weight', lr_mult=1.0)
          conv2_offset_bias = mx.symbol.Variable(name+'_conv2_offset_bias', lr_mult=2.0)
          conv2_offset = mx.symbol.Convolution(name=name+'_conv2_offset', data = act2,
                                                        num_filter=18, pad=(1, 1), kernel=(3, 3), stride=stride,
                                                        weight=conv2_offset_weight, bias=conv2_offset_bias)
          conv2 = mx.contrib.symbol.DeformableConvolution(name=name+"_conv2", data=act2, offset=conv2_offset,
                                                                   num_filter=int(num_filter*0.25), pad=dilate, kernel=(3, 3), num_deformable_group=1,
                                                                   stride=stride, dilate=dilate, no_bias=True)
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = activate(data=bn3, act_type=act, name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = activate(data=bn1, act_type=act, name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = activate(data=bn2, act_type=act, name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def resnet_conv(data, units, num_stages, filter_list, num_classes, bottle_neck=True, bn_mom=0.9, workspace=256, dtype='float32', use_deformable=True, act='relu'):
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    dtype : str
        Precision (float32 or float16)
    """
    memonger = False
    num_unit = len(units)
    assert(num_unit == num_stages)
    #usually 224
    data = mx.sym.identity(data=data, name='id')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    data_kernel = 7
    data_pad = int( (data_kernel-1)/2 )
    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(data_kernel, data_kernel), stride=(2,2), pad=(data_pad,data_pad),
                              no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = activate(data=body, act_type=act, name='relu0')
    body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    #body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(1,1), pad=(1,1), pool_type='max')

    bodies = []
    for i in range(num_stages):
        _dilate = 1
        stride = 1 if i==0 else 2
        _use_deformable = False
        if use_deformable and i>=3:
          _use_deformable = True
          #_dilate = 2

        body = residual_unit(body, filter_list[i+1], (stride, stride), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger, dilate=(_dilate,_dilate), use_deformable = _use_deformable, act=act)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger, dilate=(_dilate,_dilate), use_deformable=_use_deformable, act=act)
    return body



def get_symbol(num_classes, num_layers, conv_workspace=256, dtype='float32', **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    #image_shape = [int(l) for l in image_shape.split(',')]
    #(nchannel, height, width) = image_shape
    if num_layers >= 50:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
    else:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False
    num_stages = 4
    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 50:
        units = [3, 4, 6, 3]
    elif num_layers == 101:
        units = [3, 4, 23, 3]
    elif num_layers == 152:
        units = [3, 8, 36, 3]
    elif num_layers == 200:
        units = [3, 24, 36, 3]
    elif num_layers == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
    use_deformable = kwargs.get('use_deformable', False)
    act = kwargs.get('act', 'relu')

    data = mx.sym.Variable(name='data')
    body = resnet_conv(data = data,
                  units       = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  bottle_neck = bottle_neck,
                  workspace   = conv_workspace,
                  dtype       = dtype,
                  use_deformable = use_deformable,
                  act = act,
                  )
    bn_mom = 0.9

    #body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    #body = activate(data=body, act_type=act, name='relu1')
    #_weight = mx.symbol.Variable("fc1_weight", lr_mult=1.0)
    #_bias = mx.symbol.Variable("fc1_bias", lr_mult=2.0, wd_mult=0.0)
    #fc1 = mx.sym.FullyConnected(data=body, weight=_weight, bias=_bias, num_hidden=num_classes, name='fc1')
    #return None, None, fc1, mx.sym.SoftmaxOutput(data=fc1, name='softmax')

    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = activate(data=body, act_type=act, name='relu1')
    pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.sym.Flatten(data=pool1)
    _weight = mx.symbol.Variable("fc1_weight", lr_mult=1.0)
    _bias = mx.symbol.Variable("fc1_bias", lr_mult=1.0, wd_mult=1.0)
    fc1 = mx.sym.FullyConnected(data=flat, weight=_weight, bias=_bias, num_hidden=num_classes, name='fc1')
    return relu1, flat, fc1, mx.sym.SoftmaxOutput(data=fc1, name='softmax')





def init_weights(sym, data_shape_dict, arg_params, aux_params):
    arg_name = sym.list_arguments()
    aux_name = sym.list_auxiliary_states()
    arg_shape, aaa, aux_shape = sym.infer_shape(**data_shape_dict)
    #print(data_shape_dict)
    print(arg_name)
    print(arg_shape)
    #print(aaa)
    #print(aux_shape)
    arg_shape_dict = dict(zip(arg_name, arg_shape))
    aux_shape_dict = dict(zip(aux_name, aux_shape))
    #print(aux_shape)
    #print(aux_params)
    #print(arg_shape_dict)
    for k,v in arg_shape_dict.iteritems():
      #print(k,v)
      if k.endswith('offset_weight') or k.endswith('offset_bias'):
        print('initializing',k)
        arg_params[k] = mx.nd.zeros(shape = v)
      elif k.startswith('fc6_'):
        if k.endswith('_weight'):
          print('initializing',k)
          arg_params[k] = mx.random.normal(0, 0.01, shape=v)
        elif k.endswith('_bias'):
          print('initializing',k)
          arg_params[k] = mx.nd.zeros(shape=v)
        #pass
      elif k.startswith('upsampling') and k.endswith("_weight"):
        print('initializing',k)
        arg_params[k] = mx.nd.zeros(shape=v)
        init = mx.init.Initializer()
        init._init_bilinear(k, arg_params[k])
      else:
        if k.startswith('stage'):
          stage_id = int(k[5])
          if stage_id>4:
            rk = "stage4"+k[6:]
            if rk in arg_params:
              print('initializing', k, rk)
              if arg_shape_dict[rk]==v:
                arg_params[k] = arg_params[rk].copy()
              else:
                if k.endswith('_beta'):
                  arg_params[k] = mx.nd.zeros(shape=v)
                elif k.endswith('_gamma'):
                  arg_params[k] = mx.nd.random_uniform(shape=v)
                else:
                  arg_params[k] = mx.random.normal(0, 0.01, shape=v)
    for k,v in aux_shape_dict.iteritems():
        if k.startswith('stage'):
          stage_id = int(k[5])
          if stage_id>4:
            rk = "stage4"+k[6:]
            if rk in aux_params:
              print('initializing aux', k, rk)
              if aux_shape_dict[rk]==v:
                aux_params[k] = aux_params[rk].copy()
              else:
                if k.endswith('_moving_var'):
                  aux_params[k] = mx.nd.zeros(shape=v)
                elif k.endswith('_moving_mean'):
                  aux_params[k] = mx.nd.ones(shape=v)

