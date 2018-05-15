from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mxnet as mx
import numpy as np


ACT_BIT = 1
N = 4
use_STN = False
use_DLA = 0
DCN = 0



def Conv(**kwargs):
    #name = kwargs.get('name')
    #_weight = mx.symbol.Variable(name+'_weight')
    #_bias = mx.symbol.Variable(name+'_bias', lr_mult=2.0, wd_mult=0.0)
    #body = mx.sym.Convolution(weight = _weight, bias = _bias, **kwargs)
    body = mx.sym.Convolution(**kwargs)
    return body


def Act(data, act_type, name):
    if act_type=='prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    else:
      body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body

def lin(data, num_filter, workspace, name, binarize, dcn):
  bn_mom = 0.9
  bit = 1
  if not binarize:
    if not dcn:
        conv1 = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn')
        act1 = Act(data=bn1, act_type='relu', name=name + '_relu')
        return act1
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn')
        act1 = Act(data=bn1, act_type='relu', name=name + '_relu')
        conv1_offset = mx.symbol.Convolution(name=name+'_conv_offset', data = act1,
                num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
        conv1 = mx.contrib.symbol.DeformableConvolution(name=name+"_conv", data=act1, offset=conv1_offset,
                num_filter=num_filter, pad=(1,1), kernel=(3, 3), num_deformable_group=1, stride=(1, 1), dilate=(1, 1), no_bias=False)
        #conv1 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
        #                              no_bias=False, workspace=workspace, name=name + '_conv')
        return conv1
  else:
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn')
    act1 = Act(data=bn1, act_type='relu', name=name + '_relu')
    conv1 = mx.sym.QConvolution_v1(data=act1, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                               no_bias=True, workspace=workspace, name=name + '_conv', act_bit=ACT_BIT, weight_bit=bit)
    conv1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
    return conv1

def lin2(data, num_filter, workspace, name):
    bn_mom = 0.9
    conv1 = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                  no_bias=True, workspace=workspace, name=name + '_conv')
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn')
    act1 = Act(data=bn1, act_type='relu', name=name + '_relu')
    return act1

def lin3(data, num_filter, workspace, name, k, g=1, d=1):
    bn_mom = 0.9
    #bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn')
    #act1 = Act(data=bn1, act_type='relu', name=name + '_relu')
    #conv1 = Conv(data=act1, num_filter=num_filter, kernel=(k,k), stride=(1,1), pad=((k-1)//2,(k-1)//2), num_group=g,
    #                              no_bias=True, workspace=workspace, name=name + '_conv')
    #return conv1
    if k!=3:
        conv1 = Conv(data=data, num_filter=num_filter, kernel=(k,k), stride=(1,1), pad=((k-1)//2,(k-1)//2), num_group=g,
                                      no_bias=True, workspace=workspace, name=name + '_conv')
    else:
        conv1 = Conv(data=data, num_filter=num_filter, kernel=(k,k), stride=(1,1), pad=(d,d), num_group=g, dilate=(d, d),
                                      no_bias=True, workspace=workspace, name=name + '_conv')
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn')
    act1 = Act(data=bn1, act_type='relu', name=name + '_relu')
    ret = act1
    #if g>1 and k==3 and d==1:
    #    body = Conv(data=ret, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), num_group=1,
    #                                  no_bias=True, workspace=workspace, name=name + '_conv2')
    #    body = mx.sym.BatchNorm(data=body, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
    #    body = Act(data=body, act_type='relu', name=name + '_relu2')
    #    ret = body
    return ret

def lin3_red(data, num_filter, workspace, name, k, g=1):
    bn_mom = 0.9
    conv1 = Conv(data=data, num_filter=num_filter, kernel=(3,3), stride=(k,k), pad=(1,1), num_group=g,
        no_bias=True, workspace=workspace, name=name + '_conv', attr={'lr_mult':'1'})
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn', attr={'lr_mult':'1'})
    act1 = Act(data=bn1, act_type='sigmoid', name=name + '_relu')
    ret = act1
    #if g>1 and k==3 and d==1:
    #    body = Conv(data=ret, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), num_group=1,
    #                                  no_bias=True, workspace=workspace, name=name + '_conv2')
    #    body = mx.sym.BatchNorm(data=body, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
    #    body = Act(data=body, act_type='relu', name=name + '_relu2')
    #    ret = body
    return ret

class RES:
    def __init__(self, data, nFilters, nModules, n, workspace, name, dilate, group):
        self.data = data
        self.nFilters = nFilters
        self.nModules = nModules
        self.n = n
        self.workspace = workspace
        self.name = name
        self.dilate = dilate
        self.group = group
        self.sym_map = {}

    def get_output(self, w, h):
        key = (w, h)
        if key in self.sym_map:
            return self.sym_map[key]
        ret = None
        if h==self.n:
            if w==self.n:
                ret = (self.data, self.nFilters)
            else:
                x = self.get_output(w+1, h)
                f = int(x[1]*0.5)
                if w!=self.n-1:
                    body = lin3(x[0], f, self.workspace, "%s_w%d_h%d_1"%(self.name, w, h), 3, self.group, 1)
                else:
                    body = lin3(x[0], f, self.workspace, "%s_w%d_h%d_1"%(self.name, w, h), 3, self.group, self.dilate)
                ret = (body,f)
        else:
            x = self.get_output(w+1, h+1)
            y = self.get_output(w, h+1)
            if h%2==1 and h!=w:
                xbody = lin3(x[0], x[1], self.workspace, "%s_w%d_h%d_2"%(self.name, w, h), 3, x[1])
                #xbody = xbody+x[0]
            else:
                xbody = x[0]
            #xbody = x[0]
            #xbody = lin3(x[0], x[1], self.workspace, "%s_w%d_h%d_2"%(self.name, w, h), 3, x[1])
            if w==0:
                ybody = lin3(y[0], y[1], self.workspace, "%s_w%d_h%d_3"%(self.name, w, h), 3, self.group)
            else:
                ybody = y[0]
            ybody = mx.sym.concat(y[0], ybody, dim=1)
            body = mx.sym.add_n(xbody,ybody, name="%s_w%d_h%d_add"%(self.name, w, h))
            body = body/2
            ret = (body, x[1])
        self.sym_map[key] = ret
        return ret

    def get(self):
        return self.get_output(1, 1)[0]

def residual_unit_a(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    bit = 1
    #print('in unit2')
    # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    if not binarize:
      act1 = Act(data=bn1, act_type='relu', name=name + '_relu1')
      conv1 = Conv(data=act1, num_filter=int(num_filter*0.5), kernel=(1,1), stride=(1,1), pad=(0,0),
                                 no_bias=True, workspace=workspace, name=name + '_conv1')
    else:
      act1 = mx.sym.QActivation(data=bn1, act_bit=ACT_BIT, name=name + '_relu1', backward_only=True)
      conv1 = mx.sym.QConvolution(data=act1, num_filter=int(num_filter*0.5), kernel=(1,1), stride=(1,1), pad=(0,0),
                                 no_bias=True, workspace=workspace, name=name + '_conv1', act_bit=ACT_BIT, weight_bit=bit)
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    if not binarize:
      act2 = Act(data=bn2, act_type='relu', name=name + '_relu2')
      conv2 = Conv(data=act2, num_filter=int(num_filter*0.5), kernel=(3,3), stride=(1,1), pad=(1,1),
                                 no_bias=True, workspace=workspace, name=name + '_conv2')
    else:
      act2 = mx.sym.QActivation(data=bn2, act_bit=ACT_BIT, name=name + '_relu2', backward_only=True)
      conv2 = mx.sym.QConvolution(data=act2, num_filter=int(num_filter*0.5), kernel=(3,3), stride=(1,1), pad=(1,1),
                                 no_bias=True, workspace=workspace, name=name + '_conv2', act_bit=ACT_BIT, weight_bit=bit)
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    if not binarize:
      act3 = Act(data=bn3, act_type='relu', name=name + '_relu3')
      conv3 = Conv(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                 workspace=workspace, name=name + '_conv3')
    else:
      act3 = mx.sym.QActivation(data=bn3, act_bit=ACT_BIT, name=name + '_relu3', backward_only=True)
      conv3 = mx.sym.QConvolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
                                 no_bias=True, workspace=workspace, name=name + '_conv3', act_bit=ACT_BIT, weight_bit=bit)
    #if binarize:
    #  conv3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4')
    if dim_match:
        shortcut = data
    else:
        if not binarize:
          shortcut = Conv(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name+'_sc')
        else:
          shortcut = mx.sym.QConvolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, pad=(0,0),
                               no_bias=True, workspace=workspace, name=name + '_sc', act_bit=ACT_BIT, weight_bit=bit)
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return conv3 + shortcut


def residual_unit_g(data, num_filter, stride, dim_match, name, binarize, dcn, dilation, **kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    bit = 1
    #print('in unit2')
    # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    if not binarize:
      act1 = Act(data=bn1, act_type='relu', name=name + '_relu1')
      if not dcn:
          conv1 = Conv(data=act1, num_filter=int(num_filter*0.5), kernel=(3,3), stride=(1,1), pad=(dilation,dilation), dilate=(dilation,dilation),
                                     no_bias=True, workspace=workspace, name=name + '_conv1')
      else:
          conv1_offset = mx.symbol.Convolution(name=name+'_conv1_offset', data = act1,
                num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
          conv1 = mx.contrib.symbol.DeformableConvolution(name=name+'_conv1', data=act1, offset=conv1_offset,
                num_filter=int(num_filter*0.5), pad=(1,1), kernel=(3, 3), num_deformable_group=1, stride=(1, 1), dilate=(1, 1), no_bias=True)
    else:
      act1 = mx.sym.QActivation(data=bn1, act_bit=ACT_BIT, name=name + '_relu1', backward_only=True)
      conv1 = mx.sym.QConvolution_v1(data=act1, num_filter=int(num_filter*0.5), kernel=(3,3), stride=(1,1), pad=(1,1),
                                 no_bias=True, workspace=workspace, name=name + '_conv1', act_bit=ACT_BIT, weight_bit=bit)
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    if not binarize:
      act2 = Act(data=bn2, act_type='relu', name=name + '_relu2')
      if not dcn:
          conv2 = Conv(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(dilation,dilation), dilate=(dilation,dilation),
                                     no_bias=True, workspace=workspace, name=name + '_conv2')
      else:
          conv2_offset = mx.symbol.Convolution(name=name+'_conv2_offset', data = act2,
                num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
          conv2 = mx.contrib.symbol.DeformableConvolution(name=name+'_conv2', data=act2, offset=conv2_offset,
                num_filter=int(num_filter*0.25), pad=(1,1), kernel=(3, 3), num_deformable_group=1, stride=(1, 1), dilate=(1, 1), no_bias=True)
    else:
      act2 = mx.sym.QActivation(data=bn2, act_bit=ACT_BIT, name=name + '_relu2', backward_only=True)
      conv2 = mx.sym.QConvolution_v1(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                 no_bias=True, workspace=workspace, name=name + '_conv2', act_bit=ACT_BIT, weight_bit=bit)
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    if not binarize:
      act3 = Act(data=bn3, act_type='relu', name=name + '_relu3')
      if not dcn:
          conv3 = Conv(data=act3, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(dilation,dilation), dilate=(dilation,dilation), 
                  no_bias=True, workspace=workspace, name=name + '_conv3')
      else:
          conv3_offset = mx.symbol.Convolution(name=name+'_conv3_offset', data = act3,
                num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
          conv3 = mx.contrib.symbol.DeformableConvolution(name=name+'_conv3', data=act3, offset=conv3_offset,
                num_filter=int(num_filter*0.25), pad=(1,1), kernel=(3, 3), num_deformable_group=1, stride=(1, 1), dilate=(1, 1), no_bias=True)
    else:
      act3 = mx.sym.QActivation(data=bn3, act_bit=ACT_BIT, name=name + '_relu3', backward_only=True)
      conv3 = mx.sym.QConvolution_v1(data=act3, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(1,1),
                                 no_bias=True, workspace=workspace, name=name + '_conv3', act_bit=ACT_BIT, weight_bit=bit)
    conv4 = mx.symbol.Concat(*[conv1, conv2, conv3])
    if binarize:
      conv4 = mx.sym.BatchNorm(data=conv4, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4')
    if dim_match:
        shortcut = data
    else:
        if not binarize:
          shortcut = Conv(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                          workspace=workspace, name=name+'_sc')
        else:
          #assert(False)
          shortcut = mx.sym.QConvolution_v1(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, pad=(0,0),
                               no_bias=True, workspace=workspace, name=name + '_sc', act_bit=ACT_BIT, weight_bit=bit)
          shortcut = mx.sym.BatchNorm(data=shortcut, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return conv4 + shortcut
    #return bn4 + shortcut
    #return act4 + shortcut

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

def block35(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}):
    M = 1.0
    tower_conv = ConvFactory(net, int(input_num_channels*0.25*M), (1, 1))
    tower_conv1_0 = ConvFactory(net, int(input_num_channels*0.25*M), (1, 1))
    tower_conv1_1 = ConvFactory(tower_conv1_0, int(input_num_channels*0.25*M), (3, 3), pad=(1, 1))
    tower_conv2_0 = ConvFactory(net, int(input_num_channels*0.25*M), (1, 1))
    tower_conv2_1 = ConvFactory(tower_conv2_0, int(input_num_channels*0.375*M), (3, 3), pad=(1, 1))
    tower_conv2_2 = ConvFactory(tower_conv2_1, int(input_num_channels*0.5*M), (3, 3), pad=(1, 1))
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

def residual_unit_i(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    assert not binarize
    if stride[0]>1 or not dim_match:
        return residual_unit_a(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs)
    conv4 = block35(data, num_filter)
    return conv4

def residual_unit_cab(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    if stride[0]>1 or not dim_match:
        return residual_unit_g(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs)
    res = RES(data, num_filter, 1, 4, workspace, name, dilate, 1)
    return res.get()

def residual_unit(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs):
  #binarize = False
  #binarize = BINARIZE
  return residual_unit_cab(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs)

def hourglass(data, nFilters, nModules, n, workspace, name, binarize, dcn):
  s = 2
  _dcn = False
  up1 = data
  for i in xrange(nModules):
    up1 = residual_unit(up1, nFilters, (1,1), True, "%s_up1_%d"%(name,i), binarize, _dcn, 1)
  low1 = mx.sym.Pooling(data=data, kernel=(s, s), stride=(s,s), pad=(0,0), pool_type='max')
  for i in xrange(nModules):
    low1 = residual_unit(low1, nFilters, (1,1), True, "%s_low1_%d"%(name,i), binarize, _dcn, 1)
  if n>1:
    low2 = hourglass(low1, nFilters, nModules, n-1, workspace, "%s_%d"%(name, n-1), binarize, dcn)
  else:
    low2 = low1
    for i in xrange(nModules):
      low2 = residual_unit(low2, nFilters, (1,1), True, "%s_low2_%d"%(name,i), binarize, _dcn, 1) #TODO
      #low2 = residual_unit(low2, nFilters, (1,1), True, "%s_low2_%d"%(name,i), False) #TODO
  low3 = low2
  for i in xrange(nModules):
    low3 = residual_unit(low3, nFilters, (1,1), True, "%s_low3_%d"%(name,i), binarize, _dcn, 1)
  up2 = mx.symbol.Deconvolution(data=low3, num_filter=nFilters, kernel=(s,s), 
      stride=(s, s),
      num_group=nFilters, no_bias=True, name='%s_upsampling_%s'%(name,n),
      attr={'lr_mult': '0.0', 'wd_mult': '0.0'}, workspace=workspace)
  return mx.symbol.add_n(up1, up2)

def hourglass2(data, nFilters, nModules, n, workspace, name, binarize, dcn):
  s = 2
  _dcn = dcn
  if DCN and n==N:
      _dcn = True
  _dcn = False
  up1 = data
  dilate = 2**(4-n)
  for i in xrange(nModules):
    up1 = residual_unit(up1, nFilters, (1,1), True, "%s_up1_%d"%(name,i), binarize, _dcn, dilate)
  #low1 = mx.sym.Pooling(data=data, kernel=(s, s), stride=(s,s), pad=(0,0), pool_type='max')
  low1 = data
  for i in xrange(nModules):
    low1 = residual_unit(low1, nFilters, (1,1), True, "%s_low1_%d"%(name,i), binarize, _dcn, dilate)
  if n>1:
    low2 = hourglass2(low1, nFilters, nModules, n-1, workspace, "%s_%d"%(name, n-1), binarize, dcn)
  else:
    low2 = low1
    for i in xrange(nModules):
      low2 = residual_unit(low2, nFilters, (1,1), True, "%s_low2_%d"%(name,i), binarize, _dcn, dilate) #TODO
      #low2 = residual_unit(low2, nFilters, (1,1), True, "%s_low2_%d"%(name,i), False) #TODO
  low3 = low2
  for i in xrange(nModules):
    low3 = residual_unit(low3, nFilters, (1,1), True, "%s_low3_%d"%(name,i), binarize, _dcn, dilate)
  up2 = low3
  #up2 = mx.symbol.Deconvolution(data=low3, num_filter=nFilters, kernel=(s,s), 
  #    stride=(s, s),
  #    num_group=nFilters, no_bias=True, name='%s_upsampling_%s'%(name,n),
  #    attr={'lr_mult': '0.0', 'wd_mult': '0.0'}, workspace=workspace)
  return mx.symbol.add_n(up1, up2)


class DLA:
    def __init__(self, data, nFilters, nModules, n, workspace, name):
        self.data = data
        self.nFilters = nFilters
        self.nModules = nModules
        self.n = n
        self.workspace = workspace
        self.name = name
        self.sym_map = {}


    def residual_unit(self, data, name, dilate=1, group=1):
        res = RES(data, self.nFilters, self.nModules, 4, self.workspace, name, dilate, group)
        return res.get()
        #body = data
        #for i in xrange(self.nModules):
        #    body = residual_unit(body, self.nFilters, (1,1), True, name, False, False, 1)
        #return body

    def get_output(self, w, h):
        #print(w,h)
        assert w>=1 and w<=N+1
        assert h>=1 and h<=N+1
        s = 2
        bn_mom = 0.9
        key = (w,h)
        if key in self.sym_map:
            return self.sym_map[key]
        ret = None
        if h==self.n:
            if w==self.n:
                ret = self.data,64
            #elif w==1:
            #    x = self.get_output(w+1, h)
            #    body = self.residual_unit(x[0], "%s_w%d_h%d_1"%(self.name, w, h))
            #    body = self.residual_unit(body, "%s_w%d_h%d_2"%(self.name, w, h), 2)
            #    ret = body,x[1]
            else:
                x = self.get_output(w+1, h)
                body = self.residual_unit(x[0], "%s_w%d_h%d_1"%(self.name, w, h))
                body = mx.sym.Pooling(data=body, kernel=(s, s), stride=(s,s), pad=(0,0), pool_type='max')
                body = self.residual_unit(body, "%s_w%d_h%d_2"%(self.name, w, h))
                ret = body, x[1]//2
        else:
            x = self.get_output(w+1, h+1)
            y = self.get_output(w, h+1)
            #xbody = Conv(data=x, num_filter=self.nFilters, kernel=(3,3), stride=(1,1), pad=(1,1),
            #                          no_bias=True, workspace=self.workspace, name="%s_w%d_h%d_x_conv"%(self.name, w, h))
            #xbody = mx.sym.BatchNorm(data=xbody, fix_gamma=False, momentum=bn_mom, eps=2e-5, name="%s_w%d_h%d_x_bn"%(self.name, w, h))
            #xbody = Act(data=xbody, act_type='relu', name="%s_w%d_h%d_x_act"%(self.name, w, h))

            HC = False

            if use_DLA<10:
                if h%2==1 and h!=w:
                    xbody = lin3(x[0], self.nFilters, self.workspace, "%s_w%d_h%d_x"%(self.name, w, h), 3, self.nFilters, 1)
                    HC = True
                    #xbody = x[0]
                else:
                    xbody = x[0]
            else:
                xbody = lin3(x[0], self.nFilters, self.workspace, "%s_w%d_h%d_x"%(self.name, w, h), 3, 1, 1)
            #xbody = x[0]
            if x[1]//y[1]==2:
                if w>1:
                    ybody = mx.symbol.Deconvolution(data=y[0], num_filter=self.nFilters, kernel=(s,s), 
                      stride=(s, s),
                      name='%s_upsampling_w%d_h%d'%(self.name,w, h),
                      attr={'lr_mult': '1.0'}, workspace=self.workspace)
                    ybody = mx.sym.BatchNorm(data=ybody, fix_gamma=False, momentum=bn_mom, eps=2e-5, name="%s_w%d_h%d_y_bn"%(self.name, w, h))
                    ybody = Act(data=ybody, act_type='relu', name="%s_w%d_h%d_y_act"%(self.name, w, h))
                    #ybody = Conv(data=ybody, num_filter=self.nFilters, kernel=(3,3), stride=(1,1), pad=(1,1),
                    #                      no_bias=True, name="%s_w%d_h%d_y_conv2"%(self.name, w, h), workspace=self.workspace)
                    #ybody = mx.sym.BatchNorm(data=ybody, fix_gamma=False, momentum=bn_mom, eps=2e-5, name="%s_w%d_h%d_y_bn2"%(self.name, w, h))
                    #ybody = Act(data=ybody, act_type='relu', name="%s_w%d_h%d_y_act2"%(self.name, w, h))
                else:
                    if h>=1:
                        ybody = mx.symbol.Deconvolution(data=y[0], num_filter=self.nFilters, kernel=(s,s), 
                          stride=(s, s),
                          num_group=self.nFilters, no_bias=True, name='%s_upsampling_w%d_h%d'%(self.name,w, h),
                          attr={'lr_mult': '0.0', 'wd_mult': '0.0'}, workspace=self.workspace)
                        #ybody = mx.sym.BatchNorm(data=ybody, fix_gamma=False, momentum=bn_mom, eps=2e-5, name="%s_w%d_h%d_y_bn"%(self.name, w, h))
                        import math
                        #ybody = Act(data=ybody, act_type='relu', name="%s_w%d_h%d_y_act"%(self.name, w, h))
                        ybody = self.residual_unit(ybody, "%s_w%d_h%d_4"%(self.name, w, h))
                    else:
                        ybody = mx.symbol.Deconvolution(data=y[0], num_filter=self.nFilters, kernel=(s,s), 
                          stride=(s, s),
                          name='%s_upsampling_w%d_h%d'%(self.name,w, h),
                          attr={'lr_mult': '1.0'}, workspace=self.workspace)
                        ybody = mx.sym.BatchNorm(data=ybody, fix_gamma=False, momentum=bn_mom, eps=2e-5, name="%s_w%d_h%d_y_bn"%(self.name, w, h))
                        ybody = Act(data=ybody, act_type='relu', name="%s_w%d_h%d_y_act"%(self.name, w, h))
                        ybody = Conv(data=ybody, num_filter=self.nFilters, kernel=(3,3), stride=(1,1), pad=(1,1),
                                              no_bias=True, name="%s_w%d_h%d_y_conv2"%(self.name, w, h), workspace=self.workspace)
                        ybody = mx.sym.BatchNorm(data=ybody, fix_gamma=False, momentum=bn_mom, eps=2e-5, name="%s_w%d_h%d_y_bn2"%(self.name, w, h))
                        ybody = Act(data=ybody, act_type='relu', name="%s_w%d_h%d_y_act2"%(self.name, w, h))
            else:
                ybody = self.residual_unit(y[0], "%s_w%d_h%d_5"%(self.name, w, h))
            #if not HC:
            if use_DLA<10:
                if use_DLA>1 and h==3 and w==2:
                  z = self.get_output(w+1, h)
                  zbody = z[0]
                  #zbody = lin3_red(zbody, self.nFilters, self.workspace, "%s_w%d_h%d_z"%(self.name, w, h), 2, self.nFilters)
                  #zbody = mx.sym.Pooling(data=zbody, kernel=(s, s), stride=(s,s), pad=(0,0), pool_type='avg')
                  zbody = mx.sym.Pooling(data=zbody, kernel=(z[1], z[1]), stride=(z[1],z[1]), pad=(0,0), pool_type='avg')
                  #zbody = mx.sym.Activation(data = zbody, act_type='sigmoid')

                  #body = zbody+ybody
                  #body = body/2
                  body = xbody+ybody
                  body = body/2
                  #body = body*zbody
                  body = mx.sym.broadcast_mul(body, zbody)
                  #body = mx.sym.add_n(*[xbody, ybody, zbody])
                  #body = body/3
                else:
                  body = xbody+ybody
                  body = body/2
            else:
                if use_DLA==12 and h!=w:
                    zbody = self.get_output(w+1, h)[0]
                    zbody = lin3_red(zbody, self.nFilters, self.workspace, "%s_w%d_h%d_z"%(self.name, w, h), 2, 1)
                    body = mx.sym.add_n(*[xbody, ybody, zbody])
                    body = body/3
                else:
                    body = xbody+ybody
                    body = body/2
            ret = body, x[1]

        assert ret is not None
        self.sym_map[key] = ret
        return ret

    def get(self):
        return self.get_output(1, 1)[0]


def l2_loss(x, y):
  loss = x-y
  loss = loss*loss
  loss = mx.symbol.mean(loss)
  return loss

def ce_loss(x, y):
  body = mx.sym.exp(x)
  sums = mx.sym.sum(body, axis=[2,3], keepdims=True)
  body = mx.sym.broadcast_div(body, sums)
  loss = mx.sym.log(body)
  loss = loss*y*-1.0
  loss = mx.symbol.mean(loss)
  return loss

def get_symbol(num_classes, **kwargs):
    global use_DLA
    global N
    global DCN
    mirror_set = [
              (22,23),
              (21,24),
              (20,25),
              (19,26),
              (18,27),
              (40,43),
              (39,44),
              (38,45),
              (37,46),
              (42,47),
              (41,48),
              (33,35),
              (32,36),
              (51,53),
              (50,54),
              (62,64),
              (61,65),
              (49,55),
              (49,55),
              (68,66),
              (60,56),
              (59,57),
              (1,17),
              (2,16),
              (3,15),
              (4,14),
              (5,13),
              (6,12),
              (7,11),
              (8,10),
          ]
    mirror_map = {}
    for mm in mirror_set:
      mirror_map[mm[0]-1] = mm[1]-1
      mirror_map[mm[1]-1] = mm[0]-1
    sFilters = 64
    mFilters = 128
    nFilters = 256

    nModules = 1
    nStacks = 2
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    binarize = kwargs.get('binarize', False)
    input_size = kwargs.get('input_size', 128)
    label_size = kwargs.get('label_size', 64)
    use_coherent = kwargs.get('use_coherent', 0)
    use_DLA = kwargs.get('use_dla', 0)
    N = kwargs.get('use_N', 4)
    DCN = kwargs.get('use_DCN', 0)
    per_batch_size = kwargs.get('per_batch_size', 0)
    print('binarize', binarize)
    print('use_coherent', use_coherent)
    print('use_DLA', use_DLA)
    print('use_N', N)
    print('use_DCN', DCN)
    print('per_batch_size', per_batch_size)
    assert(label_size==64 or label_size==32)
    assert(input_size==128 or input_size==256)
    D = input_size // label_size
    print(input_size, label_size, D)
    dcn = False
    kwargs = {}
    data = mx.sym.Variable(name='data')
    data = data-127.5
    data = data*0.0078125
    gt_label = mx.symbol.Variable(name='softmax_label')
    losses = []
    closses = []
    if use_coherent:
        M = mx.sym.Variable(name="coherent_label")
        #gt_label2 = mx.sym.Variable(name="softmax_label2")
        coherent_weight = 0.0001
    ref_label = gt_label
    if use_STN:
        lr_mult = '0.00001'
        loc_net = Conv(data=data, num_filter=sFilters, kernel=(7, 7), stride=(2,2), pad=(3, 3),
            no_bias=True, name="stn_conv0", workspace=workspace, attr={'lr_mult': lr_mult})
        loc_net = mx.sym.BatchNorm(data=loc_net, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='stn_bn0', attr={'lr_mult': lr_mult})
        loc_net = Act(data=loc_net, act_type='relu', name='stn_relu0')
        loc_net = Conv(data=loc_net, num_filter=mFilters, kernel=(3,3), stride=(1,1), pad=(1,1),
            no_bias=True, name="stn_conv1", workspace=workspace, attr={'lr_mult': lr_mult})
        loc_net = mx.sym.BatchNorm(data=loc_net, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='stn_bn1', attr={'lr_mult': lr_mult})
        loc_net = Act(data=loc_net, act_type='relu', name='stn_relu1')
        loc_net = mx.sym.Pooling(data=loc_net, kernel=(2, 2), stride=(2,2), pad=(0,0), pool_type='max')

        loc_net = mx.sym.FullyConnected(data=loc_net, num_hidden=int(nFilters*0.5), name='loc_net_half', attr={'lr_mult': lr_mult})
        loc_net = mx.sym.Activation(data=loc_net, act_type='tanh', name='loc_net_act')
        #loc_net = mx.sym.Activation(data=loc_net, act_type='relu', name='loc_net_act')
        #loc_theta  = mx.sym.FullyConnected(data=loc_net, num_hidden=6, name='loc_theta', attr={'lr_mult': lr_mult})
        #loc_theta = mx.sym.Activation(data=loc_theta, act_type='tanh', name='loc_theta_tanh')
        loc_theta  = mx.sym.FullyConnected(data=loc_net, num_hidden=1, name='loc_theta', attr={'lr_mult': lr_mult})
        loc_theta = mx.sym.Activation(data=loc_theta, act_type='tanh', name='loc_theta_tanh')
        loc_theta = loc_theta*0.5
        sin_t = mx.sym.sin(loc_theta)
        m_sin_t = sin_t*-1.0
        cos_t = mx.sym.cos(loc_theta)
        zero_t = mx.sym.zeros_like(loc_theta)
        loc_theta = mx.sym.concat(*[cos_t, m_sin_t, zero_t, sin_t, cos_t, zero_t], dim=1)
        data = mx.sym.SpatialTransformer(data = data, loc = loc_theta, target_shape=(input_size,input_size), transform_type="affine", sampler_type="bilinear")
        ref_label = mx.sym.SpatialTransformer(data = ref_label, loc = loc_theta, target_shape=(label_size,label_size), transform_type="affine", sampler_type="bilinear")
    #data = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn_data')
    if D==4:
      body = Conv(data=data, num_filter=sFilters, kernel=(7, 7), stride=(2,2), pad=(3, 3),
                              no_bias=True, name="conv0", workspace=workspace)
    else:
      body = Conv(data=data, num_filter=sFilters, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                              no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = Act(data=body, act_type='relu', name='relu0')

    body = residual_unit(body, mFilters, (1,1), sFilters==mFilters, 'res0', False, dcn, 1, **kwargs)
    #body = residual_unit(body, nFilters, (1,1), False, 'res0', binarize, **kwargs)

    body = mx.sym.Pooling(data=body, kernel=(2, 2), stride=(2,2), pad=(0,0), pool_type='max')

    body = residual_unit(body, mFilters, (1,1), True, 'res1', False, dcn, 1, **kwargs) #TODO
    #body = residual_unit(body, nFilters, (1,1), True, 'res1', binarize, **kwargs) #TODO
    body = residual_unit(body, nFilters, (1,1), mFilters==nFilters, 'res2', binarize, dcn, 1, **kwargs) #binarize=True?
    #body = residual_unit(body, nFilters, (1,1), False, 'res2', False, **kwargs) #binarize=True?

    use_lin = True
    heatmap = None

    for i in xrange(nStacks):
      shortcut = body
      if use_DLA>0:
        dla = DLA(body, nFilters, nModules, N+1, workspace, 'dla%d'%(i))
        body = dla.get()
      else:
        body = hourglass(body, nFilters, nModules, N, workspace, 'stack%d_hg'%(i), binarize, dcn)
      for j in xrange(nModules):
        body = residual_unit(body, nFilters, (1,1), True, 'stack%d_unit%d'%(i,j), binarize, dcn, 1, **kwargs)
      if use_lin:
        _dcn = True if DCN>=2 else False
        ll = lin(body, nFilters, workspace, name='stack%d_ll'%(i), binarize = False, dcn = _dcn) #TODO
        #ll = lin(body, nFilters, workspace, name='stack%d_ll'%(i), binarize = binarize)
      else:
        ll = body
      _name = "heatmap%d"%(i)
      if i==nStacks-1:
        _name = "heatmap"

      _dcn = True if DCN>=2 else False
      if not _dcn:
          out = Conv(data=ll, num_filter=num_classes, kernel=(1, 1), stride=(1,1), pad=(0,0),
                                    name=_name, workspace=workspace)
      else:
          out_offset = mx.symbol.Convolution(name=_name+'_offset', data = ll,
                num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
          out = mx.contrib.symbol.DeformableConvolution(name=_name, data=ll, offset=out_offset,
                num_filter=num_classes, pad=(1,1), kernel=(3, 3), num_deformable_group=1, stride=(1, 1), dilate=(1, 1), no_bias=False)
          #out = Conv(data=ll, num_filter=num_classes, kernel=(3,3), stride=(1,1), pad=(1,1),
          #                          name=_name, workspace=workspace)
      if i==nStacks-1:
          heatmap = out
      #outs.append(out)
      if use_coherent>0:
          #px = mx.sym.slice_axis(out, axis=0, begin=0, end=b)
          #py = mx.sym.slice_axis(ref_label, axis=0, begin=0, end=b)
          px = out
          py = ref_label
          gloss = ce_loss(px, py)
          gloss = gloss/nStacks
          losses.append(gloss)

          b = per_batch_size//2
          ux = mx.sym.slice_axis(out, axis=0, begin=0, end=b)
          dx = mx.sym.slice_axis(out, axis=0, begin=b, end=b*2)
          if use_coherent==1:
            ux = mx.sym.flip(ux, axis=3)
            ux_list = [None]*68
            for k in xrange(68):
              if k in mirror_map:
                vk = mirror_map[k]
                #print('k', k, vk)
                ux_list[vk] = mx.sym.slice_axis(ux, axis=1, begin=k, end=k+1)
              else:
                ux_list[k] = mx.sym.slice_axis(ux, axis=1, begin=k, end=k+1)
            ux = mx.sym.concat(*ux_list, dim=1)
            #dx = mx.sym.slice_axis(ref_label, axis=0, begin=b, end=b*2)
            #closs = ce_loss(ux, dx)
            closs = l2_loss(ux, dx)
            closs = closs/nStacks
            closses.append(closs)
          else:
            m = mx.sym.slice_axis(M, axis=0, begin=0, end=b)
            ux = mx.sym.SpatialTransformer(data=ux, loc=m, target_shape=(label_size, label_size), transform_type='affine', sampler_type='bilinear')
            closs = l2_loss(ux, dx)
            closs = closs/nStacks
            closses.append(closs)

      else:
          loss = ce_loss(out, ref_label)
          loss = loss/nStacks
          losses.append(loss)

      if i<nStacks-1:
        if use_lin:
          ll2 = Conv(data=ll, num_filter=nFilters, kernel=(1, 1), stride=(1,1), pad=(0,0),
                                    name="stack%d_ll2"%(i), workspace=workspace)
        else:
          ll2 = body
        out2 = Conv(data=out, num_filter=nFilters, kernel=(1, 1), stride=(1,1), pad=(0,0),
                                  name="stack%d_out2"%(i), workspace=workspace)
        body = mx.symbol.add_n(shortcut, ll2, out2)
        _dcn = True if (DCN==1 or DCN==3) else False
        if _dcn:
            _name = "stack%d_out3" % (i)
            out3_offset = mx.symbol.Convolution(name=_name+'_offset', data = body,
                  num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
            out3 = mx.contrib.symbol.DeformableConvolution(name=_name, data=body, offset=out3_offset,
                  num_filter=nFilters, pad=(1,1), kernel=(3, 3), num_deformable_group=1, stride=(1, 1), dilate=(1, 1), no_bias=False)
            body = out3
        #elif use_STN:
        #  loc_net = dla.get2()
        #  #loc_net = mx.sym.Pooling(data=loc_net, global_pool=True, kernel=(7, 7), pool_type='avg', name='loc_net_pool')
        #  loc_net = mx.sym.FullyConnected(data=loc_net, num_hidden=int(nFilters*0.5), name='loc_net_half', attr={'lr_mult': '0.0001'})
        #  loc_net = mx.sym.Activation(data=loc_net, act_type='tanh', name='loc_net_act')
        #  #loc_net = mx.sym.Activation(data=loc_net, act_type='relu', name='loc_net_act')
        #  loc_theta  = mx.sym.FullyConnected(data=loc_net, num_hidden=6, name='loc_theta', attr={'lr_mult': '0.0001'})
        #  loc_theta = mx.sym.Activation(data=loc_theta, act_type='tanh', name='loc_theta_tanh')
        #  body = mx.sym.SpatialTransformer(data = body, loc = loc_theta, target_shape=(label_size,label_size), transform_type="affine", sampler_type="bilinear")
        #  ref_label = mx.sym.SpatialTransformer(data = gt_label, loc = loc_theta, target_shape=(label_size,label_size), transform_type="affine", sampler_type="bilinear")

    pred = mx.symbol.BlockGrad(heatmap)
    loss = mx.symbol.add_n(*losses)

    loss = mx.symbol.MakeLoss(loss)
    syms = [loss]
    if len(closses)>0:
        closs = mx.symbol.add_n(*closses)
        closs = mx.symbol.MakeLoss(closs, grad_scale = coherent_weight)
        syms.append(closs)
    #syms.append(mx.symbol.BlockGrad(M))
    #syms.append(mx.symbol.BlockGrad(px))
    #syms.append(mx.symbol.BlockGrad(qx))
    #syms.append(mx.symbol.BlockGrad(m))
    #syms.append(mx.symbol.BlockGrad(closs))
    if use_coherent>1:
        syms.append(mx.symbol.BlockGrad(gt_label))
    if use_coherent>0:
        syms.append(mx.symbol.BlockGrad(M))
    syms.append(pred)
    sym = mx.symbol.Group( syms )
    return sym

def init_weights(sym, data_shape_dict):
    print('in hg2')
    arg_name = sym.list_arguments()
    aux_name = sym.list_auxiliary_states()
    arg_shape, _, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(arg_name, arg_shape))
    aux_shape_dict = dict(zip(aux_name, aux_shape))
    #print(aux_shape)
    #print(aux_params)
    #print(arg_shape_dict)
    arg_params = {}
    aux_params = {}
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
      elif k.find('upsampling')>=0:
        print('initializing upsampling_weight', k)
        arg_params[k] = mx.nd.zeros(shape=arg_shape_dict[k])
        init = mx.init.Initializer()
        init._init_bilinear(k, arg_params[k])
      elif k.find('loc_theta')>=0:
        print('initializing STN', k, v)
        if k.endswith('_weight'):
          arg_params[k] = mx.nd.zeros(shape=v)
        elif k.endswith('_bias'):
          #val = np.array([4.0, 0.0, 0.0, 0.0, 4.0, 0.0], dtype=np.float32)
          #val = np.array([0.0], dtype=np.float32)
          #arg_params[k] = mx.nd.array(val)
          arg_params[k] = mx.random.normal(0, 0.01, shape=v)
    return arg_params, aux_params

