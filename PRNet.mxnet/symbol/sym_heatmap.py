from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mxnet as mx
import numpy as np
from config import config


ACT_BIT = 1
bn_mom = 0.9
workspace = 256
memonger = False



def Conv(**kwargs):
    body = mx.sym.Convolution(**kwargs)
    return body

def Act(data, act_type, name):
    if act_type=='prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    else:
      body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body

#def lin(data, num_filter, workspace, name, binarize, dcn):
#  bit = 1
#  if not binarize:
#    if not dcn:
#        conv1 = Conv(data=data, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
#                                      no_bias=True, workspace=workspace, name=name + '_conv')
#        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn')
#        act1 = Act(data=bn1, act_type='relu', name=name + '_relu')
#        return act1
#    else:
#        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn')
#        act1 = Act(data=bn1, act_type='relu', name=name + '_relu')
#        conv1_offset = mx.symbol.Convolution(name=name+'_conv_offset', data = act1,
#                num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
#        conv1 = mx.contrib.symbol.DeformableConvolution(name=name+"_conv", data=act1, offset=conv1_offset,
#                num_filter=num_filter, pad=(1,1), kernel=(3, 3), num_deformable_group=1, stride=(1, 1), dilate=(1, 1), no_bias=False)
#        #conv1 = Conv(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
#        #                              no_bias=False, workspace=workspace, name=name + '_conv')
#        return conv1
#  else:
#    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn')
#    act1 = Act(data=bn1, act_type='relu', name=name + '_relu')
#    conv1 = mx.sym.QConvolution_v1(data=act1, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),
#                               no_bias=True, workspace=workspace, name=name + '_conv', act_bit=ACT_BIT, weight_bit=bit)
#    conv1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
#    return conv1

def lin3(data, num_filter, workspace, name, k, g=1, d=1):
    if k!=3:
        conv1 = Conv(data=data, num_filter=num_filter, kernel=(k,k), stride=(1,1), pad=((k-1)//2,(k-1)//2), num_group=g,
                                      no_bias=True, workspace=workspace, name=name + '_conv')
    else:
        conv1 = Conv(data=data, num_filter=num_filter, kernel=(k,k), stride=(1,1), pad=(d,d), num_group=g, dilate=(d, d),
                                      no_bias=True, workspace=workspace, name=name + '_conv')
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn')
    act1 = Act(data=bn1, act_type='relu', name=name + '_relu')
    ret = act1
    return ret

def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type="relu", mirror_attr={}, with_act=True, dcn=False, name=''):
    if not dcn:
      conv = mx.symbol.Convolution(
          data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True, workspace=workspace, name=name+'_conv')
    else:
        conv_offset = mx.symbol.Convolution(name=name+'_conv_offset', data = data,
                num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
        conv = mx.contrib.symbol.DeformableConvolution(name=name+"_conv", data=data, offset=conv_offset,
                num_filter=num_filter, pad=(1,1), kernel=(3,3), num_deformable_group=1, stride=stride, dilate=(1, 1), no_bias=False)
    bn = mx.symbol.BatchNorm(data=conv, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name+'_bn')
    if with_act:
      act = Act(bn, act_type, name=name+'_relu')
      #act = mx.symbol.Activation(
      #    data=bn, act_type=act_type, attr=mirror_attr, name=name+'_relu')
      return act
    else:
      return bn

class CAB:
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

def conv_resnet(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs):
    bit = 1
    #print('in unit2')
    # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = Act(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = Conv(data=act1, num_filter=int(num_filter*0.5), kernel=(1,1), stride=(1,1), pad=(0,0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act2 = Act(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = Conv(data=act2, num_filter=int(num_filter*0.5), kernel=(3,3), stride=(1,1), pad=(1,1),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    act3 = Act(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = Conv(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    #if binarize:
    #  conv3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4')
    if dim_match:
        shortcut = data
    else:
        shortcut = Conv(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                        workspace=workspace, name=name+'_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return conv3 + shortcut


def conv_prnet(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs):
    #print('in unit2')
    # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = Act(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = Conv(data=act1, num_filter=int(num_filter*0.5), kernel=(1,1), stride=(1,1), pad=(0,0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act2 = Act(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = Conv(data=act2, num_filter=int(num_filter*0.5), kernel=(3,3), stride=(1,1), pad=(1,1),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    act3 = Act(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = Conv(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = Conv(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                        workspace=workspace, name=name+'_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return conv3 + shortcut

def conv_hpm(data, num_filter, stride, dim_match, name, binarize, dcn, dilation, **kwargs):
    bit = 1
    #print('in unit2')
    # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = Act(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = Conv(data=act1, num_filter=int(num_filter*0.5), kernel=(3,3), stride=(1,1), pad=(dilation,dilation), dilate=(dilation,dilation),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act2 = Act(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = Conv(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(dilation,dilation), dilate=(dilation,dilation),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    act3 = Act(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = Conv(data=act3, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1), pad=(dilation,dilation), dilate=(dilation,dilation), 
            no_bias=True, workspace=workspace, name=name + '_conv3')
    conv4 = mx.symbol.Concat(*[conv1, conv2, conv3])
    if dim_match:
        shortcut = data
    else:
        shortcut = Conv(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                        workspace=workspace, name=name+'_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return conv4 + shortcut


def block17(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}, name=''):
    tower_conv = ConvFactory(net, 192, (1, 1), name=name+'_conv')
    tower_conv1_0 = ConvFactory(net, 129, (1, 1), name=name+'_conv1_0')
    tower_conv1_1 = ConvFactory(tower_conv1_0, 160, (1, 7), pad=(1, 2), name=name+'_conv1_1')
    tower_conv1_2 = ConvFactory(tower_conv1_1, 192, (7, 1), pad=(2, 1), name=name+'_conv1_2')
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
    tower_out = ConvFactory(
        tower_mixed, input_num_channels, (1, 1), with_act=False, name=name+'_conv_out')
    net = net+scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net

def block35(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}, name=''):
    M = 1.0
    tower_conv = ConvFactory(net, int(input_num_channels*0.25*M), (1, 1), name=name+'_conv')
    tower_conv1_0 = ConvFactory(net, int(input_num_channels*0.25*M), (1, 1), name=name+'_conv1_0')
    tower_conv1_1 = ConvFactory(tower_conv1_0, int(input_num_channels*0.25*M), (3, 3), pad=(1, 1), name=name+'_conv1_1')
    tower_conv2_0 = ConvFactory(net, int(input_num_channels*0.25*M), (1, 1), name=name+'_conv2_0')
    tower_conv2_1 = ConvFactory(tower_conv2_0, int(input_num_channels*0.375*M), (3, 3), pad=(1, 1), name=name+'_conv2_1')
    tower_conv2_2 = ConvFactory(tower_conv2_1, int(input_num_channels*0.5*M), (3, 3), pad=(1, 1), name=name+'_conv2_2')
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_1, tower_conv2_2])
    tower_out = ConvFactory(
        tower_mixed, input_num_channels, (1, 1), with_act=False, name=name+'_conv_out')

    net = net+scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net

def conv_inception(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs):
    assert not binarize
    if stride[0]>1 or not dim_match:
        return conv_resnet(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs)
    conv4 = block35(data, num_filter, name=name+'_block35')
    return conv4

def conv_cab(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs):
    if stride[0]>1 or not dim_match:
        return conv_hpm(data, num_filter, stride, dim_match, name, binarize, dcn, dilate, **kwargs)
    cab = CAB(data, num_filter, 1, 4, workspace, name, dilate, 1)
    return cab.get()

def conv_block(data, num_filter, stride, dim_match, name, binarize, dcn, dilate):
  if config.net_block=='resnet':
    return conv_resnet(data, num_filter, stride, dim_match, name, binarize, dcn, dilate)
  elif config.net_block=='inception':
    return conv_inception(data, num_filter, stride, dim_match, name, binarize, dcn, dilate)
  elif config.net_block=='hpm':
    return conv_hpm(data, num_filter, stride, dim_match, name, binarize, dcn, dilate)
  elif config.net_block=='cab':
    return conv_cab(data, num_filter, stride, dim_match, name, binarize, dcn, dilate)
  elif config.net_block=='prnet':
    return conv_prnet(data, num_filter, stride, dim_match, name, binarize, dcn, dilate)

def hourglass(data, nFilters, nModules, n, workspace, name, binarize, dcn):
  s = 2
  _dcn = False
  up1 = data
  for i in xrange(nModules):
    up1 = conv_block(up1, nFilters, (1,1), True, "%s_up1_%d"%(name,i), binarize, _dcn, 1)
  low1 = mx.sym.Pooling(data=data, kernel=(s, s), stride=(s,s), pad=(0,0), pool_type='max')
  #low1 = ConvFactory(data, nFilters, (4,4), stride=(2,2), pad=(1,1), name=name+'_conv')
  #low1 = ConvFactory(data, nFilters, (3,3), stride=(2,2), pad=(1,1), name=name+'_conv')
  #low1 = ConvFactory(up1, nFilters, (3,3), stride=(2,2), pad=(1,1), name=name+'_conv')
  for i in xrange(nModules):
    low1 = conv_block(low1, nFilters, (1,1), True, "%s_low1_%d"%(name,i), binarize, _dcn, 1)
  if n>1:
    low2 = hourglass(low1, nFilters, nModules, n-1, workspace, "%s_%d"%(name, n-1), binarize, dcn)
  else:
    low2 = low1
    for i in xrange(nModules):
      low2 = conv_block(low2, nFilters, (1,1), True, "%s_low2_%d"%(name,i), binarize, _dcn, 1) #TODO
  low3 = low2
  for i in xrange(nModules):
    low3 = conv_block(low3, nFilters, (1,1), True, "%s_low3_%d"%(name,i), binarize, _dcn, 1)
  up2 = mx.symbol.UpSampling(low3, scale=s, sample_type='nearest', workspace=512, name='%s_upsampling_%s'%(name,n), num_args=1)
  #up2 = mx.symbol.UpSampling(low3, scale=s, sample_type='bilinear', num_filter=nFilters, workspace=512, name='%s_upsampling_%s'%(name,n), num_args=1)
  #up2 = mx.symbol.Deconvolution(data=low3, num_filter=nFilters, kernel=(s*2,s*2), 
  #  stride=(s, s), pad=(s//2, s//2),
  #  name='%s_upsampling_%s'%(name,n),
  #  attr={'lr_mult': '0.1'})
  #return mx.symbol.add_n(up1, up2)
  return up2


def prnet_loss(pred, gt_label, mask_label):
  loss = pred - gt_label
  #loss = mx.symbol.smooth_l1(loss, scalar=3.0)
  loss = mx.symbol.abs(loss)
  loss = mx.symbol.broadcast_mul(loss, mask_label)
  #loss = mx.symbol.mean(loss, axis=0)
  #loss = loss*loss
  #loss = mx.symbol.mean(loss)
  return loss

def ce_loss(x, y):
  #loss = mx.sym.SoftmaxOutput(data = x, label = y, normalization='valid', multi_output=True)
  x_max = mx.sym.max(x, axis=[2,3], keepdims=True)
  x = mx.sym.broadcast_minus(x, x_max)
  body = mx.sym.exp(x)
  sums = mx.sym.sum(body, axis=[2,3], keepdims=True)
  body = mx.sym.broadcast_div(body, sums)
  loss = mx.sym.log(body)
  loss = loss*y*-1.0
  #loss = mx.symbol.mean(loss, axis=[1,2,3])
  loss = mx.symbol.mean(loss)
  return loss

def get_symbol(num_classes):
    m = config.multiplier
    sFilters = max(int(64*m), 16)
    mFilters = max(int(128*m), 32)
    nFilters = int(256*m)

    nModules = config.net_modules
    nStacks = config.net_stacks
    binarize = config.net_binarize
    input_size = config.input_img_size
    label_size = config.output_label_size
    use_STA = config.net_sta
    N = config.net_n
    DCN = config.net_dcn
    per_batch_size = config.per_batch_size
    print('binarize', binarize)
    print('use_STA', use_STA)
    print('use_N', N)
    print('use_DCN', DCN)
    print('per_batch_size', per_batch_size)
    #assert(label_size==64 or label_size==32)
    #assert(input_size==128 or input_size==256)
    D = input_size // label_size
    print(input_size, label_size, D)
    data = mx.sym.Variable(name='data')
    data = data-127.5
    data = data*0.0078125
    gt_label = mx.symbol.Variable(name='softmax_label')
    mask_label = mx.symbol.Variable(name='mask_label')
    losses = []
    closses = []
    #body = Conv(data=data, num_filter=sFilters, kernel=(3, 3), stride=(1,1), pad=(1, 1),
    #                        no_bias=True, name="conv0", workspace=workspace)
    body = Conv(data=data, num_filter=sFilters, kernel=(7,7), stride=(2,2), pad=(3,3),
                            no_bias=True, name="conv0", workspace=workspace)
    #body = Conv(data=data, num_filter=sFilters, kernel=(4,4), stride=(2,2), pad=(1,1),
    #                        no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = Act(data=body, act_type='relu', name='relu0')

    dcn = False
    body = conv_block(body, mFilters, (1,1), sFilters==mFilters, 'res0', False, dcn, 1)
    body = mx.sym.Pooling(data=body, kernel=(2, 2), stride=(2,2), pad=(0,0), pool_type='max')

    #body = Conv(data=body, num_filter=mFilters, kernel=(4,4), stride=(2,2), pad=(1,1),
    #                        no_bias=True, name="conv1", workspace=workspace)
    #body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    #body = Act(data=body, act_type='relu', name='relu1')

    #body = conv_block(body, mFilters, (1,1), True, 'res1', False, dcn, 1) #TODO
    body = conv_block(body, nFilters, (1,1), mFilters==nFilters, 'res2', binarize, dcn, 1) #binarize=True?

    heatmap = None
    outs = []

    body = hourglass(body, nFilters, nModules, config.net_n, workspace, 'stack0_hg', binarize, dcn)
    for j in xrange(nModules):
      body = conv_block(body, nFilters, (1,1), True, 'stack0_unit%d'%(j), binarize, dcn, 1)
    _dcn = False
    ll = ConvFactory(body, nFilters, (1,1), dcn = _dcn, name='stack0_ll')
    _name = 'heatmap'
    pred = Conv(data=ll, num_filter=num_classes, kernel=(1, 1), stride=(1,1), pad=(0,0),
                              name=_name, workspace=workspace)
    loss = prnet_loss(pred, gt_label, mask_label)
    outs.append(mx.sym.MakeLoss(loss))


    pred = mx.symbol.BlockGrad(pred)
    #loss = mx.symbol.add_n(*losses)
    #loss = mx.symbol.MakeLoss(loss)
    #syms = [loss]
    outs.append(pred)
    sym = mx.symbol.Group( outs )
    return sym

def init_weights(sym, data_shape_dict):
    #print('in hg')
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
    return arg_params, aux_params

