import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag
import numpy as np
from rcnn.config import config
from rcnn.PY_OP import rpn_fpn_ohem3
from symbol_common import get_sym_train

def conv_only(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), bias_wd_mult=0.0):
  weight = mx.symbol.Variable(name="{}_weight".format(name),   
      init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
  bias = mx.symbol.Variable(name="{}_bias".format(name),   
      init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(bias_wd_mult)})
  conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
      stride=stride, num_filter=num_filter, name="{}".format(name), weight = weight, bias=bias)
  return conv

def conv_act_layer_dw(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", bias_wd_mult=0.0):
    assert kernel[0]==3
    weight = mx.symbol.Variable(name="{}_weight".format(name),   
        init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
    bias = mx.symbol.Variable(name="{}_bias".format(name),   
        init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(bias_wd_mult)})
    conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, num_group=num_filter, name="{}".format(name), weight=weight, bias=bias)
    conv = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + '_bn')
    if len(act_type)>0:
      relu = mx.symbol.Activation(data=conv, act_type=act_type, \
          name="{}_{}".format(name, act_type))
    else:
      relu = conv
    return relu

def conv_act_layer(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", bias_wd_mult=0.0, separable=False, filter_in = -1):

    separable = False
    if separable:
      assert kernel[0]==3
    if not separable:
      weight = mx.symbol.Variable(name="{}_weight".format(name),   
          init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
      bias = mx.symbol.Variable(name="{}_bias".format(name),   
          init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(bias_wd_mult)})
      conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
          stride=stride, num_filter=num_filter, name="{}".format(name), weight=weight, bias=bias)
      conv = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + '_bn')
    else:
      if filter_in<0:
        filter_in = num_filter
      conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
          stride=stride, num_filter=filter_in, num_group=filter_in, name="{}_sep".format(name))
      conv = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + '_sep_bn')
      conv = mx.symbol.Activation(data=conv, act_type='relu', \
          name="{}_sep_bn_relu".format(name))
      conv = mx.symbol.Convolution(data=conv, kernel=(1,1), pad=(0,0), \
          stride=(1,1), num_filter=num_filter, name="{}".format(name))
      conv = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + '_bn')
    if len(act_type)>0:
      relu = mx.symbol.Activation(data=conv, act_type=act_type, \
          name="{}_{}".format(name, act_type))
    else:
      relu = conv
    return relu

def ssh_context_module(body, num_filter, filter_in, name):
  conv_dimred = conv_act_layer(body, name+'_conv1',
      num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', separable=True, filter_in = filter_in)
  conv5x5 = conv_act_layer(conv_dimred, name+'_conv2',
      num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='', separable=True)
  conv7x7_1 = conv_act_layer(conv_dimred, name+'_conv3_1',
      num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', separable=True)
  conv7x7 = conv_act_layer(conv7x7_1, name+'_conv3_2',
      num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='', separable=True)
  return (conv5x5, conv7x7)

def conv_deformable(net, num_filter, num_group=1, act_type='relu',name=''):
  f = num_group*18
  conv_offset = mx.symbol.Convolution(name=name+'_conv_offset', data = net,
                      num_filter=f, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
  net = mx.contrib.symbol.DeformableConvolution(name=name+"_conv", data=net, offset=conv_offset,
                      num_filter=num_filter, pad=(1,1), kernel=(3, 3), num_deformable_group=num_group, stride=(1, 1), no_bias=False)
  net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + '_bn')
  if len(act_type)>0:
    net = mx.symbol.Activation(data=net, act_type=act_type, name=name+'_act')
  return net

def ssh_detection_module(body, num_filter, filter_in, name):
  conv3x3 = conv_act_layer(body, name+'_conv1',
      num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='', separable=True, filter_in=filter_in)
  conv5x5, conv7x7 = ssh_context_module(body, num_filter//2, filter_in, name+'_context')
  ret = mx.sym.concat(*[conv3x3, conv5x5, conv7x7], dim=1, name = name+'_concat')
  ret = mx.symbol.Activation(data=ret, act_type='relu', name=name+'_concat_relu')
  if config.USE_DCN==1:
    ret = conv_deformable(ret, num_filter = num_filter*2, name = name+'_concat_dcn')
  elif config.USE_DCN==2:
    ret = conv_deformable2(ret, num_filter = num_filter*2, name = name+'_concat_dcn')
  return ret


def get_resnet_conv(data, sym):
    all_layers = sym.get_internals()
    isize = 640
    _, out_shape, _ = all_layers.infer_shape(data = (1,3,isize,isize))
    last_entry = None
    c1 = None
    c2 = None
    c3 = None
    #print(len(all_layers), len(out_shape))
    #print(all_layers.__class__)
    outputs = all_layers.list_outputs()
    #print(outputs.__class__, len(outputs))
    count = len(outputs)
    for i in range(count):
      name = outputs[i]
      shape = out_shape[i]
      if not name.endswith('_output'):
        continue
      if len(shape)!=4:
        continue
      print(name, shape)
      if c1 is None and shape[2]==isize//16:
        cname = last_entry[0]
        print('c1', last_entry)
        c1 = all_layers[cname]
      if c2 is None and shape[2]==isize//32:
        cname = last_entry[0]
        print('c2', last_entry)
        c2 = all_layers[cname]
      if shape[2]==isize//32:
        c3 = all_layers[name]
        print('c3', name, shape)

      last_entry = (name, shape)

    c1_filter = -1
    c2_filter = -1
    c3_filter = -1

    F1 = 256
    F2 = 256
    _bwm = 1.0
    if config.NET_MODE==0:
      c1_lateral = conv_act_layer(c1, 'ssh_m1_red_conv',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c2_lateral = conv_act_layer(c2, 'ssh_m2_red_conv',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      #conv5_128_up = mx.symbol.Deconvolution(data=conv5_128, num_filter=F2, kernel=(4,4),  stride=(2, 2), pad=(1,1),
      #    num_group = F2, no_bias = True, attr={'__lr_mult__': '0.0', '__wd_mult__': '0.0'},
      #    name='ssh_m2_red_upsampling')
      c2_up = mx.symbol.UpSampling(c2_lateral, scale=2, sample_type='nearest', workspace=512, name='ssh_m2_red_up', num_args=1)
      #conv4_128 = mx.symbol.Crop(*[conv4_128, conv5_128_up])
      c2_up = mx.symbol.Crop(*[c2_up, c1_lateral])

      c1 = c1_lateral+c2_up

      c1 = conv_act_layer(c1, 'ssh_m1_conv',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      m1 = ssh_detection_module(c1, F2, F2, 'ssh_m1_det')
      m2 = ssh_detection_module(c2, F1, c2_filter, 'ssh_m2_det')
      m3 = ssh_detection_module(c3, F1, c3_filter, 'ssh_m3_det')
    elif config.NET_MODE==1:
      c3_lateral = conv_act_layer(c3, 'ssh_c3_lateral',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c3_up = mx.symbol.UpSampling(c3_lateral, scale=2, sample_type='nearest', workspace=512, name='ssh_c3_up', num_args=1)
      c2_lateral = conv_act_layer(c2, 'ssh_c2_lateral',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c3_up = mx.symbol.Crop(*[c3_up, c2_lateral])
      c2 = c2_lateral+c3_up
      c2 = conv_act_layer(c2, 'ssh_c2_aggr',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c1_lateral = conv_act_layer(c1, 'ssh_m1_red_conv',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c2_up = mx.symbol.UpSampling(c2, scale=2, sample_type='nearest', workspace=512, name='ssh_m2_red_up', num_args=1)
      #conv4_128 = mx.symbol.Crop(*[conv4_128, conv5_128_up])
      c2_up = mx.symbol.Crop(*[c2_up, c1_lateral])

      c1 = c1_lateral+c2_up

      c1 = conv_act_layer(c1, 'ssh_m1_conv',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      m1 = ssh_detection_module(c1, F2, F2, 'ssh_m1_det')
      m2 = ssh_detection_module(c2, F1, c2_filter, 'ssh_m2_det')
      m3 = ssh_detection_module(c3, F1, c3_filter, 'ssh_m3_det')
    elif config.NET_MODE==2:
      n1 = ssh_detection_module(c1, F2, F2, 'ssh_n1_det')
      n2 = ssh_detection_module(c2, F1, c2_filter, 'ssh_n2_det')
      n3 = ssh_detection_module(c3, F1, c3_filter, 'ssh_n3_det')
      c3 = conv_act_layer(c3, 'ssh_c3_lateral',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c3_up = mx.symbol.UpSampling(c3, scale=2, sample_type='nearest', workspace=512, name='ssh_c3_up', num_args=1)
      c2_lateral = conv_act_layer(c2, 'ssh_c2_lateral',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c3_up = mx.symbol.Crop(*[c3_up, c2_lateral])
      c2 = c2_lateral+c3_up
      c2 = conv_act_layer(c2, 'ssh_c2_aggr',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c1_lateral = conv_act_layer(c1, 'ssh_m1_red_conv',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c2_up = mx.symbol.UpSampling(c2, scale=2, sample_type='nearest', workspace=512, name='ssh_m2_red_up', num_args=1)
      #conv4_128 = mx.symbol.Crop(*[conv4_128, conv5_128_up])
      c2_up = mx.symbol.Crop(*[c2_up, c1_lateral])
      c1 = c1_lateral+c2_up
      c1 = conv_act_layer(c1, 'ssh_c1_aggr',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)

      m1 = ssh_detection_module(c1, F2, F2, 'ssh_m1_det')
      m2 = ssh_detection_module(c2, F1, c2_filter, 'ssh_m2_det')
      m3 = ssh_detection_module(c3, F1, c3_filter, 'ssh_m3_det')
    elif config.NET_MODE==3:
      #c3 = conv_act_layer(c3, 'ssh_c3_lateral',
      #    F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c3 = ssh_detection_module(c3, F2//2, c3_filter, 'ssh_c3_lateral')
      c3_up = mx.symbol.UpSampling(c3, scale=2, sample_type='nearest', workspace=512, name='ssh_c3_up', num_args=1)
      #c2_lateral = conv_act_layer(c2, 'ssh_c2_lateral',
      #    F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c2_lateral = ssh_detection_module(c2, F2//2, c2_filter, 'ssh_c2_lateral')
      c3_up = mx.symbol.Crop(*[c3_up, c2_lateral])
      c2 = c2_lateral+c3_up
      c2 = conv_act_layer(c2, 'ssh_c2_aggr',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      #c1_lateral = conv_act_layer(c1, 'ssh_m1_red_conv',
      #    F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c1_lateral = ssh_detection_module(c1, F2//2, c1_filter, 'ssh_c1_lateral')
      c2_up = mx.symbol.UpSampling(c2, scale=2, sample_type='nearest', workspace=512, name='ssh_m2_red_up', num_args=1)
      #conv4_128 = mx.symbol.Crop(*[conv4_128, conv5_128_up])
      c2_up = mx.symbol.Crop(*[c2_up, c1_lateral])
      c1 = c1_lateral+c2_up
      c1 = conv_act_layer(c1, 'ssh_c1_aggr',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)

      m1 = ssh_detection_module(c1, F2, F2, 'ssh_m1_det')
      m2 = ssh_detection_module(c2, F1, c2_filter, 'ssh_m2_det')
      m3 = ssh_detection_module(c3, F1, c3_filter, 'ssh_m3_det')
    elif config.NET_MODE==4:
      c3 = conv_act_layer(c3, 'ssh_c3_lateral',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c3_up = mx.symbol.UpSampling(c3, scale=2, sample_type='nearest', workspace=512, name='ssh_c3_up', num_args=1)
      c2_lateral = conv_act_layer(c2, 'ssh_c2_lateral',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c3_up = mx.symbol.Crop(*[c3_up, c2_lateral])
      c2 = c2_lateral+c3_up
      c2 = conv_act_layer(c2, 'ssh_c2_aggr',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c1_lateral = conv_act_layer(c1, 'ssh_m1_red_conv',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c2_up = mx.symbol.UpSampling(c2, scale=2, sample_type='nearest', workspace=512, name='ssh_m2_red_up', num_args=1)
      #conv4_128 = mx.symbol.Crop(*[conv4_128, conv5_128_up])
      c2_up = mx.symbol.Crop(*[c2_up, c1_lateral])
      c1 = c1_lateral+c2_up
      c1 = conv_act_layer(c1, 'ssh_c1_aggr',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)

      m1 = ssh_detection_module(c1, F2//2, F2, 'ssh_m1_det')
      m2 = ssh_detection_module(c2, F1//2, c2_filter, 'ssh_m2_det')
      m3 = ssh_detection_module(c3, F1//2, c3_filter, 'ssh_m3_det')
    elif config.NET_MODE==5:
      c3 = conv_act_layer_dw(c3, 'ssh_c3_lateral_m',
          F2, kernel=(3,3), pad=(1,1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c3 = conv_act_layer(c3, 'ssh_c3_lateral',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c3_up = mx.symbol.UpSampling(c3, scale=2, sample_type='nearest', workspace=512, name='ssh_c3_up', num_args=1)
      c2 = conv_act_layer_dw(c2, 'ssh_c2_lateral_m',
          F2, kernel=(3,3), pad=(1,1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c2_lateral = conv_act_layer(c2, 'ssh_c2_lateral',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c3_up = mx.symbol.Crop(*[c3_up, c2_lateral])
      c2 = c2_lateral+c3_up
      c2 = conv_act_layer(c2, 'ssh_c2_aggr',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c1 = conv_act_layer_dw(c1, 'ssh_c1_lateral_m',
          F2, kernel=(3,3), pad=(1,1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c1_lateral = conv_act_layer(c1, 'ssh_m1_red_conv',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c2_up = mx.symbol.UpSampling(c2, scale=2, sample_type='nearest', workspace=512, name='ssh_m2_red_up', num_args=1)
      #conv4_128 = mx.symbol.Crop(*[conv4_128, conv5_128_up])
      c2_up = mx.symbol.Crop(*[c2_up, c1_lateral])
      c1 = c1_lateral+c2_up
      c1 = conv_act_layer(c1, 'ssh_c1_aggr',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)

      m1 = ssh_detection_module(c1, F2, F2, 'ssh_m1_det')
      m2 = ssh_detection_module(c2, F1, c2_filter, 'ssh_m2_det')
      m3 = ssh_detection_module(c3, F1, c3_filter, 'ssh_m3_det')

    return {8: m1, 16:m2, 32: m3}, {8: n1, 16:n2, 32: n3}

def get_out(conv_fpn_feat, prefix, stride, landmark=False, lr_mult=1.0):
    A = config.NUM_ANCHORS
    bbox_pred_len = 4
    landmark_pred_len = 10
    if config.USE_BLUR:
      bbox_pred_len = 5
    if config.USE_OCCLUSION:
      landmark_pred_len = 15
    ret_group = []
    num_anchors = config.RPN_ANCHOR_CFG[str(stride)]['NUM_ANCHORS']
    label = mx.symbol.Variable(name='%s_label_stride%d'%(prefix,stride))
    bbox_target = mx.symbol.Variable(name='%s_bbox_target_stride%d'%(prefix,stride))
    bbox_weight = mx.symbol.Variable(name='%s_bbox_weight_stride%d'%(prefix,stride))
    if landmark:
      landmark_target = mx.symbol.Variable(name='%s_landmark_target_stride%d'%(prefix,stride))
      landmark_weight = mx.symbol.Variable(name='%s_landmark_weight_stride%d'%(prefix,stride))
    rpn_relu = conv_fpn_feat[stride]
    maxout_stat = 0
    if config.USE_MAXOUT>=1 and stride==config.RPN_FEAT_STRIDE[-1]:
      maxout_stat = 1
    if config.USE_MAXOUT>=2 and stride!=config.RPN_FEAT_STRIDE[-1]:
      maxout_stat = 2

    if maxout_stat==0:
      rpn_cls_score = conv_only(rpn_relu, '%s_rpn_cls_score_stride%d'%(prefix, stride), 2*num_anchors,
          kernel=(1,1), pad=(0,0), stride=(1, 1))
    elif maxout_stat==1:
      cls_list = []
      for a in range(num_anchors):
        rpn_cls_score_bg = conv_only(rpn_relu, '%s_rpn_cls_score_stride%d_anchor%d_bg'%(prefix,stride,a), 3,
            kernel=(1,1), pad=(0,0), stride=(1, 1))
        rpn_cls_score_bg = mx.sym.max(rpn_cls_score_bg, axis=1, keepdims=True)
        cls_list.append(rpn_cls_score_bg)
        rpn_cls_score_fg = conv_only(rpn_relu, '%s_rpn_cls_score_stride%d_anchor%d_fg'%(prefix,stride,a), 1,
            kernel=(1,1), pad=(0,0), stride=(1, 1))
        cls_list.append(rpn_cls_score_fg)
      rpn_cls_score = mx.sym.concat(*cls_list, dim=1, name='%s_rpn_cls_score_stride%d'%(prefix,stride))
    else:
      cls_list = []
      for a in range(num_anchors):
        rpn_cls_score_bg = conv_only(rpn_relu, '%s_rpn_cls_score_stride%d_anchor%d_bg'%(prefix,stride,a), 1,
            kernel=(1,1), pad=(0,0), stride=(1, 1))
        cls_list.append(rpn_cls_score_bg)
        rpn_cls_score_fg = conv_only(rpn_relu, '%s_rpn_cls_score_stride%d_anchor%d_fg'%(prefix,stride,a), 3,
            kernel=(1,1), pad=(0,0), stride=(1, 1))
        rpn_cls_score_fg = mx.sym.max(rpn_cls_score_fg, axis=1, keepdims=True)
        cls_list.append(rpn_cls_score_fg)
      rpn_cls_score = mx.sym.concat(*cls_list, dim=1, name='%s_rpn_cls_score_stride%d'%(prefix,stride))

    rpn_bbox_pred = conv_only(rpn_relu, '%s_rpn_bbox_pred_stride%d'%(prefix,stride), bbox_pred_len*num_anchors,
        kernel=(1,1), pad=(0,0), stride=(1, 1))

    # prepare rpn data
    if not config.FBN:
      rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                shape=(0, 2, -1),
                                                name="%s_rpn_cls_score_reshape_stride%s" % (prefix,stride))
    else:
      rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                shape=(0, 2, -1),
                                                name="%s_rpn_cls_score_reshape_stride%s_pre" % (prefix,stride))
      rpn_cls_score_reshape = mx.symbol.BatchNorm(rpn_cls_score_reshape, fix_gamma=True, eps=2e-5, name="%s_rpn_cls_score_reshape_stride%s"%(prefix, stride))

    rpn_bbox_pred_reshape = mx.symbol.Reshape(data=rpn_bbox_pred,
                                              shape=(0, 0, -1),
                                              name="%s_rpn_bbox_pred_reshape_stride%s" % (prefix,stride))
    if landmark:
      rpn_landmark_pred = conv_only(rpn_relu, '%s_rpn_landmark_pred_stride%d'%(prefix,stride), landmark_pred_len*num_anchors,
          kernel=(1,1), pad=(0,0), stride=(1, 1))
      rpn_landmark_pred_reshape = mx.symbol.Reshape(data=rpn_landmark_pred,
                                              shape=(0, 0, -1),
                                              name="%s_rpn_landmark_pred_reshape_stride%s" % (prefix,stride))

    if config.TRAIN.RPN_ENABLE_OHEM>=2:
      label, anchor_weight = mx.sym.Custom(op_type='rpn_fpn_ohem3', stride=int(stride), network=config.network, dataset=config.dataset, prefix=prefix, cls_score=rpn_cls_score_reshape, labels = label)

      _bbox_weight = mx.sym.tile(anchor_weight, (1,1,bbox_pred_len))
      _bbox_weight = _bbox_weight.reshape((0, -1, A * bbox_pred_len)).transpose((0,2,1))
      bbox_weight = mx.sym.elemwise_mul(bbox_weight, _bbox_weight, name='%s_bbox_weight_mul_stride%s'%(prefix,stride))

      if landmark:
        _landmark_weight = mx.sym.tile(anchor_weight, (1,1,landmark_pred_len))
        _landmark_weight = _landmark_weight.reshape((0, -1, A * landmark_pred_len)).transpose((0,2,1))
        landmark_weight = mx.sym.elemwise_mul(landmark_weight, _landmark_weight, name='%s_landmark_weight_mul_stride%s'%(prefix,stride))
      #if not config.FACE_LANDMARK:
      #  label, bbox_weight = mx.sym.Custom(op_type='rpn_fpn_ohem', stride=int(stride), cls_score=rpn_cls_score_reshape, bbox_weight = bbox_weight , labels = label)
      #else:
      #  label, bbox_weight, landmark_weight = mx.sym.Custom(op_type='rpn_fpn_ohem2', stride=int(stride), cls_score=rpn_cls_score_reshape, bbox_weight = bbox_weight, landmark_weight=landmark_weight, labels = label)
    #cls loss
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape,
                                           label=label,
                                           multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1,
                                           grad_scale = lr_mult,
                                           name='%s_rpn_cls_prob_stride%d'%(prefix,stride))
    ret_group.append(rpn_cls_prob)
    ret_group.append(mx.sym.BlockGrad(label))

    #bbox loss
    bbox_diff = rpn_bbox_pred_reshape-bbox_target
    bbox_diff = bbox_diff * bbox_weight
    rpn_bbox_loss_ = mx.symbol.smooth_l1(name='%s_rpn_bbox_loss_stride%d_'%(prefix,stride), scalar=3.0, data=bbox_diff)
    rpn_bbox_loss = mx.sym.MakeLoss(name='%s_rpn_bbox_loss_stride%d'%(prefix,stride), data=rpn_bbox_loss_, grad_scale=1.0*lr_mult / (config.TRAIN.RPN_BATCH_SIZE))
    ret_group.append(rpn_bbox_loss)
    ret_group.append(mx.sym.BlockGrad(bbox_weight))

    #landmark loss
    if landmark:
      landmark_diff = rpn_landmark_pred_reshape-landmark_target
      landmark_diff = landmark_diff * landmark_weight
      rpn_landmark_loss_ = mx.symbol.smooth_l1(name='%s_rpn_landmark_loss_stride%d_'%(prefix,stride), scalar=3.0, data=landmark_diff)
      rpn_landmark_loss = mx.sym.MakeLoss(name='%s_rpn_landmark_loss_stride%d'%(prefix,stride), data=rpn_landmark_loss_, grad_scale=0.5*lr_mult / (config.TRAIN.RPN_BATCH_SIZE))
      ret_group.append(rpn_landmark_loss)
      ret_group.append(mx.sym.BlockGrad(landmark_weight))
    return ret_group

def get_resnet_train(sym):
    return get_sym_train(sym)
    #data = mx.symbol.Variable(name="data")
    ## shared convolutional layers
    #conv_fpn_feat, conv_fpn_feat2 = get_resnet_conv(data, sym)
    #ret_group = []
    #for stride in config.RPN_FEAT_STRIDE:
    #  ret = get_out(conv_fpn_feat, 'face', stride, config.FACE_LANDMARK, lr_mult=1.0)
    #  ret_group += ret
    #  if config.HEAD_BOX:
    #    ret = get_out(conv_fpn_feat2, 'head', stride, False, lr_mult=1.0)
    #    ret_group += ret

    #return mx.sym.Group(ret_group)


