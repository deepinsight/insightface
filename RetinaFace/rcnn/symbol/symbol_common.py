import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from rcnn.config import config
from rcnn.PY_OP import rpn_fpn_ohem3, cascade_refine

def conv_only(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), bias_wd_mult=0.0, shared_weight=None, shared_bias = None):
  if shared_weight is None:
    weight = mx.symbol.Variable(name="{}_weight".format(name),   
        init=mx.init.Normal(0.01), attr={'__lr_mult__': '1.0'})
    bias = mx.symbol.Variable(name="{}_bias".format(name),   
        init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0', '__wd_mult__': str(bias_wd_mult)})
  else:
    weight = shared_weight
    bias = shared_bias
    print('reuse shared var in', name)
  conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
      stride=stride, num_filter=num_filter, name="{}".format(name), weight = weight, bias=bias)
  return conv

def conv_deformable(net, num_filter, num_group=1, act_type='relu',name=''):
  if config.USE_DCN==1:
    f = num_group*18
    conv_offset = mx.symbol.Convolution(name=name+'_conv_offset', data = net,
                        num_filter=f, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
    net = mx.contrib.symbol.DeformableConvolution(name=name+"_conv", data=net, offset=conv_offset,
                        num_filter=num_filter, pad=(1,1), kernel=(3, 3), num_deformable_group=num_group, stride=(1, 1), no_bias=False)
  else:
    print('use dcnv2 at', name)
    lr_mult = 0.1
    weight_var = mx.sym.Variable(name=name+'_conv2_offset_weight', init=mx.init.Zero(), lr_mult=lr_mult)
    bias_var = mx.sym.Variable(name=name+'_conv2_offset_bias', init=mx.init.Zero(), lr_mult=lr_mult)
    conv2_offset = mx.symbol.Convolution(name=name + '_conv2_offset', data=net, num_filter=27,
      pad=(1, 1), kernel=(3, 3), stride=(1,1), weight=weight_var, bias=bias_var, lr_mult=lr_mult)
    conv2_offset_t = mx.sym.slice_axis(conv2_offset, axis=1, begin=0, end=18)
    conv2_mask =  mx.sym.slice_axis(conv2_offset, axis=1, begin=18, end=None)
    conv2_mask = 2 * mx.sym.Activation(conv2_mask, act_type='sigmoid')

    conv2 = mx.contrib.symbol.ModulatedDeformableConvolution(name=name + '_conv2', data=net, offset=conv2_offset_t, mask=conv2_mask,
        num_filter=num_filter, pad=(1, 1), kernel=(3, 3), stride=(1,1), 
        num_deformable_group=num_group, no_bias=True)
    net = conv2
  net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, momentum=0.9, name=name + '_bn')
  if len(act_type)>0:
    net = mx.symbol.Activation(data=net, act_type=act_type, name=name+'_act')
  return net

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

    if config.USE_DCN>1 and kernel==(3,3) and pad==(1,1) and stride==(1,1) and not separable:
      return conv_deformable(from_layer, num_filter, num_group=1, act_type = act_type, name=name)

    if separable:
      assert kernel[0]>1
      assert filter_in>0
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
      num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', separable=False, filter_in = filter_in)
  conv5x5 = conv_act_layer(conv_dimred, name+'_conv2',
      num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='', separable=False)
  conv7x7_1 = conv_act_layer(conv_dimred, name+'_conv3_1',
      num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', separable=False)
  conv7x7 = conv_act_layer(conv7x7_1, name+'_conv3_2',
      num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='', separable=False)
  return (conv5x5, conv7x7)


def ssh_detection_module(body, num_filter, filter_in, name):
  assert num_filter%4==0
  conv3x3 = conv_act_layer(body, name+'_conv1',
      num_filter//2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='', separable=False, filter_in=filter_in)
  #_filter = max(num_filter//4, 16)
  _filter = num_filter//4
  conv5x5, conv7x7 = ssh_context_module(body, _filter, filter_in, name+'_context')
  ret = mx.sym.concat(*[conv3x3, conv5x5, conv7x7], dim=1, name = name+'_concat')
  ret = mx.symbol.Activation(data=ret, act_type='relu', name=name+'_concat_relu')
  out_filter = num_filter//2+_filter*2
  if config.USE_DCN>0:
    ret = conv_deformable(ret, num_filter = out_filter, name = name+'_concat_dcn')
  return ret

#def retina_context_module(body, kernel, num_filter, filter_in, name):
#  conv_dimred = conv_act_layer(body, name+'_conv0',
#      num_filter, kernel=(1,1), pad=(0,0), stride=(1, 1), act_type='relu', separable=False, filter_in = filter_in)
#  conv1 = conv_act_layer(conv_dimred, name+'_conv1',
#      num_filter*6, kernel=(1,1), pad=(0,0), stride=(1, 1), act_type='relu', separable=False, filter_in = filter_in)
#  conv2 = conv_act_layer(conv1, name+'_conv2',
#      num_filter*6, kernel=kernel, pad=((kernel[0]-1)//2, (kernel[1]-1)//2), stride=(1, 1), act_type='relu', separable=True, filter_in = num_filter*6)
#  conv3 = conv_act_layer(conv2, name+'_conv3',
#      num_filter, kernel=(1,1), pad=(0,0), stride=(1, 1), act_type='relu', separable=False)
#  conv3 = conv3 + conv_dimred
#  return conv3

def retina_detection_module(body, num_filter, filter_in, name):
  assert num_filter%4==0
  conv1 = conv_act_layer(body, name+'_conv1',
      num_filter//2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', separable=False, filter_in=filter_in)
  conv2 = conv_act_layer(conv1, name+'_conv2',
      num_filter//2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', separable=False, filter_in=num_filter//2)
  conv3 = conv_act_layer(conv2, name+'_conv3',
      num_filter//2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', separable=False, filter_in=num_filter//2)
  conv4 = conv2 + conv3
  body = mx.sym.concat(*[conv1, conv4], dim=1, name = name+'_concat')
  if config.USE_DCN>0:
    body = conv_deformable(body, num_filter = num_filter, name = name+'_concat_dcn')
  return body


def head_module(body, num_filter, filter_in, name):
  if config.HEAD_MODULE=='SSH':
    return ssh_detection_module(body, num_filter, filter_in, name)
  else:
    return retina_detection_module(body, num_filter, filter_in, name)


def upsampling(data, num_filter, name):
    #ret = mx.symbol.Deconvolution(data=data, num_filter=num_filter, kernel=(4,4),  stride=(2, 2), pad=(1,1),
    #    num_group = num_filter, no_bias = True, attr={'__lr_mult__': '0.0', '__wd_mult__': '0.0'},
    #    name=name)
    #ret = mx.symbol.Deconvolution(data=data, num_filter=num_filter, kernel=(2,2),  stride=(2, 2), pad=(0,0),
    #    num_group = num_filter, no_bias = True, attr={'__lr_mult__': '0.0', '__wd_mult__': '0.0'},
    #    name=name)
    ret = mx.symbol.UpSampling(data, scale=2, sample_type='nearest', workspace=512, name=name, num_args=1)
    return ret

def get_sym_conv(data, sym):
    all_layers = sym.get_internals()

    isize = 640
    _, out_shape, _ = all_layers.infer_shape(data = (1,3,isize,isize))
    last_entry = None
    c1 = None
    c2 = None
    c3 = None
    c1_name = None
    c2_name = None
    c3_name = None
    c1_filter = -1
    c2_filter = -1
    c3_filter = -1
    #print(len(all_layers), len(out_shape))
    #print(all_layers.__class__)
    outputs = all_layers.list_outputs()
    #print(outputs.__class__, len(outputs))
    count = len(outputs)
    stride2name = {}
    stride2layer = {}
    stride2shape = {}
    for i in range(count):
      name = outputs[i]
      shape = out_shape[i]
      print(i, name, count, shape)
      if not name.endswith('_output'):
        continue
      if len(shape)!=4:
        continue
      assert isize%shape[2]==0
      if shape[1]>config.max_feat_channel:
        break
      stride = isize//shape[2]
      stride2name[stride] = name
      stride2layer[stride] = all_layers[name]
      stride2shape[stride] = shape
      #print(name, shape)
      #if c1 is None and shape[2]==isize//16:
      #  cname = last_entry[0]
      #  #print('c1', last_entry)
      #  c1 = all_layers[cname]
      #  c1_name = cname
      #if c2 is None and shape[2]==isize//32:
      #  cname = last_entry[0]
      #  #print('c2', last_entry)
      #  c2 = all_layers[cname]
      #  c2_name = cname
      #if shape[2]==isize//32:
      #  c3 = all_layers[name]
      #  #print('c3', name, shape)
      #  c3_name = name

      #last_entry = (name, shape)

    F1 = config.HEAD_FILTER_NUM
    F2 = F1
    strides = sorted(stride2name.keys())
    for stride in strides:
      print('stride', stride, stride2name[stride], stride2shape[stride])
    print('F1_F2', F1, F2)
    #print('cnames', c1_name, c2_name, c3_name, F1, F2)
    _bwm = 1.0
    c0 = stride2layer[4]
    c1 = stride2layer[8]
    c2 = stride2layer[16]
    c3 = stride2layer[32]
    c3 = conv_act_layer(c3, 'rf_c3_lateral',
        F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
    #c3_up = mx.symbol.UpSampling(c3, scale=2, sample_type='nearest', workspace=512, name='ssh_c3_up', num_args=1)
    c3_up = upsampling(c3, F2, 'rf_c3_upsampling')
    c2_lateral = conv_act_layer(c2, 'rf_c2_lateral',
        F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
    if config.USE_CROP:
      c3_up = mx.symbol.Crop(*[c3_up, c2_lateral])
    c2 = c2_lateral+c3_up
    c2 = conv_act_layer(c2, 'rf_c2_aggr',
        F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
    c1_lateral = conv_act_layer(c1, 'rf_c1_red_conv',
        F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
    #c2_up = mx.symbol.UpSampling(c2, scale=2, sample_type='nearest', workspace=512, name='ssh_m2_red_up', num_args=1)
    c2_up = upsampling(c2, F2, 'rf_c2_upsampling')
    #conv4_128 = mx.symbol.Crop(*[conv4_128, conv5_128_up])
    if config.USE_CROP:
      c2_up = mx.symbol.Crop(*[c2_up, c1_lateral])
    c1 = c1_lateral+c2_up
    c1 = conv_act_layer(c1, 'rf_c1_aggr',
        F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
    #m1 = head_module(c1, F2*config.CONTEXT_FILTER_RATIO, F2, 'rf_c1_det')
    #m2 = head_module(c2, F1*config.CONTEXT_FILTER_RATIO, F2, 'rf_c2_det')
    #m3 = head_module(c3, F1*config.CONTEXT_FILTER_RATIO, F2, 'rf_c3_det')
    m1 = c1
    m2 = c2
    m3 = c3
    if len(config.RPN_ANCHOR_CFG)==3:
      ret = {8: m1, 16:m2, 32: m3}
    elif len(config.RPN_ANCHOR_CFG)==1:
      ret = {16:m2}
    elif len(config.RPN_ANCHOR_CFG)==2:
      ret = {8: m1, 16:m2}
    elif len(config.RPN_ANCHOR_CFG)==4:
      c0_lateral = conv_act_layer(c0, 'rf_c0_lateral',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c1_up = upsampling(c1, F2, 'rf_c1_upsampling')
      if config.USE_CROP:
        c1_up = mx.symbol.Crop(*[c1_up, c0_lateral])
      c0 = c0_lateral+c1_up
      c0 = conv_act_layer(c0, 'rf_c0_aggr',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)

      #m0 = head_module(c0, F2*config.CONTEXT_FILTER_RATIO, F2, 'rf_c0_det') 
      m0 = c0
      ret = {4: m0, 8: m1, 16:m2, 32: m3}
    elif len(config.RPN_ANCHOR_CFG)==5:
      c0_lateral = conv_act_layer(c0, 'rf_c0_lateral',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c1_up = upsampling(c1, F2, 'rf_c1_upsampling')
      if config.USE_CROP:
        c1_up = mx.symbol.Crop(*[c1_up, c0_lateral])
      c0 = c0_lateral+c1_up
      c0 = conv_act_layer(c0, 'rf_c0_aggr',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)

      c4 = conv_act_layer(c3, 'rf_c4',
          F2, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu', bias_wd_mult=_bwm)
      #m0 = head_module(c0, F2*config.CONTEXT_FILTER_RATIO, F2, 'rf_c0_det')
      #m4 = head_module(c4, F1*config.CONTEXT_FILTER_RATIO, F2, 'rf_c4_det')
      m0 = c0
      m4 = c4
      ret = {4: m0, 8: m1, 16:m2, 32: m3, 64: m4}
    elif len(config.RPN_ANCHOR_CFG)==6:
      c0_lateral = conv_act_layer(c0, 'rf_c0_lateral',
          F2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)
      c1_up = upsampling(c1, F2, 'rf_c1_upsampling')
      if config.USE_CROP:
        c1_up = mx.symbol.Crop(*[c1_up, c0_lateral])
      c0 = c0_lateral+c1_up
      c0 = conv_act_layer(c0, 'rf_c0_aggr',
          F2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), act_type='relu', bias_wd_mult=_bwm)

      c4 = conv_act_layer(c3, 'rf_c4',
          F2, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu', bias_wd_mult=_bwm)
      c5 = conv_act_layer(c4, 'rf_c5',
          F2, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu', bias_wd_mult=_bwm)
      #m0 = head_module(c0, F2*config.CONTEXT_FILTER_RATIO, F2, 'rf_c0_det')
      #m4 = head_module(c4, F1*config.CONTEXT_FILTER_RATIO, F2, 'rf_c4_det')
      #m5 = head_module(c5, F1*config.CONTEXT_FILTER_RATIO, F2, 'rf_c5_det')
      m0 = c0
      m4 = c4
      m5 = c5
      ret = {4: m0, 8: m1, 16:m2, 32: m3, 64: m4, 128: m5}

    #return {8: m1, 16:m2, 32: m3}
    return ret

def get_out(conv_fpn_feat, prefix, stride, landmark=False, lr_mult=1.0, gt_boxes=None):
    A = config.NUM_ANCHORS
    bbox_pred_len = 4
    landmark_pred_len = 10
    if config.USE_BLUR:
      bbox_pred_len = 5
    if config.USE_OCCLUSION:
      landmark_pred_len = 15
    ret_group = []
    num_anchors = config.RPN_ANCHOR_CFG[str(stride)]['NUM_ANCHORS']
    cls_label = mx.symbol.Variable(name='%s_label_stride%d'%(prefix,stride))
    bbox_target = mx.symbol.Variable(name='%s_bbox_target_stride%d'%(prefix,stride))
    bbox_weight = mx.symbol.Variable(name='%s_bbox_weight_stride%d'%(prefix,stride))
    if landmark:
      landmark_target = mx.symbol.Variable(name='%s_landmark_target_stride%d'%(prefix,stride))
      landmark_weight = mx.symbol.Variable(name='%s_landmark_weight_stride%d'%(prefix,stride))
    conv_feat = conv_fpn_feat[stride]
    F1 = config.HEAD_FILTER_NUM
    F2 = F1
    rpn_relu = head_module(conv_feat, F1*config.CONTEXT_FILTER_RATIO, F2, 'rf_head_stride%d'%stride)

    rpn_cls_score = conv_only(rpn_relu, '%s_rpn_cls_score_stride%d'%(prefix, stride), 2*num_anchors,
        kernel=(1,1), pad=(0,0), stride=(1, 1))

    rpn_bbox_pred = conv_only(rpn_relu, '%s_rpn_bbox_pred_stride%d'%(prefix,stride), bbox_pred_len*num_anchors,
        kernel=(1,1), pad=(0,0), stride=(1, 1))

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                              shape=(0, 2, -1),
                                              name="%s_rpn_cls_score_reshape_stride%s" % (prefix,stride))

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
      label, anchor_weight, pos_count = mx.sym.Custom(op_type='rpn_fpn_ohem3', stride=int(stride), network=config.network, dataset=config.dataset, prefix=prefix, cls_score=rpn_cls_score_reshape, labels = cls_label)

      _bbox_weight = mx.sym.tile(anchor_weight, (1,1,bbox_pred_len))
      _bbox_weight = _bbox_weight.reshape((0, -1, A * bbox_pred_len)).transpose((0,2,1))
      bbox_weight = mx.sym.elemwise_mul(bbox_weight, _bbox_weight, name='%s_bbox_weight_mul_stride%s'%(prefix,stride))

      if landmark:
        _landmark_weight = mx.sym.tile(anchor_weight, (1,1,landmark_pred_len))
        _landmark_weight = _landmark_weight.reshape((0, -1, A * landmark_pred_len)).transpose((0,2,1))
        landmark_weight = mx.sym.elemwise_mul(landmark_weight, _landmark_weight, name='%s_landmark_weight_mul_stride%s'%(prefix,stride))
    else:
      label = cls_label
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

    pos_count = mx.symbol.sum(pos_count)
    pos_count = pos_count + 0.001 #avoid zero

    #bbox loss
    bbox_diff = rpn_bbox_pred_reshape-bbox_target
    bbox_diff = bbox_diff * bbox_weight
    rpn_bbox_loss_ = mx.symbol.smooth_l1(name='%s_rpn_bbox_loss_stride%d_'%(prefix,stride), scalar=3.0, data=bbox_diff)
    bbox_lr_mode0 = 0.25*lr_mult*config.TRAIN.BATCH_IMAGES / config.TRAIN.RPN_BATCH_SIZE
    landmark_lr_mode0 = 0.4*config.LANDMARK_LR_MULT*bbox_lr_mode0
    if config.LR_MODE==0:
      rpn_bbox_loss = mx.sym.MakeLoss(name='%s_rpn_bbox_loss_stride%d'%(prefix,stride), data=rpn_bbox_loss_, grad_scale=bbox_lr_mode0)
    else:
      rpn_bbox_loss_ = mx.symbol.broadcast_div(rpn_bbox_loss_, pos_count)
      rpn_bbox_loss = mx.sym.MakeLoss(name='%s_rpn_bbox_loss_stride%d'%(prefix,stride), data=rpn_bbox_loss_, grad_scale=0.5*lr_mult)
    ret_group.append(rpn_bbox_loss)
    ret_group.append(mx.sym.BlockGrad(bbox_weight))

    #landmark loss
    if landmark:
      landmark_diff = rpn_landmark_pred_reshape-landmark_target
      landmark_diff = landmark_diff * landmark_weight
      rpn_landmark_loss_ = mx.symbol.smooth_l1(name='%s_rpn_landmark_loss_stride%d_'%(prefix,stride), scalar=3.0, data=landmark_diff)
      if config.LR_MODE==0:
        rpn_landmark_loss = mx.sym.MakeLoss(name='%s_rpn_landmark_loss_stride%d'%(prefix,stride), data=rpn_landmark_loss_, grad_scale=landmark_lr_mode0)
      else:
        rpn_landmark_loss_ = mx.symbol.broadcast_div(rpn_landmark_loss_, pos_count)
        rpn_landmark_loss = mx.sym.MakeLoss(name='%s_rpn_landmark_loss_stride%d'%(prefix,stride), data=rpn_landmark_loss_, grad_scale=0.2*config.LANDMARK_LR_MULT*lr_mult)
      ret_group.append(rpn_landmark_loss)
      ret_group.append(mx.sym.BlockGrad(landmark_weight))
    if config.USE_3D:
      from rcnn.PY_OP import rpn_3d_mesh
      pass
    if config.CASCADE>0:
      if config.CASCADE_MODE==0:
        body = rpn_relu
      elif config.CASCADE_MODE==1:
        body = head_module(conv_feat, F1*config.CONTEXT_FILTER_RATIO, F2, 'rf_head_stride%d_cas'%stride)
      elif config.CASCADE_MODE==2:
        body = conv_feat + rpn_relu
        body = head_module(body, F1*config.CONTEXT_FILTER_RATIO, F2, 'rf_head_stride%d_cas'%stride)
      else:
        body = head_module(conv_feat, F1*config.CONTEXT_FILTER_RATIO, F2, 'rf_head_stride%d_cas'%stride)
        body = mx.sym.concat(body, rpn_cls_score, rpn_bbox_pred, rpn_landmark_pred, dim=1)

      #cls_pred = rpn_cls_prob
      cls_pred_t0 = rpn_cls_score_reshape
      cls_label_raw = cls_label
      cls_label_t0 = label
      bbox_pred_t0 = rpn_bbox_pred_reshape
      #bbox_pred = rpn_bbox_pred
      #bbox_pred = mx.sym.transpose(bbox_pred, (0, 2, 3, 1))
      #bbox_pred_len = 4
      #bbox_pred = mx.sym.reshape(bbox_pred, (0, -1, bbox_pred_len))
      bbox_label_t0 = bbox_target
      #prefix = prefix+'2'
      for casid in range(config.CASCADE):
        #pseudo-code
        #anchor_label = GENANCHOR(bbox_label, bbox_pred, stride)
        #bbox_label = F(anchor_label, bbox_pred)
        #bbox_label = bbox_label - bbox_pred
        cls_pred = conv_only(body, '%s_rpn_cls_score_stride%d_cas%d'%(prefix, stride, casid), 2*num_anchors,
            kernel=(1,1), pad=(0,0), stride=(1, 1))
        rpn_cls_score_reshape = mx.symbol.Reshape(data=cls_pred,
                                                  shape=(0, 2, -1),
                                                  name="%s_rpn_cls_score_reshape_stride%s_cas%d" % (prefix,stride, casid))

        #bbox_label equals to bbox_target
        #cls_pred, cls_label, bbox_pred, bbox_label, bbox_weight, pos_count = mx.sym.Custom(op_type='cascade_refine', stride=int(stride), network=config.network, dataset=config.dataset, prefix=prefix, cls_pred=cls_pred, cls_label = cls_label, bbox_pred = bbox_pred, bbox_label = bbox_label)
        #cls_label, bbox_label, anchor_weight, pos_count = mx.sym.Custom(op_type='cascade_refine', stride=int(stride), network=config.network, dataset=config.dataset, prefix=prefix, cls_pred_t0=cls_pred_t0, cls_label_t0 = cls_label_t0, cls_pred = rpn_cls_score_reshape, bbox_pred_t0 = bbox_pred_t0, bbox_label_t0 = bbox_label_t0)
        cls_label, bbox_label, anchor_weight, pos_count = mx.sym.Custom(op_type='cascade_refine', stride=int(stride), network=config.network, 
            dataset=config.dataset, prefix=prefix, 
            cls_label_t0 = cls_label_t0, cls_pred_t0=cls_pred_t0, cls_pred = rpn_cls_score_reshape, 
            bbox_pred_t0 = bbox_pred_t0, bbox_label_t0 = bbox_label_t0, 
            cls_label_raw = cls_label_raw, cas_gt_boxes = gt_boxes)
        if stride in config.CASCADE_CLS_STRIDES:
          rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape,
                                                 label=cls_label,
                                                 multi_output=True,
                                                 normalization='valid', use_ignore=True, ignore_label=-1,
                                                 grad_scale = lr_mult,
                                                 name='%s_rpn_cls_prob_stride%d_cas%d'%(prefix,stride,casid))
          ret_group.append(rpn_cls_prob)
          ret_group.append(mx.sym.BlockGrad(cls_label))
        if stride in config.CASCADE_BBOX_STRIDES:
          bbox_pred = conv_only(body, '%s_rpn_bbox_pred_stride%d_cas%d'%(prefix,stride,casid), bbox_pred_len*num_anchors,
              kernel=(1,1), pad=(0,0), stride=(1, 1))

          rpn_bbox_pred_reshape = mx.symbol.Reshape(data=bbox_pred,
                                                    shape=(0, 0, -1),
                                                    name="%s_rpn_bbox_pred_reshape_stride%s_cas%d" % (prefix,stride,casid))
          _bbox_weight = mx.sym.tile(anchor_weight, (1,1,bbox_pred_len))
          _bbox_weight = _bbox_weight.reshape((0, -1, A * bbox_pred_len)).transpose((0,2,1))
          bbox_weight = _bbox_weight
          pos_count = mx.symbol.sum(pos_count)
          pos_count = pos_count + 0.01 #avoid zero
          #bbox_weight = mx.sym.elemwise_mul(bbox_weight, _bbox_weight, name='%s_bbox_weight_mul_stride%s'%(prefix,stride))
          #bbox loss
          bbox_diff = rpn_bbox_pred_reshape-bbox_label
          bbox_diff = bbox_diff * bbox_weight
          rpn_bbox_loss_ = mx.symbol.smooth_l1(name='%s_rpn_bbox_loss_stride%d_cas%d'%(prefix,stride,casid), scalar=3.0, data=bbox_diff)
          if config.LR_MODE==0:
            rpn_bbox_loss = mx.sym.MakeLoss(name='%s_rpn_bbox_loss_stride%d_cas%d'%(prefix,stride,casid), data=rpn_bbox_loss_, grad_scale=bbox_lr_mode0)
          else:
            rpn_bbox_loss_ = mx.symbol.broadcast_div(rpn_bbox_loss_, pos_count)
            rpn_bbox_loss = mx.sym.MakeLoss(name='%s_rpn_bbox_loss_stride%d_cas%d'%(prefix,stride,casid), data=rpn_bbox_loss_, grad_scale=0.5*lr_mult)
          ret_group.append(rpn_bbox_loss)
          ret_group.append(mx.sym.BlockGrad(bbox_weight))
          #bbox_pred = rpn_bbox_pred_reshape

    return ret_group

def get_sym_train(sym):
    data = mx.symbol.Variable(name="data")

    # shared convolutional layers
    conv_fpn_feat = get_sym_conv(data, sym)
    ret_group = []
    gt_boxes = None
    if config.CASCADE>0:
      gt_boxes = mx.sym.Variable('gt_boxes')


    for stride in config.RPN_FEAT_STRIDE:
      ret = get_out(conv_fpn_feat, 'face', stride, config.FACE_LANDMARK, lr_mult=1.0, gt_boxes = gt_boxes)
      ret_group += ret

    return mx.sym.Group(ret_group)


