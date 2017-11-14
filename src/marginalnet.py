import mxnet as mx
import numpy as np



def Conv(data, num_filter, kernel, stride, pad, name, no_bias=False, workspace=256):
    _weight = mx.symbol.Variable(name+'_weight')
    _bias = mx.symbol.Variable(name+'_bias', lr_mult=2.0, wd_mult=0.0)
    body = mx.sym.Convolution(data=data, weight = _weight, bias = _bias, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias = no_bias, workspace = workspace, name = name)
    return body

def Act(data, name):
    body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    return body

def resnet_unit0(data, num_filter, name, workspace = 256):
  bn_mom = 0.9
  body = Conv(data=data, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1, 1),
                            name=name+"_conv", workspace=workspace)
  body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name+'_bn')
  body = Act(data=body, name=name+'_relu')
  return body

def resnet_unit1(data, num_filter, name, dim_match=True, workspace = 256):
  bn_mom = 0.9
  shortcut = data
  body = Conv(data=data, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1, 1),
                            name=name+"_conv1", workspace=workspace)
  body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name+'_bn1')
  body = Act(data=body, name=name+'_relu1')
  body = Conv(data=body, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1, 1),
                            name=name+"_conv2", workspace=workspace)
  body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name+'_bn2')
  if dim_match:
    body = body+shortcut
  body = Act(data=body, name=name+'_relu2')
  return body

def resnet_unit2(data, num_filter, name, dim_match=True, workspace = 256):
  bn_mom = 0.9
  shortcut = data
  body = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name+'_bn1')
  body = Act(data=body, name=name+'_relu1')
  body = Conv(data=body, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1, 1),
                            name=name+"_conv1", workspace=workspace)
  #body = mx.symbol.Dropout(data=body, p=0.2)
  body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name+'_bn2')
  body = Act(data=body, name=name+'_relu2')
  body = Conv(data=body, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1, 1),
                            name=name+"_conv2", workspace=workspace)
  if dim_match:
    body = body+shortcut
  return body

def resnet_unit3(data, num_filter, name, dim_match=True, workspace = 256):
  bn_mom = 0.9
  shortcut = data
  body = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name+'_bn1')
  body = Conv(data=body, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1, 1),
                            name=name+"_conv1", workspace=workspace)
  body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name+'_bn2')
  body = Act(data=body, name=name+'_relu1')
  body = Conv(data=body, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1, 1),
                            name=name+"_conv2", workspace=workspace)
  body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name+'_bn3')
  if dim_match:
    body = body+shortcut
  return body

def resnet_unit(rtype, data, num_filter, name, dim_match=True, workspace = 256):
  if rtype==1:
    return resnet_unit1(data=data, num_filter=num_filter, name=name, dim_match=dim_match, workspace=workspace)
  elif rtype==2:
    return resnet_unit2(data=data, num_filter=num_filter, name=name, dim_match=dim_match, workspace=workspace)
  else:
    return resnet_unit3(data=data, num_filter=num_filter, name=name, dim_match=dim_match, workspace=workspace)

def resnet(data, units, filters, rtype, workspace):
  body = resnet_unit0(data=data, num_filter=32, name="stage%d_unit%d"%(0, 0))
  for i in xrange(len(units)):
    f = filters[i]
    if rtype>=10:
      body = resnet_unit(rtype=rtype, data=body, num_filter=f, name="stage%d_unit%d"%(i+1, 0), dim_match=False) # do not connect to last layer, dim not match
    else:
      body = resnet_unit0(data=body, num_filter=f, name="stage%d_unit%d"%(i+1, 0)) # do not connect to last layer, dim not match
    body = mx.sym.Pooling(data=body, kernel=(2, 2), stride=(2,2), pad=(0,0), pool_type='max', name="stage%d_pool"%(i+1))
    for j in xrange(units[i]):
      body = resnet_unit(rtype=rtype, data=body, num_filter=f, name="stage%d_unit%d"%(i+1, j+1), dim_match=True)

  return body

def get_symbol(num_classes, num_layers, conv_workspace=256):
    data = mx.symbol.Variable('data')
    bn_mom = 0.9
    #data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    units = [1,2,5,3] # all number of layers = sum(units)*2+len(units)+1
    filter_list = [64, 128, 256, 512]
    rtype = 1
    use_last_bn = True
    use_dropout = True
    if num_layers==27:
      rtype = 1
    elif num_layers==28:
      rtype = 2
    elif num_layers==29:
      rtype = 3
      #use_last_bn = False
      #use_dropout = False

    body = resnet(data = data, units = units, filters = filter_list, rtype=rtype, workspace = conv_workspace)
    if use_dropout:
      body = mx.symbol.Dropout(data=body, p=0.4)
    _weight = mx.symbol.Variable("fc1_weight")
    _bias = mx.symbol.Variable("fc1_bias", lr_mult=2.0, wd_mult=0.0)
    if use_last_bn:
      fc1 = mx.sym.FullyConnected(data=body, weight=_weight, bias=_bias, num_hidden=num_classes, name='pre_fc1')
      fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
    else:
      fc1 = mx.sym.FullyConnected(data=body, weight=_weight, bias=_bias, num_hidden=num_classes, name='fc1')
    return fc1

def init_weights(sym, data_shape_dict, num_layers):
  arg_name = sym.list_arguments()
  aux_name = sym.list_auxiliary_states()
  arg_shape, aaa, aux_shape = sym.infer_shape(**data_shape_dict)
  #print(data_shape_dict)
  #print(arg_name)
  #print(arg_shape)
  arg_params = {}
  aux_params = None
  #print(aaa)
  #print(aux_shape)
  arg_shape_dict = dict(zip(arg_name, arg_shape))
  aux_shape_dict = dict(zip(aux_name, aux_shape))
  #print(aux_shape)
  #print(aux_params)
  #print(arg_shape_dict)
  for k,v in arg_shape_dict.iteritems():
    #print('find', k)
    if k.endswith('_weight') and k.find('_conv')>=0:
      if not k.find('_unit0_')>=0:
        arg_params[k] = mx.random.normal(0, 0.01, shape=v)
        print('init', k)
    if k.endswith('_bias'):
      arg_params[k] = mx.nd.zeros(shape=v)
      print('init', k)
  return arg_params, aux_params

