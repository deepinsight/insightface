import mxnet as mx
import numpy as np
import math
from mxnet.base import _Null

def conv_main(data, units, filters, workspace):
  body = data
  for i in xrange(len(units)):
    f = filters[i]
    _weight = mx.symbol.Variable("conv%d_%d_weight"%(i+1, 1), lr_mult=1.0)
    _bias = mx.symbol.Variable("conv%d_%d_bias"%(i+1, 1), lr_mult=2.0, wd_mult=0.0)
    body = mx.sym.Convolution(data=body, weight = _weight, bias = _bias, num_filter=f, kernel=(3, 3), stride=(2,2), pad=(1, 1),
                              name= "conv%d_%d"%(i+1, 1), workspace=workspace)

    body = mx.sym.LeakyReLU(data = body, act_type='prelu', name = "relu%d_%d" % (i+1, 1))
    idx = 2
    for j in xrange(units[i]):
      _body = mx.sym.Convolution(data=body, no_bias=True, num_filter=f, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                name= "conv%d_%d"%(i+1, idx), workspace=workspace)

      _body = mx.sym.LeakyReLU(data = _body, act_type='prelu', name = "relu%d_%d" % (i+1, idx))
      idx+=1
      _body = mx.sym.Convolution(data=_body, no_bias=True, num_filter=f, kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                name= "conv%d_%d"%(i+1, idx), workspace=workspace)
      _body = mx.sym.LeakyReLU(data = _body, act_type='prelu', name = "relu%d_%d" % (i+1, idx))
      idx+=1
      body = body+_body

  return body

def get_symbol(num_classes, num_layers, conv_workspace=256, **kwargs):
  if num_layers==64:
    units = [3,8,16,3]
    filters = [64,128,256,512]
  elif num_layers==20:
    units = [1,2,4,1]
    filters = [64,128,256,512]
    #filters = [64, 256, 512, 1024]
  elif num_layers==36:
    units = [2,4,8,2]
    filters = [64,128,256,512]
    #filters = [64, 256, 512, 1024]
  elif num_layers==60:
    units = [3,8,14,3]
    filters = [64,128,256,512]
  elif num_layers==104:
    units = [3,8,36,3]
    filters = [64,128,256,512]
    #filters = [64, 256, 512, 1024]
  data = mx.symbol.Variable('data')
  data = data-127.5
  data = data*0.0078125
  body = conv_main(data = data, units = units, filters = filters, workspace = conv_workspace)

  _weight = mx.symbol.Variable("fc1_weight", lr_mult=1.0)
  _bias = mx.symbol.Variable("fc1_bias", lr_mult=2.0, wd_mult=0.0)
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
    if k.startswith('conv') and k.endswith('_weight'):
      if not k.find('_1_')>=0:
        if num_layers<100:
          arg_params[k] = mx.random.normal(0, 0.01, shape=v)
          print('init', k)
    if k.endswith('_bias'):
      arg_params[k] = mx.nd.zeros(shape=v)
      print('init', k)
  return arg_params, aux_params

