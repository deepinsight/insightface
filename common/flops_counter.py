
'''
@author: insightface
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json
import argparse
import numpy as np
import mxnet as mx


def is_no_bias(attr):
  ret = False
  if 'no_bias' in attr and (attr['no_bias']==True or attr['no_bias']=='True'):
    ret = True
  return ret

def count_fc_flops(input_filter, output_filter, attr):
  #print(input_filter, output_filter ,attr)
  ret = 2*input_filter*output_filter
  if is_no_bias(attr):
    ret -= output_filter
  return int(ret)


def count_conv_flops(input_shape, output_shape, attr):
  kernel = attr['kernel'][1:-1].split(',')
  kernel = [int(x) for x in kernel]

  #print('kernel', kernel)
  if is_no_bias(attr):
    ret = (2*input_shape[1]*kernel[0]*kernel[1]-1)*output_shape[2]*output_shape[3]*output_shape[1]
  else:
    ret = 2*input_shape[1]*kernel[0]*kernel[1]*output_shape[2]*output_shape[3]*output_shape[1]
  num_group = 1
  if 'num_group' in attr:
    num_group = int(attr['num_group'])
  ret /= num_group
  return int(ret)


def count_flops(sym, **data_shapes):
  all_layers = sym.get_internals()
  #print(all_layers)
  arg_shapes, out_shapes, aux_shapes = all_layers.infer_shape(**data_shapes)
  out_shape_dict = dict(zip(all_layers.list_outputs(), out_shapes))

  nodes = json.loads(sym.tojson())['nodes']
  nodeid_shape = {}
  for nodeid, node in enumerate(nodes):
    name = node['name']
    layer_name = name+"_output"
    if layer_name in out_shape_dict:
      nodeid_shape[nodeid] = out_shape_dict[layer_name]
  #print(nodeid_shape)
  FLOPs = 0
  for nodeid, node in enumerate(nodes):
    flops = 0
    if node['op']=='Convolution':
      output_shape = nodeid_shape[nodeid]
      name = node['name']
      attr = node['attrs']
      input_nodeid = node['inputs'][0][0]
      input_shape = nodeid_shape[input_nodeid]
      flops = count_conv_flops(input_shape, output_shape, attr)
    elif node['op']=='FullyConnected':
      attr = node['attrs']
      output_shape = nodeid_shape[nodeid]
      input_nodeid = node['inputs'][0][0]
      input_shape = nodeid_shape[input_nodeid]
      output_filter = output_shape[1]
      input_filter = input_shape[1]*input_shape[2]*input_shape[3]
      #assert len(input_shape)==4 and input_shape[2]==1 and input_shape[3]==1
      flops = count_fc_flops(input_filter, output_filter, attr)
    #print(node, flops)
    FLOPs += flops

  return FLOPs

def flops_str(FLOPs):
  preset = [ (1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'K') ]

  for p in preset:
    if FLOPs//p[0]>0:
      N = FLOPs/p[0]
      ret = "%.1f%s"%(N, p[1])
      return ret
  ret = "%.1f"%(FLOPs)
  return ret

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='flops counter')
  # general
  #parser.add_argument('--model', default='../models2/y2-arcface-retinat1/model,1', help='path to load model.')
  #parser.add_argument('--model', default='../models2/r100fc-arcface-retinaa/model,1', help='path to load model.')
  parser.add_argument('--model', default='../models2/r50fc-arcface-emore/model,1', help='path to load model.')
  args = parser.parse_args()
  _vec = args.model.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers['fc1_output']
  FLOPs = count_flops(sym, data=(1,3,112,112))
  print('FLOPs:', FLOPs)

