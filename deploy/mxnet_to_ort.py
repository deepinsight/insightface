import sys
import os
import argparse
import onnx
import mxnet as mx
from onnx import helper
from onnx import TensorProto
from onnx import numpy_helper

print('mxnet version:', mx.__version__)
print('onnx version:', onnx.__version__)

assert mx.__version__ >= '1.8', 'mxnet version should >= 1.8'
assert onnx.__version__ >= '1.2.1', 'onnx version should >= 1.2.1'

import numpy as np
from mxnet.contrib import onnx as onnx_mxnet

def create_map(graph_member_list):
    member_map={}
    for n in graph_member_list:
        member_map[n.name]=n
    return member_map


parser = argparse.ArgumentParser(description='convert arcface models to onnx')
# general
parser.add_argument('params', default='./r100a/model-0000.params', help='mxnet params to load.')
parser.add_argument('output', default='./r100a.onnx', help='path to write onnx model.')
parser.add_argument('--input-shape', default='3,112,112', help='input shape.')
args = parser.parse_args()
input_shape = (1,) + tuple( [int(x) for x in args.input_shape.split(',')] )
print('input-shape:', input_shape)

params_file = args.params
pos = params_file.rfind('-')
sym_file = params_file[:pos] + "-symbol.json"
assert os.path.exists(sym_file)
assert os.path.exists(params_file)

print('exporting from', sym_file, params_file)
converted_model_path = onnx_mxnet.export_model(sym_file, params_file, [input_shape], np.float32, args.output)
model = onnx.load(args.output)
graph = model.graph
input_map = create_map(graph.input)
node_map = create_map(graph.node)
init_map = create_map(graph.initializer)

#fix PRelu issue
for input_name in input_map.keys():
    if input_name.endswith('_gamma'):
        node_name = input_name[:-6]
        if not node_name in node_map:
            continue
        node = node_map[node_name]
        if node.op_type!='PRelu':
            continue
        input_shape = input_map[input_name].type.tensor_type.shape.dim
        input_dim_val=input_shape[0].dim_value
        
        graph.initializer.remove(init_map[input_name])
        weight_array = numpy_helper.to_array(init_map[input_name])
        
        b=[]
        for w in weight_array:
            b.append(w)
        new_nv = helper.make_tensor(input_name, TensorProto.FLOAT, [input_dim_val,1,1], b)
        graph.initializer.extend([new_nv])

for init_name in init_map.keys():
    weight_array = numpy_helper.to_array(init_map[init_name])
    assert weight_array.dtype==np.float32
    if init_name in input_map:
        graph.input.remove(input_map[init_name])

#support batch-inference
graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'

onnx.save(model, args.output)

