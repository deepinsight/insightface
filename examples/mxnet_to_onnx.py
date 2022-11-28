import sys
import os
import argparse
import onnx
import json
import mxnet as mx
from onnx import helper
from onnx import TensorProto
from onnx import numpy_helper
import onnxruntime
import cv2

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


parser = argparse.ArgumentParser(description='convert mxnet model to onnx')
# general
parser.add_argument('params', default='./r100a/model-0000.params', help='mxnet params to load.')
parser.add_argument('output', default='./r100a.onnx', help='path to write onnx model.')
parser.add_argument('--eps', default=1.0e-8, type=float, help='eps for weights.')
parser.add_argument('--input-shape', default='3,112,112', help='input shape.')
parser.add_argument('--check', action='store_true')
parser.add_argument('--batch', action='store_true')
parser.add_argument('--input-mean', default=0.0, type=float, help='input mean for checking.')
parser.add_argument('--input-std', default=1.0, type=float, help='input std for checking.')
args = parser.parse_args()
input_shape = (1,) + tuple( [int(x) for x in args.input_shape.split(',')] )

params_file = args.params
pos = params_file.rfind('-')
prefix = params_file[:pos]
epoch = int(params_file[pos+1:pos+5])
sym_file = prefix + "-symbol.json"
assert os.path.exists(sym_file)
assert os.path.exists(params_file)

sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

nodes = json.loads(sym.tojson())['nodes']
bn_fixgamma_list = []
for nodeid, node in enumerate(nodes):
    if node['op'] == 'BatchNorm':
        attr = node['attrs']
        fix_gamma = False
        if attr is not None and 'fix_gamma' in attr:
            if str(attr['fix_gamma']).lower()=='true':
                fix_gamma = True
        if fix_gamma:
            bn_fixgamma_list.append(node['name'])
        #print(node, fix_gamma)

print('fixgamma list:', bn_fixgamma_list)
layer = None
#layer = 'conv_2_dw_relu' #for debug

if layer is not None:
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']


eps = args.eps

arg = {}
aux = {}
invalid = 0
ac = 0
for k in arg_params:
    v = arg_params[k]
    nv = v.asnumpy()
    nv = nv.astype(np.float32)
    #print(k, nv.shape)
    if k.endswith('_gamma'):
        bnname = k[:-6]
        if bnname in bn_fixgamma_list:
            nv[:] = 1.0
    ac += nv.size
    invalid += np.count_nonzero(np.abs(nv)<eps)
    nv[np.abs(nv) < eps] = 0.0
    arg[k] = mx.nd.array(nv, dtype='float32')
arg_params = arg
invalid = 0
ac = 0
for k in aux_params:
    v = aux_params[k]
    nv = v.asnumpy().astype(np.float32)

    ac += nv.size
    invalid += np.count_nonzero(np.abs(nv)<eps)
    nv[np.abs(nv) < eps] = 0.0
    aux[k] = mx.nd.array(nv, dtype='float32')
aux_params = aux

all_args = {}
all_args.update(arg_params)
all_args.update(aux_params)
converted_model_path = onnx_mxnet.export_model(sym, all_args, [input_shape], np.float32, args.output, opset_version=11)


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
        _input_shape = input_map[input_name].type.tensor_type.shape.dim
        input_dim_val=_input_shape[0].dim_value
        
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
if args.batch:
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'

onnx.save(model, args.output)

#start to check correctness
if args.check:
    im_size = tuple(input_shape[2:])+(3,)
    img = np.random.randint(0, 256, size=im_size, dtype=np.uint8)
    input_size = tuple(input_shape[2:4][::-1])
    input_std = args.input_std
    input_mean = args.input_mean
    #print(img.shape, input_size)
    img = cv2.dnn.blobFromImage(img, 1.0/input_std, input_size, (input_mean, input_mean, input_mean), swapRB=True)
    ctx = mx.cpu()
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(for_training=False, data_shapes=[('data', input_shape)])
    _, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch) #reload original params
    model.set_params(arg_params, aux_params)

    data = mx.nd.array(img)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    x1 = model.get_outputs()[-1].asnumpy()

    session = onnxruntime.InferenceSession(args.output, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    x2 = session.run([output_name], {input_name : img})[0]
    print(x1.shape, x2.shape)
    print(x1.flatten()[:20])
    print(x2.flatten()[:20])

