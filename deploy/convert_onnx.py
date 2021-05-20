import sys
import os
import argparse
import onnx
import mxnet as mx

print('mxnet version:', mx.__version__)
print('onnx version:', onnx.__version__)
#make sure to install onnx-1.2.1
#pip uninstall onnx
#pip install onnx==1.2.1
assert onnx.__version__ == '1.2.1'
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet

parser = argparse.ArgumentParser(
    description='convert insightface models to onnx')
# general
parser.add_argument('--prefix',
                    default='./r100-arcface/model',
                    help='prefix to load model.')
parser.add_argument('--epoch',
                    default=0,
                    type=int,
                    help='epoch number to load model.')
parser.add_argument('--input-shape', default='3,112,112', help='input shape.')
parser.add_argument('--output-onnx',
                    default='./r100.onnx',
                    help='path to write onnx model.')
args = parser.parse_args()
input_shape = (1, ) + tuple([int(x) for x in args.input_shape.split(',')])
print('input-shape:', input_shape)

sym_file = "%s-symbol.json" % args.prefix
params_file = "%s-%04d.params" % (args.prefix, args.epoch)
assert os.path.exists(sym_file)
assert os.path.exists(params_file)
converted_model_path = onnx_mxnet.export_model(sym_file, params_file,
                                               [input_shape], np.float32,
                                               args.output_onnx)
