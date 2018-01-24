from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import datetime
import mxnet as mx
from mxnet import ndarray as nd
import random
import argparse
import cv2
import time
import sklearn
from sklearn.decomposition import PCA
from easydict import EasyDict as edict
from sklearn.cluster import DBSCAN
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_image

def ch_dev(arg_params, aux_params, ctx):
  new_args = dict()
  new_auxs = dict()
  for k, v in arg_params.items():
    new_args[k] = v.as_in_context(ctx)
  for k, v in aux_params.items():
    new_auxs[k] = v.as_in_context(ctx)
  return new_args, new_auxs


def main(args):
  ctx = mx.gpu(args.gpu)
  args.ctx_num = 1
  prop = face_image.load_property(args.data)
  image_size = prop.image_size
  print('image_size', image_size)
  vec = args.model.split(',')
  prefix = vec[0]
  epoch = int(vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
  all_layers = sym.get_internals()
  sym = all_layers['fc1_output']
  #model = mx.mod.Module.load(prefix, epoch, context = ctx)
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  path_imgrec = os.path.join(args.data, 'train.rec')
  path_imgidx = os.path.join(args.data, 'train.idx')
  imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
  s = imgrec.read_idx(0)
  header, _ = mx.recordio.unpack(s)
  assert header.flag>0
  print('header0 label', header.label)
  header0 = (int(header.label[0]), int(header.label[1]))
  #assert(header.flag==1)
  imgidx = range(1, int(header.label[0]))
  stat = []
  count = 0
  data = nd.zeros( (1 ,3, image_size[0], image_size[1]) )
  label = nd.zeros( (1,) )
  for idx in imgidx:
    if len(stat)%100==0:
      print('processing', len(stat))
    s = imgrec.read_idx(idx)
    header, img = mx.recordio.unpack(s)
    img = mx.image.imdecode(img)
    img = nd.transpose(img, axes=(2, 0, 1))
    data[0][:] = img
    #input_blob = np.expand_dims(img.asnumpy(), axis=0)
    #arg_params["data"] = mx.nd.array(input_blob, ctx)
    #arg_params["softmax_label"] = mx.nd.empty((1,), ctx)
    time_now = datetime.datetime.now()
    #exe = sym.bind(ctx, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
    #exe.forward(is_train=False)
    #_embedding = exe.outputs[0].asnumpy().flatten()
    #db = mx.io.DataBatch(data=(data,), label=(label,))
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    net_out = model.get_outputs()[0].asnumpy()
    time_now2 = datetime.datetime.now()
    diff = time_now2 - time_now
    stat.append(diff.total_seconds())
    if len(stat)==args.param1:
      break
  stat = stat[10:]
  print('avg infer time', np.mean(stat))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='do network benchmark')
  # general
  parser.add_argument('--gpu', default=0, type=int, help='')
  parser.add_argument('--data', default='', type=str, help='')
  parser.add_argument('--model', default='../model/softmax,50', help='path to load model.')
  parser.add_argument('--batch-size', default=1, type=int, help='')
  parser.add_argument('--param1', default=1010, type=int, help='')
  args = parser.parse_args()
  main(args)

