from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
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

sys.path.append(os.path.join(os.path.dirname(__file__),'..', 'common'))
import face_image

sys.path.append(os.path.join(os.path.dirname(__file__),'..', 'eval'))
import verification

def ch_dev(arg_params, aux_params, ctx):
  new_args = dict()
  new_auxs = dict()
  for k, v in arg_params.items():
    new_args[k] = v.as_in_context(ctx)
  for k, v in aux_params.items():
    new_auxs[k] = v.as_in_context(ctx)
  return new_args, new_auxs

def get_embedding(args, imgrec, id, image_size, model):
  s = imgrec.read_idx(id)
  header, _ = mx.recordio.unpack(s)
  ocontents = []
  for idx in xrange(int(header.label[0]), int(header.label[1])):
    s = imgrec.read_idx(idx)
    ocontents.append(s)
  embeddings = None
  #print(len(ocontents))
  ba = 0
  while True:
    bb = min(ba+args.batch_size, len(ocontents))
    if ba>=bb:
      break
    _batch_size = bb-ba
    _batch_size2 = max(_batch_size, args.ctx_num)
    data = nd.zeros( (_batch_size2,3, image_size[0], image_size[1]) )
    label = nd.zeros( (_batch_size2,) )
    count = bb-ba
    ii=0
    for i in xrange(ba, bb):
      header, img = mx.recordio.unpack(ocontents[i])
      img = mx.image.imdecode(img)
      img = nd.transpose(img, axes=(2, 0, 1))
      data[ii][:] = img
      label[ii][:] = header.label
      ii+=1
    while ii<_batch_size2:
      data[ii][:] = data[0][:]
      label[ii][:] = label[0][:]
      ii+=1
    #db = mx.io.DataBatch(data=(data,), label=(label,))
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    net_out = model.get_outputs()
    net_out = net_out[0].asnumpy()
    if embeddings is None:
      embeddings = np.zeros( (len(ocontents), net_out.shape[1]))
    embeddings[ba:bb,:] = net_out[0:_batch_size,:]
    ba = bb
  embeddings = sklearn.preprocessing.normalize(embeddings)
  embedding = np.mean(embeddings, axis=0, keepdims=True)
  embedding = sklearn.preprocessing.normalize(embedding).flatten()
  return embedding

def main(args):
  ctx = []
  cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
  if len(cvd)>0:
    for i in xrange(len(cvd.split(','))):
      ctx.append(mx.gpu(i))
  if len(ctx)==0:
    ctx = [mx.cpu()]
    print('use cpu')
  else:
    print('gpu num:', len(ctx))
  args.ctx_num = len(ctx)
  include_datasets = args.include.split(',')
  prop = face_image.load_property(include_datasets[0])
  image_size = prop.image_size
  print('image_size', image_size)
  vec = args.model.split(',')
  prefix = vec[0]
  epoch = int(vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  #arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
  all_layers = sym.get_internals()
  sym = all_layers['fc1_output']
  #model = mx.mod.Module.load(prefix, epoch, context = ctx)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  rec_list = []
  for ds in include_datasets:
    path_imgrec = os.path.join(ds, 'train.rec')
    path_imgidx = os.path.join(ds, 'train.idx')
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
    rec_list.append(imgrec)
  id_list_map = {}
  all_id_list = []
  test_limit = 0
  for ds_id in xrange(len(rec_list)):
    id_list = []
    imgrec = rec_list[ds_id]
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    assert header.flag>0
    print('header0 label', header.label)
    header0 = (int(header.label[0]), int(header.label[1]))
    #assert(header.flag==1)
    imgidx = range(1, int(header.label[0]))
    id2range = {}
    seq_identity = range(int(header.label[0]), int(header.label[1]))
    pp=0
    for identity in seq_identity:
      pp+=1
      if pp%10==0:
        print('processing id', pp)
      embedding = get_embedding(args, imgrec, identity, image_size, model)
      #print(embedding.shape)
      id_list.append( [ds_id, identity, embedding] )
      if test_limit>0 and pp>=test_limit:
        break
    id_list_map[ds_id] = id_list
    if ds_id==0:
      all_id_list += id_list
      print(ds_id, len(id_list))
    else:
      X = []
      for id_item in all_id_list:
        X.append(id_item[2])
      X = np.array(X)
      for i in xrange(len(id_list)):
        id_item = id_list[i]
        y = id_item[2]
        sim = np.dot(X, y.T)
        idx = np.where(sim>=args.param1)[0]
        if len(idx)>0:
          continue
        all_id_list.append(id_item)
      print(ds_id, len(id_list), len(all_id_list))


  if len(args.exclude)>0:
    if os.path.isdir(args.exclude):
      _path_imgrec = os.path.join(args.exclude, 'train.rec')
      _path_imgidx = os.path.join(args.exclude, 'train.idx')
      _imgrec = mx.recordio.MXIndexedRecordIO(_path_imgidx, _path_imgrec, 'r')  # pylint: disable=redefined-variable-type
      _ds_id = len(rec_list)
      _id_list = []
      s = _imgrec.read_idx(0)
      header, _ = mx.recordio.unpack(s)
      assert header.flag>0
      print('header0 label', header.label)
      header0 = (int(header.label[0]), int(header.label[1]))
      #assert(header.flag==1)
      imgidx = range(1, int(header.label[0]))
      seq_identity = range(int(header.label[0]), int(header.label[1]))
      pp=0
      for identity in seq_identity:
        pp+=1
        if pp%10==0:
          print('processing ex id', pp)
        embedding = get_embedding(args, _imgrec, identity, image_size, model)
        #print(embedding.shape)
        _id_list.append( (_ds_id, identity, embedding) )
        if test_limit>0 and pp>=test_limit:
          break
    else:
      _id_list = []
      data_set = verification.load_bin(args.exclude, image_size)[0][0]
      print(data_set.shape)
      data = nd.zeros( (1,3,image_size[0], image_size[1]))
      for i in xrange(data_set.shape[0]):
        data[0] = data_set[i]
        db = mx.io.DataBatch(data=(data,))
        model.forward(db, is_train=False)
        net_out = model.get_outputs()
        embedding = net_out[0].asnumpy().flatten()
        _norm=np.linalg.norm(embedding)
        embedding /= _norm
        _id_list.append( (i, i, embedding) )

    #X = []
    #for id_item in all_id_list:
    #  X.append(id_item[2])
    #X = np.array(X)
    #param1 = 0.3
    #while param1<=1.01:
    #  emap = {}
    #  for id_item in _id_list:
    #    y = id_item[2]
    #    sim = np.dot(X, y.T)
    #    #print(sim.shape)
    #    #print(sim)
    #    idx = np.where(sim>=param1)[0]
    #    for j in idx:
    #      emap[j] = 1
    #  exclude_removed = len(emap)
    #  print(param1, exclude_removed)
    #  param1+=0.05

      X = []
      for id_item in all_id_list:
        X.append(id_item[2])
      X = np.array(X)
      emap = {}
      for id_item in _id_list:
        y = id_item[2]
        sim = np.dot(X, y.T)
        idx = np.where(sim>=args.param2)[0]
        for j in idx:
          emap[j] = 1
          all_id_list[j][1] = -1
      print('exclude', len(emap))

  if args.test>0:
    return

  if not os.path.exists(args.output):
    os.makedirs(args.output)
  writer = mx.recordio.MXIndexedRecordIO(os.path.join(args.output, 'train.idx'), os.path.join(args.output, 'train.rec'), 'w')
  idx = 1
  identities = []
  nlabel = -1
  for id_item in all_id_list:
    if id_item[1]<0:
      continue
    nlabel+=1
    ds_id = id_item[0]
    imgrec = rec_list[ds_id]
    id = id_item[1]
    s = imgrec.read_idx(id)
    header, _ = mx.recordio.unpack(s)
    a, b = int(header.label[0]), int(header.label[1])
    identities.append( (idx, idx+b-a) )
    for _idx in xrange(a,b):
      s = imgrec.read_idx(_idx)
      _header, _content = mx.recordio.unpack(s)
      nheader = mx.recordio.IRHeader(0, nlabel, idx, 0)
      s = mx.recordio.pack(nheader, _content)
      writer.write_idx(idx, s)
      idx+=1
  id_idx = idx
  for id_label in identities:
    _header = mx.recordio.IRHeader(1, id_label, idx, 0)
    s = mx.recordio.pack(_header, '')
    writer.write_idx(idx, s)
    idx+=1
  _header = mx.recordio.IRHeader(1, (id_idx, idx), 0, 0)
  s = mx.recordio.pack(_header, '')
  writer.write_idx(0, s)
  with open(os.path.join(args.output, 'property'), 'w') as f:
    f.write("%d,%d,%d"%(len(identities), image_size[0], image_size[1]))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='do dataset merge')
  # general
  parser.add_argument('--include', default='', type=str, help='')
  parser.add_argument('--exclude', default='', type=str, help='')
  parser.add_argument('--output', default='', type=str, help='')
  parser.add_argument('--model', default='../model/softmax,50', help='path to load model.')
  parser.add_argument('--batch-size', default=32, type=int, help='')
  parser.add_argument('--param1', default=0.3, type=float, help='')
  parser.add_argument('--param2', default=0.4, type=float, help='')
  parser.add_argument('--mode', default=1, type=int, help='')
  parser.add_argument('--test', default=0, type=int, help='')
  args = parser.parse_args()
  main(args)

