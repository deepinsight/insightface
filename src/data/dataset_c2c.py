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
  ctx_num = len(ctx)
  path_imgrec = os.path.join(args.input, 'train.rec')
  path_imgidx = os.path.join(args.input, 'train.idx')
  imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
  outf = open(os.path.join(args.input, 'c2c'), 'w')
  s = imgrec.read_idx(0)
  header, _ = mx.recordio.unpack(s)
  assert header.flag>0
  print('header0 label', header.label)
  header0 = (int(header.label[0]), int(header.label[1]))
  #assert(header.flag==1)
  imgidx = range(1, int(header.label[0]))
  id2range = {}
  seq_identity = range(int(header.label[0]), int(header.label[1]))
  for identity in seq_identity:
    s = imgrec.read_idx(identity)
    header, _ = mx.recordio.unpack(s)
    id2range[identity] = (int(header.label[0]), int(header.label[1]))
  print('id2range', len(id2range))
  prop = face_image.load_property(args.input)
  image_size = prop.image_size
  print('image_size', image_size)
  vec = args.model.split(',')
  prefix = vec[0]
  epoch = int(vec[1])
  print('loading',prefix, epoch)
  model = mx.mod.Module.load(prefix, epoch, context = ctx)
  model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  nrof_images = 0
  nrof_removed = 0
  idx = 1
  id2label = {}
  pp = 0
  for _id, v in id2range.iteritems():
    pp+=1
    if pp%100==0:
      print('processing id', pp)
    _list = range(*v)
    ocontents = []
    for i in xrange(len(_list)):
      _idx = _list[i]
      #print('_idx', _id, _idx)
      s = imgrec.read_idx(_idx)
      ocontents.append(s)
    #continue
    embeddings = None
    headers = [None]*len(ocontents)
    #print(len(ocontents))
    ba = 0
    while True:
      bb = min(ba+args.batch_size, len(ocontents))
      if ba>=bb:
        break
      _batch_size = bb-ba
      _batch_size2 = max(_batch_size, ctx_num)
      data = nd.zeros( (_batch_size2,3, image_size[0], image_size[1]) )
      label = nd.zeros( (_batch_size2,) )
      count = bb-ba
      ii=0
      for i in xrange(ba, bb):
        header, img = mx.recordio.unpack(ocontents[i])
        headers[i] = header
        img = mx.image.imdecode(img)
        img = nd.transpose(img, axes=(2, 0, 1))
        data[ii][:] = img
        label[ii][:] = header.label[0]
        ii+=1
      while ii<_batch_size2:
        data[ii][:] = data[0][:]
        label[ii][:] = label[0][:]
        ii+=1
      db = mx.io.DataBatch(data=(data,), label=(label,))
      model.forward(db, is_train=False)
      net_out = model.get_outputs()
      net_out = net_out[0].asnumpy()
      if embeddings is None:
        embeddings = np.zeros( (len(ocontents), net_out.shape[1]))
      embeddings[ba:bb,:] = net_out[0:_batch_size,:]
      ba = bb
    embeddings = sklearn.preprocessing.normalize(embeddings)
    emb_mean = np.mean(embeddings, axis=0, keepdims=True)
    emb_mean = sklearn.preprocessing.normalize(emb_mean)
    sim = np.dot(embeddings, emb_mean.T)
    #print(sim.shape)
    sims = sim.flatten()
    assert len(_list)==len(sims)
    assert len(_list)==len(ocontents)
    for i in xrange(len(ocontents)):
      _sim = sims[i]
      _idx = _list[i]
      _header = headers[i]
      #TODO
      outf.write("%d,%f,%d\n"%(_idx, _sim, int(_header.label[1])))
  outf.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  # general
  parser.add_argument('--input', default='', type=str, help='')
  parser.add_argument('--model', default='../model/softmax,50', help='path to load model.')
  parser.add_argument('--batch-size', default=32, type=int, help='')
  args = parser.parse_args()
  main(args)

