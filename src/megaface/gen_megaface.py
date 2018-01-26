
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from datetime import datetime
import os.path
from easydict import EasyDict as edict
import time
import json
import sys
import numpy as np
import importlib
import itertools
import argparse
import struct
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_preprocess
from sklearn.preprocessing import normalize
#import facenet
#import lfw
import mxnet as mx
from mxnet import ndarray as nd
#from caffe.proto import caffe_pb2

megaface_out = '/raid5data/dplearn/megaface/MegaFace_Features'
#facescrub_out = '/raid5data/dplearn/megaface/FaceScrubSubset_Features'
facescrub_out = '/raid5data/dplearn/megaface/FaceScrub_Features'


def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_feature(image_path, bbox, landmark, nets, image_shape, use_align, aligned, use_mean):
  img = face_preprocess.read_image(image_path, mode='rgb')
  #print(img.shape)
  if img is None:
    print('parse image',image_path,'error')
    return None
  if not aligned:
    _landmark = landmark
    if not use_align:
      _landmark = None
    #cv2.imwrite("./align/origin_%s"%image_path.split('/')[-1], img)
    img = face_preprocess.preprocess(img, bbox=bbox, landmark=_landmark, image_size='%d,%d'%(image_shape[1], image_shape[2]))
  else:
    assert img.shape==(image_shape[1],image_shape[2],image_shape[0])
    #print('already aligned', image_path, img.shape)
    #img = cv2.resize(img, (image_shape[2], image_shape[1]))
  #cv2.imwrite("./align/%s"%image_path.split('/')[-1], img)
  if use_mean>0:
    v_mean = np.array([127.5,127.5,127.5], dtype=np.float32).reshape( (1,1,3) )
    img = img.astype(np.float32) - v_mean
    img *= 0.0078125
  img = np.transpose( img, (2,0,1) )
  F = None
  for net in nets:
    embedding = None
    #ppatch = net.patch
    for flipid in [0,1]:
      _img = np.copy(img)
      if flipid==1:
        do_flip(_img)
      #nimg = np.zeros(_img.shape, dtype=np.float32)
      #nimg[:,ppatch[1]:ppatch[3],ppatch[0]:ppatch[2]] = _img[:, ppatch[1]:ppatch[3], ppatch[0]:ppatch[2]]
      #_img = nimg
      input_blob = np.expand_dims(_img, axis=0)
      data = mx.nd.array(input_blob)
      db = mx.io.DataBatch(data=(data,))
      net.model.forward(db, is_train=False)
      _embedding = net.model.get_outputs()[0].asnumpy().flatten()
      #print(_embedding.shape)
      if embedding is None:
        embedding = _embedding
      else:
        embedding += _embedding
    _norm=np.linalg.norm(embedding)
    embedding /= _norm
    if F is None:
      F = embedding
    else:
      #F += embedding
      F = np.concatenate((F,embedding), axis=0)
  _norm=np.linalg.norm(F)
  F /= _norm
  #print(F.shape)
  return F


def write_bin(path, feature):
  feature = list(feature)
  with open(path, 'wb') as f:
    f.write(struct.pack('4i', len(feature),1,4,5))
    f.write(struct.pack("%df"%len(feature), *feature))

def main(args):

  print(args)
  gpuid = args.gpu
  ctx = mx.gpu(gpuid)
  nets = []
  image_shape = [int(x) for x in args.image_size.split(',')]
  for model in args.model.split('|'):
    vec = model.split(',')
    assert len(vec)>1
    prefix = vec[0]
    epoch = int(vec[1])
    print('loading',prefix, epoch)
    net = edict()
    net.ctx = ctx
    net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(prefix, epoch)
    #net.arg_params, net.aux_params = ch_dev(net.arg_params, net.aux_params, net.ctx)
    all_layers = net.sym.get_internals()
    net.sym = all_layers['fc1_output']
    net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names = None)
    net.model.bind(data_shapes=[('data', (1, 3, image_shape[1], image_shape[2]))])
    net.model.set_params(net.arg_params, net.aux_params)
    #_pp = prefix.rfind('p')+1
    #_pp = prefix[_pp:]
    #net.patch = [int(x) for x in _pp.split('_')]
    #assert len(net.patch)==5
    #print('patch', net.patch)
    nets.append(net)

  #megaface_lst = "/raid5data/dplearn/faceinsight_align_megaface.lst"
  megaface_lst = "/raid5data/dplearn/megaface/megaface_mtcnn_112x112/lst"
  #facescrub_lst = "/raid5data/dplearn/faceinsight_align_facescrub.lst"
  facescrub_lst = "/raid5data/dplearn/megaface/facescrubr/small_lst"
  if args.fsall>0:
    facescrub_lst = "/raid5data/dplearn/megaface/facescrubr/lst"

  if args.skip==0:
    i = 0
    succ = 0
    for line in open(facescrub_lst, 'r'):
      if i%1000==0:
        print("writing fs",i, succ)
      i+=1
      image_path, label, bbox, landmark, aligned = face_preprocess.parse_lst_line(line)
      _path = image_path.split('/')
      a,b = _path[-2], _path[-1]
      #a = a.replace(' ', '_')
      #b = b.replace(' ', '_')
      out_dir = os.path.join(facescrub_out, a)
      if not os.path.exists(out_dir):
        os.makedirs(out_dir)
      #file, ext = os.path.splitext(b)
      #image_id = int(file.split('_')[-1])
      #if image_id==40499 or image_id==10788 or image_id==2367:
      #  b = file
      #if len(ext)==0:
      #  print(image_path)
      #  image_path = image_path+".jpg"
      #if facescrub_aligned_root is not None:
      #  _vec = image_path.split('/')
      #  _image_path = os.path.join(facescrub_aligned_root, _vec[-2], _vec[-1])
      #  _base, _ext = os.path.splitext(_image_path)
      #  if _ext=='.gif':
      #    _image_path = _base+".jpg"
      #    print('changing', _image_path)
      #  if os.path.exists(_image_path):
      #    image_path = _image_path
      #    bbox = None
      #    landmark = None
      #    aligned = True
      #  else:
      #    print("not aligned:",_image_path)
      feature = get_feature(image_path, bbox, landmark, nets, image_shape, True, aligned, args.mean)
      if feature is None:
        print('feature none', image_path)
        continue
      #print(np.linalg.norm(feature))
      out_path = os.path.join(out_dir, b+"_%s_%dx%d.bin"%(args.algo, image_shape[1], image_shape[2]))
      write_bin(out_path, feature)
      succ+=1
    print('fs stat',i, succ)

  #return
  if args.mf==0:
    return
  i = 0
  succ = 0
  for line in open(megaface_lst, 'r'):
    if i%1000==0:
      print("writing mf",i, succ)
    i+=1
    if i<=args.skip:
      continue
    image_path, label, bbox, landmark, aligned = face_preprocess.parse_lst_line(line)
    assert aligned==True
    _path = image_path.split('/')
    a1, a2, b = _path[-3], _path[-2], _path[-1]
    out_dir = os.path.join(megaface_out, a1, a2)
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
      #continue
    #print(landmark)
    feature = get_feature(image_path, bbox, landmark, nets, image_shape, True, aligned, args.mean)
    if feature is None:
      continue
    out_path = os.path.join(out_dir, b+"_%s_%dx%d.bin"%(args.algo, image_shape[1], image_shape[2]))
    #print(out_path)
    write_bin(out_path, feature)
    succ+=1
  print('mf stat',i, succ)


def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--batch_size', type=int, help='', default=100)
  parser.add_argument('--image_size', type=str, help='', default='3,112,112')
  parser.add_argument('--gpu', type=int, help='', default=0)
  parser.add_argument('--mean', type=int, help='', default=0)
  parser.add_argument('--seed', type=int, help='', default=727)
  parser.add_argument('--skip', type=int, help='', default=0)
  parser.add_argument('--concat', type=int, help='', default=0)
  parser.add_argument('--fsall', type=int, help='', default=0)
  parser.add_argument('--mf', type=int, help='', default=1)
  parser.add_argument('--algo', type=str, help='', default='mxsphereface20c')
  #parser.add_argument('--model', type=str, help='', default='../model/sphereface-20-p0_0_96_112_0,22|../model/sphereface-20-p0_0_96_95_0,21|../model/sphereface-20-p0_0_80_95_0,21')
  #parser.add_argument('--model', type=str, help='', default='../model/sphereface-s60-p0_0_96_112_0,31|../model/sphereface-s60-p0_0_96_95_0,21|../model/sphereface2-s60-p0_0_96_112_0,21|../model/sphereface3-s60-p0_0_96_95_0,23')
  #parser.add_argument('--model', type=str, help='', default='../model/sphereface-s60-p0_0_96_112_0,31|../model/sphereface-s60-p0_0_96_95_0,21|../model/sphereface2-s60-p0_0_96_112_0,21|../model/sphereface3-s60-p0_0_96_95_0,23|../model/sphereface-20-p0_0_96_112_0,22|../model/sphereface-20-p0_0_96_95_0,21|../model/sphereface-20-p0_0_80_95_0,21')
  #parser.add_argument('--model', type=str, help='', default='../model/spherefacei-s60-p0_0_96_112_0,135')
  #parser.add_argument('--model', type=str, help='', default='../model/spherefacei-s60-p0_0_96_95_0,95')
  parser.add_argument('--model', type=str, help='', default='../model/spherefacei-s60-p0_15_96_112_0,95')
  return parser.parse_args(argv)

if __name__ == '__main__':
  main(parse_arguments(sys.argv[1:]))

