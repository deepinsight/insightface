
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import time
import shutil
import sys
import numpy as np
import argparse
import struct
import cv2
import mxnet as mx
from mxnet import ndarray as nd


feature_dim = 512
feature_ext = 1

def load_bin(path, fill = 0.0):
  with open(path, 'rb') as f:
    bb = f.read(4*4)
    #print(len(bb))
    v = struct.unpack('4i', bb)
    #print(v[0])
    bb = f.read(v[0]*4)
    v = struct.unpack("%df"%(v[0]), bb)
    feature = np.full( (feature_dim+feature_ext,), fill, dtype=np.float32)
    feature[0:feature_dim] = v
    #feature = np.array( v, dtype=np.float32)
  #print(feature.shape)
  #print(np.linalg.norm(feature))
  return feature

def write_bin(path, feature):
  feature = list(feature)
  with open(path, 'wb') as f:
    f.write(struct.pack('4i', len(feature),1,4,5))
    f.write(struct.pack("%df"%len(feature), *feature))

def main(args):


  fs_noise_map = {}
  for line in open(args.facescrub_noises, 'r'):
    if line.startswith('#'):
      continue
    line = line.strip()
    fname = line.split('.')[0]
    p = fname.rfind('_')
    fname = fname[0:p]
    fs_noise_map[line] = fname

  print(len(fs_noise_map))

  i=0
  fname2center = {}
  noises = []
  for line in open(args.facescrub_lst, 'r'):
    if i%1000==0:
      print("reading fs",i)
    i+=1
    image_path = line.strip()
    _path = image_path.split('/')
    a, b = _path[-2], _path[-1]
    feature_path = os.path.join(args.feature_dir_input, 'facescrub', a, "%s_%s.bin"%(b, args.algo))
    feature_dir_out = os.path.join(args.feature_dir_out, 'facescrub', a)
    if not os.path.exists(feature_dir_out):
      os.makedirs(feature_dir_out)
    feature_path_out = os.path.join(feature_dir_out, "%s_%s.bin"%(b, args.algo))
    #print(b)
    if not b in fs_noise_map:
      #shutil.copyfile(feature_path, feature_path_out)
      feature = load_bin(feature_path)
      write_bin(feature_path_out, feature)
      if not a in fname2center:
        fname2center[a] = np.zeros((feature_dim+feature_ext,), dtype=np.float32)
      fname2center[a] += feature
    else:
      #print('n', b)
      noises.append( (a,b) )
  print(len(noises))

  for k in noises:
    a,b = k
    assert a in fname2center
    center = fname2center[a]
    g = np.zeros( (feature_dim+feature_ext,), dtype=np.float32)
    g2 = np.random.uniform(-0.001, 0.001, (feature_dim,))
    g[0:feature_dim] = g2
    f = center+g
    _norm=np.linalg.norm(f)
    f /= _norm
    feature_path_out = os.path.join(args.feature_dir_out, 'facescrub', a, "%s_%s.bin"%(b, args.algo))
    write_bin(feature_path_out, f)

  mf_noise_map = {}
  for line in open(args.megaface_noises, 'r'):
    if line.startswith('#'):
      continue
    line = line.strip()
    _vec = line.split("\t")
    if len(_vec)>1:
      line = _vec[1]
    mf_noise_map[line] = 1

  print(len(mf_noise_map))

  i=0
  nrof_noises = 0
  for line in open(args.megaface_lst, 'r'):
    if i%1000==0:
      print("reading mf",i)
    i+=1
    image_path = line.strip()
    _path = image_path.split('/')
    a1, a2, b = _path[-3], _path[-2], _path[-1]
    feature_path = os.path.join(args.feature_dir_input, 'megaface', a1, a2, "%s_%s.bin"%(b, args.algo))
    feature_dir_out = os.path.join(args.feature_dir_out, 'megaface', a1, a2)
    if not os.path.exists(feature_dir_out):
      os.makedirs(feature_dir_out)
    feature_path_out = os.path.join(feature_dir_out, "%s_%s.bin"%(b, args.algo))
    bb = '/'.join([a1, a2, b])
    #print(b)
    if not bb in mf_noise_map:
      feature = load_bin(feature_path)
      write_bin(feature_path_out, feature)
      #shutil.copyfile(feature_path, feature_path_out)
    else:
      feature = load_bin(feature_path, 100.0)
      write_bin(feature_path_out, feature)
      #g = np.random.uniform(-0.001, 0.001, (feature_dim,))
      #print('n', bb)
      #write_bin(feature_path_out, g)
      nrof_noises+=1
  print(nrof_noises)


def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--facescrub-noises', type=str, help='', default='./data/facescrub_noises.txt')
  parser.add_argument('--megaface-noises', type=str, help='', default='./data/megaface_noises.txt')
  parser.add_argument('--algo', type=str, help='', default='insightface')
  parser.add_argument('--facescrub-lst', type=str, help='', default='./data/facescrub_lst')
  parser.add_argument('--megaface-lst', type=str, help='', default='./data/megaface_lst')
  parser.add_argument('--feature-dir-input', type=str, help='', default='./feature_out')
  parser.add_argument('--feature-dir-out', type=str, help='', default='./feature_out_clean')
  return parser.parse_args(argv)

if __name__ == '__main__':
  main(parse_arguments(sys.argv[1:]))


