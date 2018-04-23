import mxnet as mx
from mxnet import ndarray as nd
import argparse
import pickle
import sys
import os


cvte_dir = '/data/victor/cvte_baby/valid/CVTE_baby_valid_aligned'
pair_fn = '/data/victor/cvte_baby/valid/p_n_pairs_9000.txt'


parser = argparse.ArgumentParser(description='Package LFW images')
# general
parser.add_argument('--image-size', type=str, default='112,112', help='')
parser.add_argument('--output', default='', help='path to save.')
args = parser.parse_args()

image_size = [int(x) for x in args.image_size.split(',')]


def get_paths(cvte_dir, pair_fn):
  path_list, issame_list = [], []
  with open(pair_fn, 'r') as f:
      lines = map(lambda s: s.strip().split(),f.readlines())
      for fn1, fn2, issame in lines:
        path_list += (os.path.join(cvte_dir,fn1), os.path.join(cvte_dir,fn2))
        if issame == '1':
          flag = True
        else:
          flag = False
        issame_list.append(flag)
  return path_list, issame_list


cvte_paths, issame_list = get_paths(cvte_dir, pair_fn)
cvte_bins = []
#cvte_data = nd.empty((len(lfw_paths), 3, image_size[0], image_size[1]))
i = 0
for path in cvte_paths:
  with open(path, 'rb') as fin:
    _bin = fin.read()
    cvte_bins.append(_bin)
    #img = mx.image.imdecode(_bin)
    #img = nd.transpose(img, axes=(2, 0, 1))
    #cvte_data[i][:] = img
    i+=1
    if i%1000==0:
      print('loading cvte', i)

with open(args.output, 'wb') as f:
  pickle.dump((cvte_bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
