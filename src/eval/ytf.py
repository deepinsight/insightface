from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import argparse
import sys
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
import datetime
import pickle
from sklearn.decomposition import PCA
import mxnet as mx
from mxnet import ndarray as nd
from verification import evaluate
from verification import calculate_accuracy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_image


model = None
feature_cache = {}
image_size = [112,112]

def get_feature(name, vid, args):
  global feature_cache
  key = (name,vid)
  if key in feature_cache:
    return feature_cache[key]

  input_dir = os.path.join(args.image_dir, name, str(vid))
  data = nd.zeros( (1 ,3, image_size[0], image_size[1]) )
  F = []
  for img in os.listdir(input_dir):
    img = os.path.join(input_dir, img)
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2,0,1))
    data[0][:] = img
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    net_out = model.get_outputs()[0].asnumpy().flatten()
    F.append(net_out)
  F = np.array(F)
  F = sklearn.preprocessing.normalize(F)
  feature = np.mean(F, axis=0, keepdims=True)
  feature = sklearn.preprocessing.normalize(feature).flatten()

  feature_cache[key] = feature
  return feature

def get_feature_set(name, vid, args):
  global feature_cache
  key = (name,vid)
  if key in feature_cache:
    return feature_cache[key]

  input_dir = os.path.join(args.image_dir, name, str(vid))
  data = nd.zeros( (1 ,3, image_size[0], image_size[1]) )
  F = []
  for img in os.listdir(input_dir):
    img = os.path.join(input_dir, img)
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2,0,1))
    data[0][:] = img
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    net_out = model.get_outputs()[0].asnumpy().flatten()
    F.append(net_out)
  F = np.array(F)
  F = sklearn.preprocessing.normalize(F)

  feature_cache[key] = F
  return F

def main(args):
  global model
  ctx = mx.gpu(args.gpu)
  args.ctx_num = 1
  print('image_size', image_size)
  vec = args.model.split(',')
  prefix = vec[0]
  epoch = int(vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers['fc1_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  args.image_dir = os.path.join(args.data_dir, 'images')
  pairs_file = os.path.join(args.data_dir, 'splits2.txt')
  embeddings = []
  issame_list = []
  data = []
  pp = 0
  for line in open(pairs_file, 'r'):
    line = line.strip()
    if line.startswith('split'):
      continue
    pp+=1
    if pp%10==0:
      print('processing', pp)
    vec = line.split(',')
    assert len(vec)>=5
    issame_list.append(int(vec[-1]))
    for i in [2,3]:
      _str = vec[i].strip()
      _vec = _str.split('/')
      assert len(_vec)==2
      name = _vec[0]
      vid = int(_vec[1])
      feature = get_feature(name, vid, args)
      print('feature', feature.shape)
      embeddings.append(feature)
      data.append( (name, vid) )
    #if len(issame_list)==20:
    #  break
  embeddings = np.array(embeddings)
  print(embeddings.shape)
  thresholds = np.arange(0, 4, 0.01)
  actual_issame = np.asarray(issame_list)
  nrof_folds = 10
  embeddings1 = embeddings[0::2]
  embeddings2 = embeddings[1::2]
  assert(embeddings1.shape[0] == embeddings2.shape[0])
  assert(embeddings1.shape[1] == embeddings2.shape[1])
  nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
  nrof_thresholds = len(thresholds)
  k_fold = KFold(n_splits=nrof_folds, shuffle=False)
  
  tprs = np.zeros((nrof_folds,nrof_thresholds))
  fprs = np.zeros((nrof_folds,nrof_thresholds))
  accuracy = np.zeros((nrof_folds))
  indices = np.arange(nrof_pairs)
  
  diff = np.subtract(embeddings1, embeddings2)
  dist = np.sum(np.square(diff),1)
  pouts = []
  nouts = []
  for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
      # Find the best threshold for the fold
      acc_train = np.zeros((nrof_thresholds))
      #print(train_set)
      #print(train_set.__class__)
      for threshold_idx, threshold in enumerate(thresholds):
          p2 = dist[train_set]
          p3 = actual_issame[train_set]
          _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, p2, p3)
      best_threshold_index = np.argmax(acc_train)
      for threshold_idx, threshold in enumerate(thresholds):
          tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
      _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
      best_threshold = thresholds[best_threshold_index]
      for iid in test_set:
        ida = iid*2
        idb = ida+1
        asame = actual_issame[iid]
        _dist = dist[iid]
        violate = _dist - best_threshold
        if not asame:
          violate *= -1.0
        if violate>0.0:
          dataa = data[ida]
          datab = data[idb]
          #print(imga.shape, imgb.shape, violate, asame, _dist)
          if asame:
            pouts.append( (dataa, datab, _dist, best_threshold, ida) )
          else:
            nouts.append( (dataa, datab, _dist, best_threshold, ida) )

        
  tpr = np.mean(tprs,0)
  fpr = np.mean(fprs,0)
  acc = np.mean(accuracy)
  pouts = sorted(pouts, key = lambda x: x[2], reverse=True)
  nouts = sorted(nouts, key = lambda x: x[2], reverse=False)
  print(len(pouts), len(nouts))
  print('acc', acc)
  if len(nouts)>0:
    threshold = nouts[0][3]
  else:
    threshold = pouts[-1][3]
  #print('threshold', threshold)
  print('positive(false negative):')
  for out in pouts:
    print("\t%s\t%s\t(distance:%f, threshold:%f)"%(out[0], out[1], out[2], out[3]))
  print('negative(false positive):')
  for out in nouts:
    print("\t%s\t%s\t(distance:%f, threshold:%f)"%(out[0], out[1], out[2], out[3]))





  #_, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=10)
  #acc2, std2 = np.mean(accuracy), np.std(accuracy)
  #print('acc', acc2)

def main2(args):
  global model
  ctx = mx.gpu(args.gpu)
  args.ctx_num = 1
  print('image_size', image_size)
  vec = args.model.split(',')
  prefix = vec[0]
  epoch = int(vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers['fc1_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  args.image_dir = os.path.join(args.data_dir, 'images')
  pairs_file = os.path.join(args.data_dir, 'splits2.txt')
  issame_list = []
  dist = []
  pp = 0
  for line in open(pairs_file, 'r'):
    line = line.strip()
    if line.startswith('split'):
      continue
    pp+=1
    if pp%10==0:
      print('processing', pp)
    vec = line.split(',')
    assert len(vec)>=5
    issame_list.append(int(vec[-1]))
    feature_sets = []
    for i in [2,3]:
      _str = vec[i].strip()
      _vec = _str.split('/')
      assert len(_vec)==2
      name = _vec[0]
      vid = int(_vec[1])
      feature = get_feature_set(name, vid, args)
      print('feature', len(feature))
      feature_sets.append(feature)
    X = feature_sets[0]
    Y = feature_sets[1]
    _dist = euclidean_distances(X, Y)
    _dist = _dist*_dist
    #_tmp = np.eye(_dist.shape[0], dtype=np.float32)
    #_dist += _tmp
    if args.mode==2:
      _dist = np.amin(_dist)
    elif args.mode==3:
      _dist = np.mean(_dist)
    else:
      _dist = np.amax(_dist)
    print(_dist)
    dist.append(_dist)
    #if len(dist)==10:
    #  break

  dist = np.array(dist)
  nrof_folds = 10
  thresholds = np.arange(0, 4, 0.01)
  actual_issame = np.array(issame_list)
  nrof_pairs = len(actual_issame)
  nrof_thresholds = len(thresholds)
  k_fold = KFold(n_splits=nrof_folds, shuffle=False)
  
  tprs = np.zeros((nrof_folds,nrof_thresholds))
  fprs = np.zeros((nrof_folds,nrof_thresholds))
  accuracy = np.zeros((nrof_folds))
  indices = np.arange(nrof_pairs)
  for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
      
    # Find the best threshold for the fold
    acc_train = np.zeros((nrof_thresholds))
    for threshold_idx, threshold in enumerate(thresholds):
        _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
    best_threshold_index = np.argmax(acc_train)
    for threshold_idx, threshold in enumerate(thresholds):
        tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
    _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
  acc2, std2 = np.mean(accuracy), np.std(accuracy)
  print('acc', acc2)

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='do verification')
  # general
  parser.add_argument('--data-dir', default='/raid5data/dplearn/YTF', help='')
  parser.add_argument('--model', default='../model/softmax,50', help='path to load model.')
  parser.add_argument('--gpu', default=0, type=int, help='gpu id')
  parser.add_argument('--batch-size', default=32, type=int, help='')
  parser.add_argument('--mode', default=1, type=int, help='')
  args = parser.parse_args()
  if args.mode>=2:
    main2(args)
  else:
    main(args)

