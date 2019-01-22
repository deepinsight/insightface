"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
from sklearn.decomposition import PCA
import mxnet as mx
from mxnet import ndarray as nd



def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca = 0):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    #print('pca', pca)
    
    if pca==0:
      diff = np.subtract(embeddings1, embeddings2)
      dist = np.sum(np.square(diff),1)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        #print('train_set', train_set)
        #print('test_set', test_set)
        if pca>0:
          print('doing pca on', fold_idx)
          embed1_train = embeddings1[train_set]
          embed2_train = embeddings2[train_set]
          _embed_train = np.concatenate( (embed1_train, embed2_train), axis=0 )
          #print(_embed_train.shape)
          pca_model = PCA(n_components=pca)
          pca_model.fit(_embed_train)
          embed1 = pca_model.transform(embeddings1)
          embed2 = pca_model.transform(embeddings2)
          embed1 = sklearn.preprocessing.normalize(embed1)
          embed2 = sklearn.preprocessing.normalize(embed2)
          #print(embed1.shape, embed2.shape)
          diff = np.subtract(embed1, embed2)
          dist = np.sum(np.square(diff),1)
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
          
    tpr = np.mean(tprs,0)
    fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc


  
def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
      
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
    
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
  
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def evaluate(embeddings, actual_issame, nrof_folds=10, pca = 0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, pca = pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

def get_paths(lfw_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            print('not exists', path0, path1)
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


def load_dataset(lfw_dir, image_size):
  lfw_pairs = read_pairs(os.path.join(lfw_dir, 'pairs.txt'))
  lfw_paths, issame_list = get_paths(lfw_dir, lfw_pairs, 'jpg')
  lfw_data_list = []
  for flip in [0,1]:
    lfw_data = nd.empty((len(lfw_paths), 3, image_size[0], image_size[1]))
    lfw_data_list.append(lfw_data)
  i = 0
  for path in lfw_paths:
    with open(path, 'rb') as fin:
      _bin = fin.read()
      img = mx.image.imdecode(_bin)
      img = nd.transpose(img, axes=(2, 0, 1))
      for flip in [0,1]:
        if flip==1:
          img = mx.ndarray.flip(data=img, axis=2)
        lfw_data_list[flip][i][:] = img
      i+=1
      if i%1000==0:
        print('loading lfw', i)
  print(lfw_data_list[0].shape)
  print(lfw_data_list[1].shape)
  return (lfw_data_list, issame_list)

def test(lfw_set, mx_model, batch_size):
  print('testing lfw..')
  lfw_data_list = lfw_set[0]
  issame_list = lfw_set[1]
  model = mx_model
  embeddings_list = []
  for i in xrange( len(lfw_data_list) ):
    lfw_data = lfw_data_list[i]
    embeddings = None
    ba = 0
    while ba<lfw_data.shape[0]:
      bb = min(ba+batch_size, lfw_data.shape[0])
      _data = nd.slice_axis(lfw_data, axis=0, begin=ba, end=bb)
      _label = nd.ones( (bb-ba,) )
      #print(_data.shape, _label.shape)
      db = mx.io.DataBatch(data=(_data,), label=(_label,))
      model.forward(db, is_train=False)
      net_out = model.get_outputs()
      #_arg, _aux = model.get_params()
      #__arg = {}
      #for k,v in _arg.iteritems():
      #  __arg[k] = v.as_in_context(_ctx)
      #_arg = __arg
      #_arg["data"] = _data.as_in_context(_ctx)
      #_arg["softmax_label"] = _label.as_in_context(_ctx)
      #for k,v in _arg.iteritems():
      #  print(k,v.context)
      #exe = sym.bind(_ctx, _arg ,args_grad=None, grad_req="null", aux_states=_aux)
      #exe.forward(is_train=False)
      #net_out = exe.outputs
      _embeddings = net_out[0].asnumpy()
      #print(_embeddings.shape)
      if embeddings is None:
        embeddings = np.zeros( (lfw_data.shape[0], _embeddings.shape[1]) )
      embeddings[ba:bb,:] = _embeddings
      ba = bb
    embeddings_list.append(embeddings)

  _xnorm = 0.0
  _xnorm_cnt = 0
  for embed in embeddings_list:
    for i in xrange(embed.shape[0]):
      _em = embed[i]
      _norm=np.linalg.norm(_em)
      #print(_em.shape, _norm)
      _xnorm+=_norm
      _xnorm_cnt+=1
  _xnorm /= _xnorm_cnt

  embeddings = embeddings_list[0].copy()
  embeddings = sklearn.preprocessing.normalize(embeddings)
  _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=10)
  acc1, std1 = np.mean(accuracy), np.std(accuracy)
  #print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
  #embeddings = np.concatenate(embeddings_list, axis=1)
  embeddings = embeddings_list[0] + embeddings_list[1]
  embeddings = sklearn.preprocessing.normalize(embeddings)
  print(embeddings.shape)
  _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=10)
  acc2, std2 = np.mean(accuracy), np.std(accuracy)
  return acc1, std1, acc2, std2, _xnorm, embeddings_list

