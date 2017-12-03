from __future__ import division

import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
import lfw
import sklearn
from sklearn.decomposition import PCA

lfw_dir = '/raid5data/dplearn/lfw_mtcnn2'
lfw_pairs = lfw.read_pairs(os.path.join(lfw_dir, 'pairs.txt'))
lfw_paths, issame_list = lfw.get_paths(lfw_dir, lfw_pairs, 'jpg')

model_dir = '../model'

models = ['sphereface_p0_20-lfw-0006.npy']
models = ['sphereface-64-p0_0_96_95_0-lfw-0001.npy']
models = ['sphereface_p0_20-lfw-0006.npy', 'sphereface-64-p0_0_96_95_0-lfw-0001.npy']
models = ['sphereface-20-p0_0_96_112_0-lfw-0022.npy','sphereface-20-p0_0_96_95_0-lfw-0021.npy', 'sphereface-20-p0_40_96_112_0-lfw-0022.npy']
models = ['sphereface-20-p0_0_96_112_0-lfw-0022.npy','sphereface-20-p0_0_96_95_0-lfw-0021.npy', 'sphereface-36-p0_0_96_112_0-lfw-0022.npy']
models = ['sphereface-20-p0_0_96_112_0-lfw-0022.npy','sphereface-20-p0_0_96_95_0-lfw-0021.npy']
models = ['sphereface-20-p0_0_96_112_0-lfw-0022.npy','sphereface-20-p0_0_96_95_0-lfw-0021.npy', 'sphereface-36-p0_0_96_112_0-lfw-0022.npy']
models = ['sphereface-20-p0_0_96_112_0-lfw-0022.npy','sphereface-20-p0_0_96_95_0-lfw-0021.npy', 'sphereface-36-p0_0_96_95_0-lfw-0021.npy']
models = [
          #'sphereface-20-p0_0_96_112_0-lfw-0022.npy',
          #'sphereface-20-p0_0_96_95_0-lfw-0021.npy', 
          #'sphereface-20-p0_0_80_95_0-lfw-0021.npy', 
          #'sphereface-36-p0_0_96_95_0-lfw-0021.npy',
          'sphereface-s60-p0_0_96_112_0-lfw-0031.npy',
          'sphereface-s60-p0_0_96_95_0-lfw-0021.npy',
          'sphereface2-s60-p0_0_96_112_0-lfw-0021.npy',
          'sphereface3-s60-p0_0_96_95_0-lfw-0023.npy',
          #'sphereface-s60-p0_0_80_95_0-lfw-0025.npy',
          #'sphereface-s60-p16_0_96_112_0-lfw-0023.npy',
          #'spherefacec-s60-p0_0_96_112_0-lfw-0021.npy',
          ]
models = [
          '../model31/sphere-m51-p0_0_96_112_0-lfw-0083.npy',
          '../model/softmax-m53-p0_0_96_112_0-lfw-0026.npy',
          #'../model32/sphere-m30-p0_0_96_112_0-lfw-0092.npy',
          ]
#models = models[0:3]
concat = True
pca = False
weights = None
#weights = [0.5, 1.0, 0.5]

F = None
ii = 0
for m in models:
  model = m
  #model = os.path.join(model_dir, m)
  X = np.load(model)
  X1 = X[0:(X.shape[0]//2),:]
  X2 = X[(X.shape[0]//2):,:]
  print(X.shape, X1.shape, X2.shape)
  #X1 = sklearn.preprocessing.normalize(X1)
  #X2 = sklearn.preprocessing.normalize(X2)
  XX = X1+X2
  XX = sklearn.preprocessing.normalize(XX)
  if weights is not None:
    weight = weights[ii]
    XX *= weight
  if F is None:
    F = XX
  else:
    if concat:
      F = np.concatenate((F,XX), axis=1)
    else:
      F += XX
  ii+=1
  #if concat:
  #  F = np.concatenate((F,X2), axis=1)
  #else:
  #  F += X2

print(F.shape)
npca = 0
if concat and pca:
  #F = sklearn.preprocessing.normalize(F)
  npca = 180
  #pca = PCA(n_components=512)
  #F = pca.fit_transform(F)
  for npca in xrange(512,513,1):
    _, _, accuracy, val, val_std, far = lfw.evaluate(F, issame_list, nrof_folds=10, pca=npca)
    print('[%d]Accuracy: %1.5f+-%1.5f' % (npca, np.mean(accuracy), np.std(accuracy)))
else:
  F = sklearn.preprocessing.normalize(F)
  _, _, accuracy, val, val_std, far = lfw.evaluate(F, issame_list, nrof_folds=10, pca=npca)
  print('[%d]Accuracy: %1.5f+-%1.5f' % (0, np.mean(accuracy), np.std(accuracy)))
  print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
