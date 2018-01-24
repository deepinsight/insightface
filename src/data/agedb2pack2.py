from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#import mxnet as mx
#from mxnet import ndarray as nd
import argparse
import cv2
import pickle
import numpy as np
import sys
from scipy import misc
import os
import tensorflow as tf
from scipy.io import loadmat
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'align'))
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import detect_face
import face_image
import face_preprocess
#import lfw

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def IOU(Reframe,GTframe):
  x1 = Reframe[0];
  y1 = Reframe[1];
  width1 = Reframe[2]-Reframe[0];
  height1 = Reframe[3]-Reframe[1];

  x2 = GTframe[0]
  y2 = GTframe[1]
  width2 = GTframe[2]-GTframe[0]
  height2 = GTframe[3]-GTframe[1]

  endx = max(x1+width1,x2+width2)
  startx = min(x1,x2)
  width = width1+width2-(endx-startx)

  endy = max(y1+height1,y2+height2)
  starty = min(y1,y2)
  height = height1+height2-(endy-starty)

  if width <=0 or height <= 0:
    ratio = 0
  else:
    Area = width*height
    Area1 = width1*height1
    Area2 = width2*height2
    ratio = Area*1./(Area1+Area2-Area)
  return ratio

parser = argparse.ArgumentParser(description='Package AgeDB images')
# general
parser.add_argument('--data-dir', default='', help='')
parser.add_argument('--image-size', type=str, default='112,96', help='')
parser.add_argument('--output', default='./', help='path to save.')
args = parser.parse_args()


for part in [ ('04_FINAL_protocol_30_years.mat', 'agedb_30') ]:
  mat_file = os.path.join(args.data_dir, part[0])
  mat_data = loadmat(mat_file)
  print(mat_data.__class__)
  data = mat_data['splits']

  bins = []
  issame_list = []
  nrof = [0, 0, 0]
  print('processing', part[1])
  pp = 0
  for i in xrange(data.shape[0]):
    split = data[i][0][0][0][0]
    print(split.shape)
    for c in xrange(split.shape[1]):
      last_name = ''
      for r in xrange(split.shape[0]):
        pp+=1
        if pp%10==0:
          print('processing', pp, nrof)
        item = split[r][c][0][0]
        path = str(item[0][0])
        vec = path.split('_')
        assert len(vec)>=5
        name = vec[0]
        if r==1:
          issame = False
          if name==last_name:
            issame = True
          #print(issame)
          issame_list.append(issame)
        last_name = name
        age = int(item[1])
        #print(path, age)
        #sys.exit(0)
        img_path = os.path.join(args.data_dir, '03_Protocol_Images', path+".jpg")
        #print(img_path)
        img = misc.imread(img_path)
        if img.ndim == 2:
          img = to_rgb(img)
        assert img.ndim==3
        assert img.shape[2]==3
        #img = img[:,:,0:3]
        all_landmark = np.zeros( (68,2), dtype=np.float32)
        pts_file = img_path[0:-3]+"pts"
        pp=0

        for line in open(pts_file, 'r'):
          pp+=1
          pointid = pp-3
          if pointid<1 or pointid>68:
            continue
          point = [float(x) for x in line.strip().split()]
          assert len(point)==2
          point = np.array(point).reshape( (1,2) )
          #print(pointid)
          all_landmark[pointid-1,:] = point

        
        _landmark = np.zeros( (5,2), dtype=np.float32)
        _landmark[0,:] = (all_landmark[36,:]+all_landmark[39,:])/2
        _landmark[1,:] = (all_landmark[42,:]+all_landmark[45,:])/2
        _landmark[2,:] = all_landmark[33,:]
        _landmark[3,:] = all_landmark[48,:]
        _landmark[4,:] = all_landmark[54,:]
        _bbox = None
        warped = face_preprocess.preprocess(img, bbox=_bbox, landmark = _landmark, image_size=args.image_size)
        warped = warped[...,::-1] #to bgr
        _, s = cv2.imencode('.jpg', warped)
        bins.append(s)
  print(nrof)
  outname = os.path.join(args.output, part[1]+'.bin')
  with open(outname, 'wb') as f:
    pickle.dump((bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)





