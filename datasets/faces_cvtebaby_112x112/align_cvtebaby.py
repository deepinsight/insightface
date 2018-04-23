# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys

#curr_path = os.path.abspath(os.path.dirname(__file__))
#sys.path.append(os.path.join(curr_path, "../python"))
import mxnet as mx
import random
import argparse
import cv2
import time
import traceback
#from builtins import range
from easydict import EasyDict as edict
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src/common'))
import face_preprocess
import numpy as np


try:
    import multiprocessing
except ImportError:
    multiprocessing = None



def read_list(path_in):
    with open(path_in) as fin:
        identities = []
        last = [-1, -1]
        _id = 1
        while True:
            line = fin.readline()
            if not line:
                break
            item = edict()
            item.flag = 0
            item.image_path, label, item.bbox, item.landmark, item.aligned = face_preprocess.parse_lst_line(line)
            if not item.aligned and item.landmark is None:
              #print('ignore line', line)
              continue
            item.id = _id
            item.label = [label, item.aligned]
            yield item
            if label!=last[0]:
              if last[1]>=0:
                identities.append( (last[1], _id) )
              last[0] = label
              last[1] = _id
            _id+=1
        identities.append( (last[1], _id) )
        item = edict()
        item.flag = 2
        item.id = 0
        item.label = [float(_id), float(_id+len(identities))]
        yield item
        for identity in identities:
          item = edict()
          item.flag = 2
          item.id = _id
          _id+=1
          item.label = [float(identity[0]), float(identity[1])]
          yield item



def image_encode(args, i, item, q_out):
    oitem = [item.id]
    #print('flag', item.flag)
    if item.flag==0:
      fullpath = item.image_path
      header = mx.recordio.IRHeader(item.flag, item.label, item.id, 0)
      #print('write', item.flag, item.id, item.label)
      if item.aligned:
        with open(fullpath, 'rb') as fin:
            img = fin.read()
        s = mx.recordio.pack(header, img)
        q_out.put((i, s, oitem))
      else:
        img = cv2.imread(fullpath, args.color)
        assert item.landmark is not None
        img = face_preprocess.preprocess(img, bbox = item.bbox, landmark=item.landmark, image_size='%d,%d'%(args.image_h, args.image_w))
        s = mx.recordio.pack_img(header, img, quality=args.quality, img_fmt=args.encoding)
        q_out.put((i, s, oitem))
    else:
      header = mx.recordio.IRHeader(item.flag, item.label, item.id, 0)
      #print('write', item.flag, item.id, item.label)
      s = mx.recordio.pack(header, '')
      q_out.put((i, s, oitem))




if __name__ == '__main__':
    image_h, image_w = (112, 112)

    fn_landmark = '/data/victor/cvte_baby/valid/landmark.txt'
    cvte_dir = '/data/victor/cvte_baby/valid/CVTE_baby_valid'
    cvte_dir_aligned = '/data/victor/cvte_baby/valid/CVTE_baby_valid_aligned'


    with open(fn_landmark, 'r') as f:
        lines = map(lambda s: s.strip().split('\t'), f.readlines())
        for line in lines:
            image_path = line[1]
            sub_folder, fn = image_path.split('/')[-2:]
            landmark = np.array(line[7:]).astype(np.float32).reshape((5,2))

            img = cv2.imread(image_path)
            img = face_preprocess.preprocess(img, bbox=None, landmark=landmark,
                                             image_size='%d,%d' % (image_h, image_w))

            dst_folder = os.path.join(cvte_dir_aligned, sub_folder)
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
            cv2.imwrite(os.path.join(dst_folder, fn), img)

