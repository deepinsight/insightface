# pylint: skip-file
import mxnet as mx
import numpy as np
import sys, os
import random
import glob
import math
import scipy.misc
import cv2
import logging
import sklearn
import datetime
import img_helper
from mxnet.io import DataIter
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
from PIL import Image
from config import config
from skimage import transform as tf


class FaceSegIter(DataIter):
    def __init__(self, path, batch_size, 
                 per_batch_size = 0,
                 aug_level = 0,
                 force_mirror = False,
                 exf = 1,
                 args = None):
      self.aug_level = aug_level
      self.force_mirror = force_mirror
      self.exf = exf
      self.batch_size = batch_size
      self.per_batch_size = per_batch_size
      self.image_file_list = []
      self.uv_file_list = []
      for _file in glob.glob(os.path.join(path, '*.jpg')):
        self.image_file_list.append(_file)
      for img in self.image_file_list:
        uv_file = img[0:-3]+"npy"
        self.uv_file_list.append(uv_file)
      self.seq = range(len(self.image_file_list))
      print('train size', len(self.seq))
      self.cur = 0
      self.reset()
      self.data_shape = (3, config.input_img_size, config.input_img_size)
      self.num_classes = config.num_classes
      self.input_img_size = config.input_img_size
      #self.label_classes = self.num_classes
      self.output_label_size = config.output_label_size
      #if aug_level>0:
      #  self.output_label_size = config.output_label_size
      #else:
      #  self.output_label_size = self.input_img_size
      self.label_shape = (self.num_classes, self.output_label_size, self.output_label_size)
      self.provide_data = [('data', (batch_size,) + self.data_shape)]
      self.provide_label = [('softmax_label', (batch_size,) + self.label_shape),
                            ('mask_label', (batch_size,)+ self.label_shape)]
      weight_mask = cv2.imread('./uv-data/uv_weight_mask.png')
      print('weight_mask', weight_mask.shape)
      if weight_mask.shape[0]!=self.output_label_size:
        weight_mask = cv2.resize(weight_mask, (self.output_label_size, self.output_label_size) )
      #idx = np.where(weight_mask>0)[0]
      #print('weight idx', idx)
      weight_mask = weight_mask.astype(np.float32)
      weight_mask /= 255.0

      vis_mask = cv2.imread('./uv-data/uv_face_mask.png')
      print('vis_mask', vis_mask.shape)
      if vis_mask.shape[0]!=self.output_label_size:
        vis_mask = cv2.resize(vis_mask, (self.output_label_size, self.output_label_size) )
      vis_mask = vis_mask.astype(np.float32)
      vis_mask /= 255.0
      weight_mask *= vis_mask
      print('weight_mask', weight_mask.shape)
      weight_mask = weight_mask.transpose( (2,0,1) )
      #WM = np.zeros( (batch_size,)+self.label_shape, dtype=np.float32 )
      #for i in range(batch_size):
      #  WM[i] = weight_mask
      #weight_mask = WM
      #weight_mask = weight_mask.reshape( (1, 3, weight_mask.shape[0], weight_mask.shape[1]) )
      weight_mask = weight_mask[np.newaxis,:,:,:]
      print('weight_mask', weight_mask.shape)
      weight_mask = np.tile(weight_mask, (batch_size,1,1,1))
      print('weight_mask', weight_mask.shape)
      self.weight_mask = nd.array(weight_mask)
      self.img_num = 0
      self.invalid_num = 0
      self.mode = 1
      self.vis = 0
      self.stats = [0,0]

    def get_data_shape(self):
        return self.data_shape

    #def get_label_shape(self):
    #    return self.label_shape

    def get_shape_dict(self):
        D = {}
        for (k,v) in self.provide_data:
            D[k] = v
        for (k,v) in self.provide_label:
            D[k] = v
        return D

    def get_label_names(self):
        D = []
        for (k,v) in self.provide_label:
            D.append(k)
        return D

    def reset(self):
      #print('reset')
      self.cur = 0
      if self.aug_level>0:
        random.shuffle(self.seq)

    def next_sample(self):
      """Helper function for reading in next sample."""
      if self.cur >= len(self.seq):
        raise StopIteration
      idx = self.seq[self.cur]
      self.cur += 1
      uv_path = self.uv_file_list[idx]
      image_path = self.image_file_list[idx]
      uvmap = np.load(uv_path)
      img = cv2.imread(image_path)[:,:,::-1]#to rgb
      hlabel = uvmap
      #print(hlabel.shape)
      #hlabel = np.array(header.label).reshape( (self.output_label_size, self.output_label_size, self.num_classes) )
      hlabel /= self.input_img_size

      return img, hlabel


    def next(self):
        """Returns the next batch of data."""
        #print('next')
        batch_size = self.batch_size
        batch_data = nd.empty((batch_size,)+self.data_shape)
        batch_label = nd.empty((batch_size,)+self.label_shape)
        i = 0
        #self.cutoff = random.randint(800,1280)
        try:
            while i < batch_size:
                #print('N', i)
                data, label = self.next_sample()
                data = nd.array(data)
                data = nd.transpose(data, axes=(2, 0, 1))
                label = nd.array(label)
                label = nd.transpose(label, axes=(2, 0, 1))
                batch_data[i][:] = data
                batch_label[i][:] = label
                i += 1
        except StopIteration:
            if i<batch_size:
                raise StopIteration

        #return {self.data_name  :  batch_data,
        #        self.label_name :  batch_label}
        #print(batch_data.shape, batch_label.shape)
        return mx.io.DataBatch([batch_data], [batch_label, self.weight_mask], batch_size - i)

