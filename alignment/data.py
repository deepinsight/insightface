# pylint: skip-file
import mxnet as mx
import numpy as np
import sys, os
import random
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


class FaceSegIter0(DataIter):
    def __init__(self, batch_size, 
                 path_imgrec = None,
                 data_name = "data",
                 label_name = "softmax_label"):
      self.batch_size = batch_size
      self.data_name = data_name
      self.label_name = label_name
      assert path_imgrec
      logging.info('loading recordio %s...',
                   path_imgrec)
      path_imgidx = path_imgrec[0:-4]+".idx"
      self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
      self.seq = list(self.imgrec.keys)
      self.cur = 0
      self.reset()
      self.num_classes = 68
      self.record_img_size = 384
      self.input_img_size = self.record_img_size
      self.data_shape = (3, self.input_img_size, self.input_img_size)
      self.label_shape = (self.num_classes, 2)
      self.provide_data = [(data_name, (batch_size,) + self.data_shape)]
      self.provide_label = [(label_name, (batch_size,) + self.label_shape)]

    def reset(self):
      print('reset')
      """Resets the iterator to the beginning of the data."""
      self.cur = 0

    def next_sample(self):
      """Helper function for reading in next sample."""
      if self.cur >= len(self.seq):
        raise StopIteration
      idx = self.seq[self.cur]
      self.cur += 1
      s = self.imgrec.read_idx(idx)
      header, img = recordio.unpack(s)
      img = mx.image.imdecode(img).asnumpy()
      #label = np.zeros( (self.num_classes, self.record_img_size, self.record_img_size), dtype=np.uint8)
      hlabel = np.array(header.label).reshape( (self.num_classes,2) )
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
                #print(data.shape, label.shape)
                batch_data[i][:] = data
                batch_label[i][:] = label
                i += 1
        except StopIteration:
            if not i:
                raise StopIteration

        return mx.io.DataBatch([batch_data], [batch_label], batch_size - i)

class FaceSegIter(DataIter):
    def __init__(self, batch_size, 
                 per_batch_size = 0,
                 path_imgrec = None,
                 aug_level = 0,
                 force_mirror = False,
                 use_coherent = 0,
                 args = None,
                 data_name = "data",
                 label_name = "softmax_label"):
      self.aug_level = aug_level
      self.force_mirror = force_mirror
      self.use_coherent = use_coherent
      self.batch_size = batch_size
      self.per_batch_size = per_batch_size
      self.data_name = data_name
      self.label_name = label_name
      assert path_imgrec
      logging.info('loading recordio %s...',
                   path_imgrec)
      path_imgidx = path_imgrec[0:-4]+".idx"
      self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
      self.seq = list(self.imgrec.keys)
      print('train size', len(self.seq))
      self.cur = 0
      self.reset()
      self.num_classes = args.num_classes
      self.record_img_size = 384
      self.input_img_size = args.input_img_size
      self.data_shape = (3, self.input_img_size, self.input_img_size)
      self.label_classes = self.num_classes
      if aug_level>0:
        self.output_label_size = args.output_label_size
        self.label_shape = (self.label_classes, self.output_label_size, self.output_label_size)
      else:
        self.output_label_size = args.input_img_size
        #self.label_shape = (self.num_classes, 2)
        self.label_shape = (self.num_classes, self.output_label_size, self.output_label_size)
      self.provide_data = [(data_name, (batch_size,) + self.data_shape)]
      self.provide_label = [(label_name, (batch_size,) + self.label_shape)]
      if self.use_coherent>0:
          #self.provide_label += [("softmax_label2", (batch_size,)+self.label_shape)]
          self.provide_label += [("coherent_label", (batch_size,6))]
      self.img_num = 0
      self.invalid_num = 0
      self.mode = 1
      self.vis = 0
      self.stats = [0,0]
      self.mirror_set = [
              (22,23),
              (21,24),
              (20,25),
              (19,26),
              (18,27),
              (40,43),
              (39,44),
              (38,45),
              (37,46),
              (42,47),
              (41,48),
              (33,35),
              (32,36),
              (51,53),
              (50,54),
              (62,64),
              (61,65),
              (49,55),
              (49,55),
              (68,66),
              (60,56),
              (59,57),
              (1,17),
              (2,16),
              (3,15),
              (4,14),
              (5,13),
              (6,12),
              (7,11),
              (8,10),
          ]

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
      print('reset')
      """Resets the iterator to the beginning of the data."""
      if self.aug_level>0:
        random.shuffle(self.seq)
      self.cur = 0

    def next_sample(self):
      """Helper function for reading in next sample."""
      if self.cur >= len(self.seq):
        raise StopIteration
      idx = self.seq[self.cur]
      self.cur += 1
      s = self.imgrec.read_idx(idx)
      header, img = recordio.unpack(s)
      img = mx.image.imdecode(img).asnumpy()
      #label = np.zeros( (self.num_classes, self.record_img_size, self.record_img_size), dtype=np.uint8)
      hlabel = np.array(header.label).reshape( (self.num_classes,2) )
      annot = {}
      ul = np.array( (50000,50000), dtype=np.int32)
      br = np.array( (0,0), dtype=np.int32)
      for i in xrange(hlabel.shape[0]):
        #hlabel[i] = hlabel[i][::-1]
        h = int(hlabel[i][0])
        w = int(hlabel[i][1])
        key = np.array((h,w))
        #print(key.shape, ul.shape, br.shape)
        ul = np.minimum(key, ul)
        br = np.maximum(key, br)
        #label[h][w] = i+1
        #label[i][h][w] = 1.0
        #print(h,w,i+1)

      if self.mode==0:
          ul_margin = np.array( (60,30) )
          br_margin = np.array( (30,30) )
          crop_ul = ul
          crop_ul = ul-ul_margin
          crop_ul = np.maximum(crop_ul, np.array( (0,0), dtype=np.int32) )
          crop_br = br
          crop_br = br+br_margin
          crop_br = np.minimum(crop_br, np.array( (img.shape[0],img.shape[1]), dtype=np.int32 ) )
      elif self.mode==1:
          #mm = (self.record_img_size - 256)//2
          #crop_ul = (mm, mm)
          #crop_br = (self.record_img_size-mm, self.record_img_size-mm)
          crop_ul = (0,0)
          crop_br = (self.record_img_size, self.record_img_size)
          annot['scale'] = 256
          #annot['center'] = np.array( (self.record_img_size/2+10, self.record_img_size/2) )
      else:
          mm = (48, 64)
          #mm = (64, 80)
          crop_ul = mm
          crop_br = (self.record_img_size-mm[0], self.record_img_size-mm[1])
      crop_ul = np.array(crop_ul)
      crop_br = np.array(crop_br)
      img = img[crop_ul[0]:crop_br[0],crop_ul[1]:crop_br[1],:]
      #print(img.shape, crop_ul, crop_br)
      invalid = False
      for i in xrange(hlabel.shape[0]):
          if (hlabel[i]<crop_ul).any() or (hlabel[i] >= crop_br).any():
              invalid = True
          hlabel[i] -= crop_ul

      #mm = np.amin(ul)
      #mm2 = self.record_img_size - br
      #mm2 = np.amin(mm2)
      #mm = min(mm, mm2)
      #print('mm',mm, ul, br)
      #print('invalid', invalid)
      if invalid:
          self.invalid_num+=1

      annot['invalid'] = invalid
      #annot['scale'] = (self.record_img_size - mm*2)*1.1
      #if self.mode==1:
      #    annot['scale'] = 256
      #print(annot)

      #center = ul+br
      #center /= 2.0
      #annot['center'] = center

      #img = img[ul[0]:br[0],ul[1]:br[1],:]
      #scale = br-ul
      #scale = max(scale[0], scale[1])
      #print(img.shape)
      return img, hlabel, annot

    def do_aug(self, data, label, annot):
      if self.vis:
        self.img_num+=1
        #if self.img_num<=self.vis:
        #  filename = './vis/raw_%d.jpg' % (self.img_num)
        #  print('save', filename)
        #  draw = data.copy()
        #  for i in xrange(label.shape[0]):
        #    cv2.circle(draw, (label[i][1], label[i][0]), 1, (0, 0, 255), 2)
        #  scipy.misc.imsave(filename, draw)

      rotate = 0
      #scale = 1.0
      if 'scale' in annot:
          scale = annot['scale']
      else:
          scale = max(data.shape[0], data.shape[1])
      if 'center' in annot:
        center = annot['center']
      else:
        center = np.array( (data.shape[0]/2, data.shape[1]/2) )
      max_retry = 3
      if self.aug_level==0:
          max_retry = 6
      retry = 0
      found = False
      _scale = scale
      while retry<max_retry:
          retry+=1
          succ = True
          if self.aug_level>0:
            rotate = np.random.randint(-40, 40)
            #rotate2 = np.random.randint(-40, 40)
            rotate2 = 0
            scale_config = 0.2
            #rotate = 0
            #scale_config = 0.0
            _scale = min(1+scale_config, max(1-scale_config, (np.random.randn() * scale_config) + 1))
            _scale *= scale
            _scale = int(_scale)
            #translate = np.random.randint(-5, 5, size=(2,))
            #center += translate
          if self.mode==1:
              cropped = img_helper.crop2(data, center, _scale, (self.input_img_size, self.input_img_size), rot=rotate)
              if self.use_coherent==2:
                cropped2 = img_helper.crop2(data, center, _scale, (self.input_img_size, self.input_img_size), rot=rotate2)
          else:
              cropped = img_helper.crop(data, center, _scale, (self.input_img_size, self.input_img_size), rot=rotate)
          #print('cropped', cropped.shape)
          label_out = np.zeros(self.label_shape, dtype=np.float32)
          label2_out = np.zeros(self.label_shape, dtype=np.float32)
          G = 0
          #if self.use_coherent:
          #    G = 1
          _g = G
          if G==0:
              _g = 1
          #print('shape', label.shape, label_out.shape)
          for i in xrange(label.shape[0]):
            pt = label[i].copy()
            pt = pt[::-1]
            #print('before gaussian', label_out[i].shape, pt.shape)
            _pt = pt.copy()
            trans = img_helper.transform(_pt, center, _scale, (self.output_label_size, self.output_label_size), rot=rotate)
            #print(trans.shape)
            if not img_helper.gaussian(label_out[i], trans, _g):
                succ = False
                break
            if self.use_coherent==2:
                _pt = pt.copy()
                trans2 = img_helper.transform(_pt, center, _scale, (self.output_label_size, self.output_label_size), rot=rotate2)
                if not img_helper.gaussian(label2_out[i], trans2, _g):
                    succ = False
                    break
          if not succ:
              if self.aug_level==0:
                  _scale+=20
              continue

          
          if self.use_coherent==1:
            cropped2 = np.copy(cropped)
            for k in xrange(cropped2.shape[2]):
                cropped2[:,:,k] = np.fliplr(cropped2[:,:,k])
            label2_out = np.copy(label_out)
            for k in xrange(label2_out.shape[0]):
                label2_out[k,:,:] = np.fliplr(label2_out[k,:,:])
            new_label2_out = np.copy(label2_out)
            for item in self.mirror_set:
                mir = (item[0]-1, item[1]-1)
                new_label2_out[mir[1]] = label2_out[mir[0]]
                new_label2_out[mir[0]] = label2_out[mir[1]]
            label2_out = new_label2_out
          elif self.use_coherent==2:
            pass
          elif ((self.aug_level>0 and np.random.rand() < 0.5) or self.force_mirror): #flip aug
              for k in xrange(cropped.shape[2]):
                  cropped[:,:,k] = np.fliplr(cropped[:,:,k])
              for k in xrange(label_out.shape[0]):
                  label_out[k,:,:] = np.fliplr(label_out[k,:,:])
              new_label_out = np.copy(label_out)
              for item in self.mirror_set:
                  mir = (item[0]-1, item[1]-1)
                  new_label_out[mir[1]] = label_out[mir[0]]
                  new_label_out[mir[0]] = label_out[mir[1]]
              label_out = new_label_out

          if G==0:
              for k in xrange(label.shape[0]):
                  ind = np.unravel_index(np.argmax(label_out[k], axis=None), label_out[k].shape)
                  label_out[k,:,:] = 0.0
                  label_out[k,ind[0],ind[1]] = 1.0
                  if self.use_coherent:
                      ind = np.unravel_index(np.argmax(label2_out[k], axis=None), label2_out[k].shape)
                      label2_out[k,:,:] = 0.0
                      label2_out[k,ind[0],ind[1]] = 1.0
          found = True
          break


      #self.stats[0]+=1
      if not found:
          #self.stats[1]+=1
          #print('find aug error', retry)
          #print(self.stats)
          return None

      if self.vis>0 and self.img_num<=self.vis:
        print('crop', data.shape, center, _scale, rotate, cropped.shape)
        filename = './vis/cropped_%d.jpg' % (self.img_num)
        print('save', filename)
        draw = cropped.copy()
        alabel = label_out.copy()
        for i in xrange(label.shape[0]):
          a = cv2.resize(alabel[i], (self.input_img_size, self.input_img_size))
          ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
          cv2.circle(draw, (ind[1], ind[0]), 1, (0, 0, 255), 2)
        scipy.misc.imsave(filename, draw)
      if not self.use_coherent:
          return cropped, label_out
      else:
          rotate2 = 0
          r = rotate - rotate2
          #r = rotate2 - rotate
          r = math.pi*r/180
          cos_r = math.cos(r)
          sin_r = math.sin(r)
          #c = cropped2.shape[0]//2
          #M = cv2.getRotationMatrix2D( (c, c), rotate2-rotate, 1)
          M = np.array( [ 
              [cos_r, -1*sin_r, 0.0], 
              [sin_r, cos_r, 0.0]
              ] )
          #print(M)
          #M=np.array([16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
          #    26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35, 34, 33,
          #    32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52,
          #    51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63, 62, 61, 60, 67, 66, 65])
          return cropped, label_out, cropped2, label2_out, M.flatten()

    def next(self):
        """Returns the next batch of data."""
        #print('next')
        batch_size = self.batch_size
        batch_data = nd.empty((batch_size,)+self.data_shape)
        batch_label = nd.empty((batch_size,)+self.label_shape)
        if self.use_coherent:
            batch_label2 = nd.empty((batch_size,)+self.label_shape)
            batch_coherent_label = nd.empty((batch_size,6))
        i = 0
        #self.cutoff = random.randint(800,1280)
        try:
            while i < batch_size:
                #print('N', i)
                data, label, annot = self.next_sample()
                if not self.use_coherent:
                    R = self.do_aug(data, label, annot)
                    if R is None:
                        continue
                    data, label = R
                    #data, label, data2, label2, M = R
                    #ind = np.unravel_index(np.argmax(label[0], axis=None), label[0].shape)
                    #print(label.shape, np.count_nonzero(label[0]), ind)
                    #print(label[0,25:35,0:10])
                    data = nd.array(data)
                    data = nd.transpose(data, axes=(2, 0, 1))
                    label = nd.array(label)
                    #print(data.shape, label.shape)
                    try:
                        self.check_valid_image(data)
                    except RuntimeError as e:
                        logging.debug('Invalid image, skipping:  %s', str(e))
                        continue
                    batch_data[i][:] = data
                    batch_label[i][:] = label
                    i += 1
                else:
                    R = self.do_aug(data, label, annot)
                    if R is None:
                        continue
                    data, label, data2, label2, M = R
                    data = nd.array(data)
                    data = nd.transpose(data, axes=(2, 0, 1))
                    label = nd.array(label)
                    data2 = nd.array(data2)
                    data2 = nd.transpose(data2, axes=(2, 0, 1))
                    label2 = nd.array(label2)
                    M = nd.array(M)
                    #print(data.shape, label.shape)
                    try:
                        self.check_valid_image(data)
                    except RuntimeError as e:
                        logging.debug('Invalid image, skipping:  %s', str(e))
                        continue
                    batch_data[i][:] = data
                    batch_label[i][:] = label
                    #batch_label2[i][:] = label2
                    batch_coherent_label[i][:] = M
                    #i+=1
                    j = i+self.per_batch_size//2
                    batch_data[j][:] = data2
                    batch_label[j][:] = label2
                    batch_coherent_label[j][:] = M
                    i += 1
                    if j%self.per_batch_size==self.per_batch_size-1:
                        i = j+1
        except StopIteration:
            if not i:
                raise StopIteration

        #return {self.data_name  :  batch_data,
        #        self.label_name :  batch_label}
        #print(batch_data.shape, batch_label.shape)
        if not self.use_coherent:
            return mx.io.DataBatch([batch_data], [batch_label], batch_size - i)
        else:
            #return mx.io.DataBatch([batch_data], [batch_label, batch_label2, batch_coherent_label], batch_size - i)
            return mx.io.DataBatch([batch_data], [batch_label, batch_coherent_label], batch_size - i)

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        return imdecode(s)

    def read_image(self, fname):
        """Reads an input image `fname` and returns the decoded raw bytes.

        Example usage:
        ----------
        >>> dataIter.read_image('Face.jpg') # returns decoded raw bytes.
        """
        with open(os.path.join(self.path_root, fname), 'rb') as fin:
            img = fin.read()
        return img


