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
from config import config
from skimage import transform as tf


class FaceSegIter(DataIter):
    def __init__(self,
                 batch_size,
                 per_batch_size=0,
                 path_imgrec=None,
                 aug_level=0,
                 force_mirror=False,
                 exf=1,
                 use_coherent=0,
                 args=None,
                 data_name="data",
                 label_name="softmax_label"):
        self.aug_level = aug_level
        self.force_mirror = force_mirror
        self.use_coherent = use_coherent
        self.exf = exf
        self.batch_size = batch_size
        self.per_batch_size = per_batch_size
        self.data_name = data_name
        self.label_name = label_name
        assert path_imgrec
        logging.info('loading recordio %s...', path_imgrec)
        path_imgidx = path_imgrec[0:-4] + ".idx"
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec,
                                                    'r')  # pylint: disable=redefined-variable-type
        self.oseq = list(self.imgrec.keys)
        print('train size', len(self.oseq))
        self.cur = 0
        self.reset()
        self.data_shape = (3, config.input_img_size, config.input_img_size)
        self.num_classes = config.num_classes
        self.input_img_size = config.input_img_size
        #self.label_classes = self.num_classes
        if config.losstype == 'heatmap':
            if aug_level > 0:
                self.output_label_size = config.output_label_size
                self.label_shape = (self.num_classes, self.output_label_size,
                                    self.output_label_size)
            else:
                self.output_label_size = self.input_img_size
                #self.label_shape = (self.num_classes, 2)
                self.label_shape = (self.num_classes, self.output_label_size,
                                    self.output_label_size)
        else:
            if aug_level > 0:
                self.output_label_size = config.output_label_size
                self.label_shape = (self.num_classes, 2)
            else:
                self.output_label_size = self.input_img_size
                #self.label_shape = (self.num_classes, 2)
                self.label_shape = (self.num_classes, 2)
        self.provide_data = [(data_name, (batch_size, ) + self.data_shape)]
        self.provide_label = [(label_name, (batch_size, ) + self.label_shape)]
        self.img_num = 0
        self.invalid_num = 0
        self.mode = 1
        self.vis = 0
        self.stats = [0, 0]
        self.flip_order = [
            16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25,
            24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35, 34, 33, 32, 31,
            45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51, 50,
            49, 48, 59, 58, 57, 56, 55, 64, 63, 62, 61, 60, 67, 66, 65
        ]
        #self.mirror_set = [
        #        (22,23),
        #        (21,24),
        #        (20,25),
        #        (19,26),
        #        (18,27),
        #        (40,43),
        #        (39,44),
        #        (38,45),
        #        (37,46),
        #        (42,47),
        #        (41,48),
        #        (33,35),
        #        (32,36),
        #        (51,53),
        #        (50,54),
        #        (62,64),
        #        (61,65),
        #        (49,55),
        #        (49,55),
        #        (68,66),
        #        (60,56),
        #        (59,57),
        #        (1,17),
        #        (2,16),
        #        (3,15),
        #        (4,14),
        #        (5,13),
        #        (6,12),
        #        (7,11),
        #        (8,10),
        #    ]

    def get_data_shape(self):
        return self.data_shape

    #def get_label_shape(self):
    #    return self.label_shape

    def get_shape_dict(self):
        D = {}
        for (k, v) in self.provide_data:
            D[k] = v
        for (k, v) in self.provide_label:
            D[k] = v
        return D

    def get_label_names(self):
        D = []
        for (k, v) in self.provide_label:
            D.append(k)
        return D

    def reset(self):
        #print('reset')
        if self.aug_level == 0:
            self.seq = self.oseq
        else:
            self.seq = []
            for _ in range(self.exf):
                _seq = self.oseq[:]
                random.shuffle(_seq)
                self.seq += _seq
            print('train size after reset', len(self.seq))
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
        hlabel = np.array(header.label).reshape((self.num_classes, 2))
        if not config.label_xfirst:
            hlabel = hlabel[:, ::-1]  #convert to X/W first
        annot = {'scale': config.base_scale}

        #ul = np.array( (50000,50000), dtype=np.int32)
        #br = np.array( (0,0), dtype=np.int32)
        #for i in range(hlabel.shape[0]):
        #  h = int(hlabel[i][0])
        #  w = int(hlabel[i][1])
        #  key = np.array((h,w))
        #  ul = np.minimum(key, ul)
        #  br = np.maximum(key, br)

        return img, hlabel, annot

    def get_flip(self, data, label):
        data_flip = np.zeros_like(data)
        label_flip = np.zeros_like(label)
        for k in range(data_flip.shape[2]):
            data_flip[:, :, k] = np.fliplr(data[:, :, k])
        for k in range(label_flip.shape[0]):
            label_flip[k, :] = np.fliplr(label[k, :])
        #print(label[0,:].shape)
        label_flip = label_flip[self.flip_order, :]
        return data_flip, label_flip

    def get_data(self, data, label, annot):
        if self.vis:
            self.img_num += 1
            #if self.img_num<=self.vis:
            #  filename = './vis/raw_%d.jpg' % (self.img_num)
            #  print('save', filename)
            #  draw = data.copy()
            #  for i in range(label.shape[0]):
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
            center = np.array((data.shape[1] / 2, data.shape[0] / 2))
        max_retry = 3
        if self.aug_level == 0:  #validation mode
            max_retry = 6
        retry = 0
        found = False
        base_scale = scale
        while retry < max_retry:
            retry += 1
            succ = True
            _scale = base_scale
            if self.aug_level > 0:
                rotate = np.random.randint(-40, 40)
                scale_config = 0.2
                #rotate = 0
                #scale_config = 0.0
                scale_ratio = min(
                    1 + scale_config,
                    max(1 - scale_config,
                        (np.random.randn() * scale_config) + 1))
                _scale = int(base_scale * scale_ratio)
                #translate = np.random.randint(-5, 5, size=(2,))
                #center += translate
            data_out, trans = img_helper.transform(data, center,
                                                   self.input_img_size, _scale,
                                                   rotate)
            #data_out = img_helper.crop2(data, center, _scale, (self.input_img_size, self.input_img_size), rot=rotate)
            label_out = np.zeros(self.label_shape, dtype=np.float32)
            #print('out shapes', data_out.shape, label_out.shape)
            for i in range(label.shape[0]):
                pt = label[i].copy()
                #pt = pt[::-1]
                npt = img_helper.transform_pt(pt, trans)
                if npt[0] >= data_out.shape[1] or npt[1] >= data_out.shape[
                        0] or npt[0] < 0 or npt[1] < 0:
                    succ = False
                    #print('err npt', npt)
                    break
                if config.losstype == 'heatmap':
                    pt_scale = float(
                        self.output_label_size) / self.input_img_size
                    npt *= pt_scale
                    npt = npt.astype(np.int32)
                    img_helper.gaussian(label_out[i], npt, config.gaussian)
                else:
                    label_out[i] = (npt / self.input_img_size)
                #print('before gaussian', label_out[i].shape, pt.shape)
                #trans = img_helper.transform(pt, center, _scale, (self.output_label_size, self.output_label_size), rot=rotate)
                #print(trans.shape)
                #if not img_helper.gaussian(label_out[i], trans, _g):
                #    succ = False
                #    break
            if not succ:
                if self.aug_level == 0:
                    base_scale += 20
                continue

            flip_data_out = None
            flip_label_out = None
            if config.net_coherent:
                flip_data_out, flip_label_out = self.get_flip(
                    data_out, label_out)
            elif ((self.aug_level > 0 and np.random.rand() < 0.5)
                  or self.force_mirror):  #flip aug
                flip_data_out, flip_label_out = self.get_flip(
                    data_out, label_out)
                data_out, label_out = flip_data_out, flip_label_out

            found = True
            break

        #self.stats[0]+=1
        if not found:
            #self.stats[1]+=1
            #print('find aug error', retry)
            #print(self.stats)
            #print('!!!ERR')
            return None
        #print('found with scale', _scale, rotate)

        if self.vis > 0 and self.img_num <= self.vis:
            print('crop', data.shape, center, _scale, rotate, data_out.shape)
            filename = './vis/cropped_%d.jpg' % (self.img_num)
            print('save', filename)
            draw = data_out.copy()
            alabel = label_out.copy()
            for i in range(label.shape[0]):
                a = cv2.resize(alabel[i],
                               (self.input_img_size, self.input_img_size))
                ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
                cv2.circle(draw, (ind[1], ind[0]), 1, (0, 0, 255), 2)
            scipy.misc.imsave(filename, draw)
            filename = './vis/raw_%d.jpg' % (self.img_num)
            scipy.misc.imsave(filename, data)

        return data_out, label_out, flip_data_out, flip_label_out

    def next(self):
        """Returns the next batch of data."""
        #print('next')
        batch_size = self.batch_size
        batch_data = nd.empty((batch_size, ) + self.data_shape)
        batch_label = nd.empty((batch_size, ) + self.label_shape)
        i = 0
        #self.cutoff = random.randint(800,1280)
        try:
            while i < batch_size:
                #print('N', i)
                data, label, annot = self.next_sample()
                R = self.get_data(data, label, annot)
                if R is None:
                    continue
                data_out, label_out, flip_data_out, flip_label_out = R
                if not self.use_coherent:
                    data = nd.array(data_out)
                    data = nd.transpose(data, axes=(2, 0, 1))
                    label = nd.array(label_out)
                    #print(data.shape, label.shape)
                    batch_data[i][:] = data
                    batch_label[i][:] = label
                    i += 1
                else:
                    data = nd.array(data_out)
                    data = nd.transpose(data, axes=(2, 0, 1))
                    label = nd.array(label_out)
                    data2 = nd.array(flip_data_out)
                    data2 = nd.transpose(data2, axes=(2, 0, 1))
                    label2 = nd.array(flip_label_out)
                    #M = nd.array(M)
                    #print(data.shape, label.shape)
                    batch_data[i][:] = data
                    batch_label[i][:] = label
                    #i+=1
                    j = i + self.per_batch_size // 2
                    batch_data[j][:] = data2
                    batch_label[j][:] = label2
                    i += 1
                    if j % self.per_batch_size == self.per_batch_size - 1:
                        i = j + 1
        except StopIteration:
            if i < batch_size:
                raise StopIteration

        #return {self.data_name  :  batch_data,
        #        self.label_name :  batch_label}
        #print(batch_data.shape, batch_label.shape)
        return mx.io.DataBatch([batch_data], [batch_label], batch_size - i)
