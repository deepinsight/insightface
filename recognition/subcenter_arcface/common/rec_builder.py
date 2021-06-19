import os
import sys
import mxnet as mx
from mxnet import ndarray as nd
import random
import argparse
import cv2
import time
import sklearn
import numpy as np


class SeqRecBuilder():
    def __init__(self, path, image_size=(112, 112)):
        self.path = path
        self.image_size = image_size
        self.last_label = -1
        self.widx = 0
        if not os.path.exists(path):
            os.makedirs(path)
        self.writer = mx.recordio.MXIndexedRecordIO(
            os.path.join(path, 'train.idx'), os.path.join(path, 'train.rec'),
            'w')
        self.label_stat = [-1, -1]

    def add(self, label, img, is_image=True):
        #img should be BGR
        #if self.sis:
        #    assert label>=self.last_label
        idx = self.widx
        self.widx += 1
        header = mx.recordio.IRHeader(0, label, idx, 0)
        if is_image:
            s = mx.recordio.pack_img(header, img, quality=95, img_fmt='.jpg')
        else:
            s = mx.recordio.pack(header, img)
        self.writer.write_idx(idx, s)
        if self.label_stat[0] < 0:
            self.label_stat = [label, label]
        else:
            self.label_stat[0] = min(self.label_stat[0], label)
            self.label_stat[1] = max(self.label_stat[1], label)

    def close(self):
        with open(os.path.join(self.path, 'property'), 'w') as f:
            f.write("%d,%d,%d\n" % (self.label_stat[1] + 1, self.image_size[0],
                                    self.image_size[1]))


class RecBuilder():
    def __init__(self, path, image_size=(112, 112)):
        self.path = path
        self.image_size = image_size
        self.last_label = -1
        self.widx = 1
        if not os.path.exists(path):
            os.makedirs(path)
        self.writer = mx.recordio.MXIndexedRecordIO(
            os.path.join(path, 'train.idx'), os.path.join(path, 'train.rec'),
            'w')
        self.label_stat = [-1, -1]
        self.identities = []

    def add(self, label, imgs):
        #img should be BGR
        assert label >= 0
        assert label > self.last_label
        assert len(imgs) > 0
        idflag = [self.widx, -1]
        for img in imgs:
            idx = self.widx
            self.widx += 1
            header = mx.recordio.IRHeader(0, label, idx, 0)
            if isinstance(img, np.ndarray):
                s = mx.recordio.pack_img(header,
                                         img,
                                         quality=95,
                                         img_fmt='.jpg')
            else:
                s = mx.recordio.pack(header, img)
            self.writer.write_idx(idx, s)
        idflag[1] = self.widx
        self.identities.append(idflag)
        if self.label_stat[0] < 0:
            self.label_stat = [label, label]
        else:
            self.label_stat[0] = min(self.label_stat[0], label)
            self.label_stat[1] = max(self.label_stat[1], label)
        self.last_label = label

    def close(self):
        id_idx = self.widx
        for id_flag in self.identities:
            idx = self.widx
            self.widx += 1
            _header = mx.recordio.IRHeader(0, id_flag, idx, 0)
            s = mx.recordio.pack(_header, b'')
            self.writer.write_idx(idx, s)

        print('id0:', (id_idx, self.widx))
        idx = 0
        _header = mx.recordio.IRHeader(0, (id_idx, self.widx), idx, 1)
        s = mx.recordio.pack(_header, b'')
        self.writer.write_idx(idx, s)
        print('label stat:', self.label_stat)

        with open(os.path.join(self.path, 'property'), 'w') as f:
            f.write("%d,%d,%d\n" % (self.label_stat[1] + 1, self.image_size[0],
                                    self.image_size[1]))
