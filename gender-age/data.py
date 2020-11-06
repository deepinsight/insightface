from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import logging
import sys
import numbers
import math
import sklearn
import datetime
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio

logger = logging.getLogger()


class FaceImageIter(io.DataIter):
    def __init__(self,
                 batch_size,
                 data_shape,
                 path_imgrec=None,
                 shuffle=False,
                 aug_list=None,
                 mean=None,
                 rand_mirror=False,
                 cutoff=0,
                 color_jittering=0,
                 data_name='data',
                 label_name='softmax_label',
                 **kwargs):
        super(FaceImageIter, self).__init__()
        assert path_imgrec
        logging.info('loading recordio %s...', path_imgrec)
        path_imgidx = path_imgrec[0:-4] + ".idx"
        self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
        s = self.imgrec.read_idx(0)
        header, _ = recordio.unpack(s)
        self.imgidx = list(self.imgrec.keys)
        self.seq = self.imgidx

        self.mean = mean
        self.nd_mean = None
        if self.mean:
            self.mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
            self.nd_mean = mx.nd.array(self.mean).reshape((1, 1, 3))

        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size, ) + data_shape)]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.image_size = '%d,%d' % (data_shape[1], data_shape[2])
        self.rand_mirror = rand_mirror
        print('rand_mirror', rand_mirror)
        self.cutoff = cutoff
        self.color_jittering = color_jittering
        self.CJA = mx.image.ColorJitterAug(0.125, 0.125, 0.125)
        self.provide_label = [(label_name, (batch_size, 101))]
        #print(self.provide_label[0][1])
        self.cur = 0
        self.nbatch = 0
        self.is_init = False

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        self.cur = 0
        if self.shuffle:
            random.shuffle(self.seq)
        if self.seq is None and self.imgrec is not None:
            self.imgrec.reset()

    def num_samples(self):
        return len(self.seq)

    def next_sample(self):
        if self.cur >= len(self.seq):
            raise StopIteration
        idx = self.seq[self.cur]
        self.cur += 1
        s = self.imgrec.read_idx(idx)
        header, img = recordio.unpack(s)
        label = header.label
        return label, img, None, None

    def brightness_aug(self, src, x):
        alpha = 1.0 + random.uniform(-x, x)
        src *= alpha
        return src

    def contrast_aug(self, src, x):
        alpha = 1.0 + random.uniform(-x, x)
        coef = nd.array([[[0.299, 0.587, 0.114]]])
        gray = src * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * nd.sum(gray)
        src *= alpha
        src += gray
        return src

    def saturation_aug(self, src, x):
        alpha = 1.0 + random.uniform(-x, x)
        coef = nd.array([[[0.299, 0.587, 0.114]]])
        gray = src * coef
        gray = nd.sum(gray, axis=2, keepdims=True)
        gray *= (1.0 - alpha)
        src *= alpha
        src += gray
        return src

    def color_aug(self, img, x):
        #augs = [self.brightness_aug, self.contrast_aug, self.saturation_aug]
        #random.shuffle(augs)
        #for aug in augs:
        #  #print(img.shape)
        #  img = aug(img, x)
        #  #print(img.shape)
        #return img
        return self.CJA(img)

    def mirror_aug(self, img):
        _rd = random.randint(0, 1)
        if _rd == 1:
            for c in range(img.shape[2]):
                img[:, :, c] = np.fliplr(img[:, :, c])
        return img

    def compress_aug(self, img):
        buf = BytesIO()
        img = Image.fromarray(img.asnumpy(), 'RGB')
        q = random.randint(2, 20)
        img.save(buf, format='JPEG', quality=q)
        buf = buf.getvalue()
        img = Image.open(BytesIO(buf))
        return nd.array(np.asarray(img, 'float32'))

    def next(self):
        if not self.is_init:
            self.reset()
            self.is_init = True
        """Returns the next batch of data."""
        #print('in next', self.cur, self.labelcur)
        self.nbatch += 1
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        if self.provide_label is not None:
            batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                #print('XXXX', i)
                label, s, bbox, landmark = self.next_sample()
                gender = int(label[0])
                age = int(label[1])
                assert age >= 0
                #assert gender==0 or gender==1
                plabel = np.zeros(shape=(101, ), dtype=np.float32)
                plabel[0] = gender
                if age == 0:
                    age = 1
                if age > 100:
                    age = 100
                plabel[1:age + 1] = 1
                label = plabel
                _data = self.imdecode(s)
                if _data.shape[0] != self.data_shape[1]:
                    _data = mx.image.resize_short(_data, self.data_shape[1])
                if self.rand_mirror:
                    _rd = random.randint(0, 1)
                    if _rd == 1:
                        _data = mx.ndarray.flip(data=_data, axis=1)
                if self.color_jittering > 0:
                    if self.color_jittering > 1:
                        _rd = random.randint(0, 1)
                        if _rd == 1:
                            _data = self.compress_aug(_data)
                    #print('do color aug')
                    _data = _data.astype('float32', copy=False)
                    #print(_data.__class__)
                    _data = self.color_aug(_data, 0.125)
                if self.nd_mean is not None:
                    _data = _data.astype('float32', copy=False)
                    _data -= self.nd_mean
                    _data *= 0.0078125
                if self.cutoff > 0:
                    _rd = random.randint(0, 1)
                    if _rd == 1:
                        #print('do cutoff aug', self.cutoff)
                        centerh = random.randint(0, _data.shape[0] - 1)
                        centerw = random.randint(0, _data.shape[1] - 1)
                        half = self.cutoff // 2
                        starth = max(0, centerh - half)
                        endh = min(_data.shape[0], centerh + half)
                        startw = max(0, centerw - half)
                        endw = min(_data.shape[1], centerw + half)
                        #print(starth, endh, startw, endw, _data.shape)
                        _data[starth:endh, startw:endw, :] = 128
                data = [_data]
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    #print(datum.shape)
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i < batch_size:
                raise StopIteration

        return io.DataBatch([batch_data], [batch_label], batch_size - i)

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError(
                'data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError(
                'This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        img = mx.image.imdecode(s)  #mx.ndarray
        return img

    def read_image(self, fname):
        """Reads an input image `fname` and returns the decoded raw bytes.

        Example usage:
        ----------
        >>> dataIter.read_image('Face.jpg') # returns decoded raw bytes.
        """
        with open(os.path.join(self.path_root, fname), 'rb') as fin:
            img = fin.read()
        return img

    def augmentation_transform(self, data):
        """Transforms input data with specified augmentation."""
        for aug in self.auglist:
            data = [ret for src in data for ret in aug(src)]
        return data

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))


class FaceImageIterList(io.DataIter):
    def __init__(self, iter_list):
        assert len(iter_list) > 0
        self.provide_data = iter_list[0].provide_data
        self.provide_label = iter_list[0].provide_label
        self.iter_list = iter_list
        self.cur_iter = None

    def reset(self):
        self.cur_iter.reset()

    def next(self):
        self.cur_iter = random.choice(self.iter_list)
        while True:
            try:
                ret = self.cur_iter.next()
            except StopIteration:
                self.cur_iter.reset()
                continue
            return ret
