from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import logging
import sys
import sklearn
import faiss
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

import mxnet as mx
from mxnet import ndarray as nd
#from . import _ndarray_internal as _internal
#from mxnet._ndarray_internal import _cvimresize as imresize
#from ._ndarray_internal import _cvcopyMakeBorder as copyMakeBorder
from mxnet import io
from mxnet import recordio
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_preprocess

logger = logging.getLogger()

#modification on ImageIter
class FaceImageIter(io.DataIter):

    def __init__(self, batch_size, data_shape, images_per_person, margin = 44, path_imglist=None, path_root=None,
                 shuffle=False, aug_list=None,
                 data_name='data', label_name='softmax_label', **kwargs):
        super(FaceImageIter, self).__init__()
        assert path_imglist
        self.label2key = {}
        self.labelkeys = []
        print('loading image list...')
        with open(path_imglist) as fin:
            imglist = {}
            imgkeys = []
            key = 0
            for line in iter(fin.readline, ''):
                line = line.strip().split('\t')
                if len(line)<17:
                  continue #skip no detected face image
                label = nd.array([float(line[2])])
                ilabel = int(line[2])
                if ilabel not in self.label2key:
                  self.label2key[ilabel] = [key]
                  self.labelkeys.append(ilabel)
                  #self.labelcur[ilabel] = 0
                else:
                  self.label2key[ilabel].append(key)
                #label = nd.array([float(i) for i in line[1:-1]])
                bbox = np.array([int(i) for i in line[3:7]])
                #key = int(line[0])
                imglist[key] = (label, line[1], bbox)
                imgkeys.append(key)
                key+=1
            self.imglist = imglist
        print('image list size', len(self.imglist))

        self.path_root = path_root
        self.margin = margin

        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        self.provide_label = [(label_name, (batch_size,))]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.images_per_person = images_per_person
        #self.label_width = label_width
        self.imgkeys = imgkeys
        self.shuffle = shuffle

        if aug_list is None:
            self.auglist = mx.image.CreateAugmenter(data_shape, **kwargs)
        else:
            self.auglist = aug_list
        print('aug size:', len(self.auglist))
        #for aug in self.auglist:
        #  print(aug.__name__)
        self.cur = 0
        self.labelcur = 0
        self.reset()

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        if self.shuffle:
            #random.shuffle(self.imgkeys)
            random.shuffle(self.labelkeys)
        self.cur = 0
        self.labelcur = 0
        #for k in self.label2key:
        #  random.shuffle(self.label2key[k])

    def _next_sample(self):
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        while True:
          if self.cur >= len(self.labelkeys):
            raise StopIteration
          ilabel = self.labelkeys[self.cur]
          if self.labelcur>=min(len(self.label2key[ilabel]), self.images_per_person):
            self.labelcur=0
            self.cur+=1
          else:
            idx = self.label2key[ilabel][self.labelcur]
            self.labelcur += 1
            label, fname, bbox = self.imglist[idx]
            return label, self.read_image(fname), bbox

    def next_sample(self):
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        while True:
          if self.cur >= len(self.labelkeys):
            raise StopIteration
          ilabel = self.labelkeys[self.cur]
          if self.labelcur>=min(len(self.label2key[ilabel]), self.images_per_person):
            self.labelcur=0
            self.cur+=1
          else:
            #print('in next_sample', self.cur, self.labelcur)
            if self.labelcur==0 and self.shuffle:
              #print('shuffling')
              random.shuffle(self.label2key[ilabel])
            idx = self.label2key[ilabel][self.labelcur]
            self.labelcur += 1
            label, fname, bbox = self.imglist[idx]
            return label, self.read_image(fname), bbox

    def next(self):
        """Returns the next batch of data."""
        if self.shuffle:
            random.shuffle(self.labelkeys)
            self.cur = 0
            self.labelcur = 0
        #print('in next', self.cur, self.labelcur)
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                label, s, bbox = self.next_sample()
                data = [self.imdecode(s, bbox)]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                data = self.augmentation_transform(data)
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if not i:
                raise StopIteration

        return io.DataBatch([batch_data], [batch_label], batch_size - i)

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

    def imdecode(self, s, bbox):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        img = mx.image.imdecode(s)
        if bbox is not None:
          #print(img.shape, bbox)
          _begin = (max(0, bbox[1]-self.margin//2), max(0, bbox[0]-self.margin//2),0)
          _end = (min(img.shape[0], bbox[3]+self.margin//2), min(img.shape[1], bbox[2]+self.margin//2), 3)
          img = nd.slice(img, begin=_begin, end=_end)
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

class FaceIter(mx.io.DataIter):
  def __init__(self, data_shape, path_imglist, mod, ctx_num, batch_size=90, bag_size=1800, images_per_person=40, alpha = 0.2, data_name='data', label_name='softmax_label'):
    assert batch_size%ctx_num==0
    assert (batch_size//ctx_num)%3==0
    assert bag_size%batch_size==0
    self.mod = mod
    self.ctx_num = ctx_num
    self.batch_size = batch_size
    #self.batch_size_per_epoch = batch_size_per_epoch
    self.bag_size = bag_size
    self.data_shape = data_shape
    self.alpha = alpha
    self.data_name = data_name
    self.label_name = label_name
    #print(source_iter.provide_data)
    self.provide_data = [(self.data_name, (self.batch_size,) + self.data_shape)]
    self.provide_label = [(self.label_name, (self.batch_size,) )]
    #self.buffer = []
    #self.buffer_index = 0
    self.triplet_index = 0
    self.triplets = []
    self.data_iter = FaceImageIter(batch_size = self.batch_size, data_shape = data_shape, 
        images_per_person = images_per_person, margin = 44, 
        path_imglist = path_imglist, shuffle=True, 
        resize=182, rand_crop=True, rand_mirror=True)


  def pick_triplets(self, embeddings, nrof_images_per_class):
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    people_per_batch = len(nrof_images_per_class)
    
    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.

    for i in xrange(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<self.alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                #all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    #triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    triplets.append( (a_idx, p_idx, n_idx) )
                    #triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' % 
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets
    #return triplets, num_trips, len(triplets)

  def select_triplets(self):
    self.triplet_index = 0
    self.triplets = []
    embeddings = None
    ba = 0
    bag_size = self.bag_size
    batch_size = self.batch_size
    data = np.zeros( (bag_size,)+self.data_shape )
    label = np.zeros( (bag_size,) )
    print('eval %d images..'%bag_size)
    #print(data.shape)
    while ba<bag_size:
      bb = ba+batch_size
      _batch = self.data_iter.next()
      _data = _batch.data[0].asnumpy()
      #print(_data.shape)
      _label = _batch.label[0].asnumpy()
      data[ba:bb,:,:,:] = _data
      label[ba:bb] = _label

      self.mod.forward(_batch, is_train=False)
      net_out = self.mod.get_outputs()
      #print('eval for selecting triplets',ba,bb)
      #print(net_out)
      #print(len(net_out))
      #print(net_out[0].asnumpy())
      net_out = net_out[0].asnumpy()
      #print(net_out)
      #print('net_out', net_out.shape)
      if embeddings is None:
        embeddings = np.zeros( (bag_size, net_out.shape[1]))
      embeddings[ba:bb,:] = net_out
      ba = bb
    nrof_images_per_class = [1]
    for i in xrange(1, bag_size):
      if label[i]==label[i-1]:
        nrof_images_per_class[-1]+=1
      else:
        nrof_images_per_class.append(1)
      
    self.triplets = self.pick_triplets(embeddings, nrof_images_per_class) # shape=(T,3)
    self.buffer_data = data
    self.buffer_label = label
    self.embeddings = embeddings
    print('buffering triplets..', len(self.triplets))
    print('epoches...', len(self.triplets)*3//self.batch_size)
    if len(self.triplets)==0:
      print(embeddings.shape, label.shape, data.shape, ba)
      print('images_per_class', nrof_images_per_class)
      print(label)
      print(embeddings)
      sys.exit(0)


  def next(self):
    batch_size = self.batch_size
    ta = self.triplet_index
    tb = ta + batch_size//3
    while tb>=len(self.triplets):
      self.select_triplets()
      ta = self.triplet_index
      tb = ta + batch_size//3
    data = np.zeros( (batch_size,)+self.data_shape )
    label = np.zeros( (batch_size,) )
    for ti in xrange(ta, tb):
      triplet = self.triplets[ti]
      anchor = self.embeddings[triplet[0]]
      positive = self.embeddings[triplet[1]]
      negative = self.embeddings[triplet[2]]
      ap = anchor-positive
      ap = ap*ap
      ap = np.sum(ap)
      an = anchor-negative
      an = an*an
      an = np.sum(an)
      assert ap<=an
      assert ap+self.alpha>=an
      _ti = ti-ta
      ctx_block = (_ti*3)//(self.batch_size//self.ctx_num)
      #apn_block = ((ti*3)%self.batch_size)%3
      #apn_pos = ((ti*3)%self.batch_size)//3
      base_pos = ctx_block*(self.batch_size//self.ctx_num) + (_ti%(self.batch_size//self.ctx_num//3)) 
      for ii in xrange(3):
        id = triplet[ii]
        pos = base_pos + ii*(self.batch_size//self.ctx_num//3)
        #print('id-pos', _ti, ii, pos)
        data[pos,:,:,:] = self.buffer_data[id, :,:,:]
        label[pos] = self.buffer_label[id]
    db = io.DataBatch(data=(nd.array(data),), label=(nd.array(label),))
    self.triplet_index = tb
    return db


  def reset(self):
    self.data_iter.reset()
    self.triplet_index = 0
    self.triplets = []
    #self.target_iter.reset()

class FaceImageIter2(io.DataIter):

    def __init__(self, batch_size, data_shape, path_imglist=None, path_root=None,
                 path_imgrec = None,
                 shuffle=False, aug_list=None, exclude_lfw = False, mean = None,
                 patch = [0,0,96,112,0], rand_mirror = False,
                 data_name='data', label_name='softmax_label', **kwargs):
        super(FaceImageIter2, self).__init__()
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4]+".idx"
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
            self.imgidx = list(self.imgrec.keys)
            if shuffle:
              self.seq = self.imgidx
            else:
              self.seq = None
        else:
            self.imgrec = None
            assert path_imglist
            print('loading image list...')
            with open(path_imglist) as fin:
                imglist = {}
                imgkeys = []
                key = 0
                for line in iter(fin.readline, ''):
                    line = line.strip().split('\t')
                    flag = int(line[0])
                    if flag==0:
                      assert len(line)==17
                    else:
                      assert len(line)==3
                    label = nd.array([float(line[2])])
                    ilabel = int(line[2])
                    bbox = None
                    landmark = None
                    if len(line)==17:
                      bbox = np.array([int(i) for i in line[3:7]])
                      landmark = np.array([float(i) for i in line[7:17]]).reshape( (2,5) ).T
                    image_path = line[1]
                    if exclude_lfw:
                      _vec = image_path.split('/')
                      person_id = int(_vec[-2])
                      if person_id==166921 or person_id==1056413 or person_id==1193098:
                        continue
                    imglist[key] = (label, image_path, bbox, landmark)
                    imgkeys.append(key)
                    key+=1
                    #if key>=10000:
                    #  break
                self.imglist = imglist
            print('image list size', len(self.imglist))
            self.seq = imgkeys

        self.path_root = path_root
        self.mean = mean
        if self.mean:
          self.mean = np.array(self.mean, dtype=np.float32).reshape(1,1,3)
        self.patch = patch

        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        self.provide_label = [(label_name, (batch_size,))]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
        self.rand_mirror = rand_mirror
        #self.cast_aug = mx.image.CastAug()
        #self.color_aug = mx.image.ColorJitterAug(0.4, 0.4, 0.4)

        if aug_list is None:
            self.auglist = mx.image.CreateAugmenter(data_shape, **kwargs)
        else:
            self.auglist = aug_list
        print('aug size:', len(self.auglist))
        for aug in self.auglist:
          print(aug.__class__)
        self.cur = 0
        self.reset()

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        if self.shuffle:
            random.shuffle(self.seq)
        if self.imgrec is not None:
            self.imgrec.reset()
        self.cur = 0

    def num_samples(self):
      return len(self.seq)

    def next_sample(self):
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        if self.seq is not None:
          if self.cur >= len(self.seq):
              raise StopIteration
          idx = self.seq[self.cur]
          self.cur += 1
          if self.imgrec is not None:
            s = self.imgrec.read_idx(idx)
            header, img = recordio.unpack(s)
            return header.label, img, None, None
          else:
            label, fname, bbox, landmark = self.imglist[idx]
            return label, self.read_image(fname), bbox, landmark
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
            return header.label, img, None, None

    def brightness_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      src *= alpha
      return src

    def contrast_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      coef = np.array([[[0.299, 0.587, 0.114]]])
      gray = src * coef
      gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
      src *= alpha
      src += gray
      return src

    def saturation_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      coef = np.array([[[0.299, 0.587, 0.114]]])
      gray = src * coef
      gray = np.sum(gray, axis=2, keepdims=True)
      gray *= (1.0 - alpha)
      src *= alpha
      src += gray
      return src

    def color_aug(self, img, x):
      augs = [self.brightness_aug, self.contrast_aug, self.saturation_aug]
      random.shuffle(augs)
      for aug in augs:
        #print(img.shape)
        img = aug(img, x)
        #print(img.shape)
      return img

    def mirror_aug(self, img):
      _rd = random.randint(0,1)
      if _rd==1:
        for c in xrange(img.shape[2]):
          img[:,:,c] = np.fliplr(img[:,:,c])
      return img


    def next(self):
        """Returns the next batch of data."""
        #print('in next', self.cur, self.labelcur)
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                label, s, bbox, landmark = self.next_sample()
                _data = self.imdecode(s)
                #_npdata = _data
                _npdata = _data.asnumpy()
                if landmark is not None:
                  _npdata = face_preprocess.preprocess(_npdata, bbox = bbox, landmark=landmark, image_size=self.image_size)
                #_npdata = self.color_aug(_npdata, 0.1)
                if self.rand_mirror:
                  _npdata = self.mirror_aug(_npdata)
                if self.mean is not None:
                  _npdata = _npdata.astype(np.float32)
                  _npdata -= self.mean
                  _npdata *= 0.0078125
                nimg = np.zeros(_npdata.shape, dtype=np.float32)
                nimg[self.patch[1]:self.patch[3],self.patch[0]:self.patch[2],:] = _npdata[self.patch[1]:self.patch[3], self.patch[0]:self.patch[2], :]
                #print(_npdata.shape)
                #print(_npdata)
                _data = mx.nd.array(nimg)
                #print(_data.shape)
                data = [_data]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                #print('aa',data[0].shape)
                #data = self.augmentation_transform(data)
                #print('bb',data[0].shape)
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    #print(datum.shape)
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i<batch_size:
                raise StopIteration

        #print('next end', batch_size, i)
        return io.DataBatch([batch_data], [batch_label], batch_size - i)

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
        #arr = np.fromstring(s, np.uint8)
        if self.patch[4]%2==0:
          img = mx.image.imdecode(s)
          #img = cv2.imdecode(arr, cv2.CV_LOAD_IMAGE_COLOR)
          #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        else:
          img = mx.image.imdecode(s, flag=0)
          img = nd.broadcast_to(img, (img.shape[0], img.shape[1], 3))
          #img = cv2.imdecode(arr, cv2.CV_LOAD_IMAGE_GRAY)
        #img = np.float32(img)
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


class FaceImageIter4(io.DataIter):

    def __init__(self, batch_size, ctx_num, images_per_identity, data_shape, 
        path_imglist=None, path_root=None,
        shuffle=False, aug_list=None, exclude_lfw = False, mean = None, use_extra = False, model = None,
        patch = [0,0,96,112,0],  rand_mirror = False,
        data_name='data', label_name='softmax_label', **kwargs):
        super(FaceImageIter4, self).__init__()
        assert path_imglist
        print('loading image list...')
        with open(path_imglist) as fin:
            self.imglist = {}
            self.imgkeys = []
            self.labels = []
            self.olabels = []
            self.labelposting = {}
            self.seq = []
            key = 0
            for line in iter(fin.readline, ''):
                line = line.strip().split('\t')
                flag = int(line[0])
                if flag==0:
                  assert len(line)==17
                else:
                  assert len(line)==3
                label = nd.array([float(line[2])])
                ilabel = int(line[2])
                bbox = None
                landmark = None
                if len(line)==17:
                  bbox = np.array([int(i) for i in line[3:7]])
                  landmark = np.array([float(i) for i in line[7:17]]).reshape( (2,5) ).T
                image_path = line[1]
                if exclude_lfw:
                  _vec = image_path.split('/')
                  person_id = int(_vec[-2])
                  if person_id==166921 or person_id==1056413 or person_id==1193098:
                    continue
                self.imglist[key] = (label, image_path, bbox, landmark)
                self.seq.append(key)
                if ilabel in self.labelposting:
                  self.labelposting[ilabel].append(key)
                else:
                  self.labelposting[ilabel] = [key]
                  self.olabels.append(ilabel)
                key+=1
                #if key>=10000:
                #  break
        print('image list size', len(self.imglist))
        print('label size', len(self.olabels))
        print('last label',self.olabels[-1])

        self.path_root = path_root
        self.mean = mean
        if self.mean:
          self.mean = np.array(self.mean, dtype=np.float32).reshape(1,1,3)
        self.patch = patch 

        self.check_data_shape(data_shape)
        per_batch_size = int(batch_size/ctx_num)
        self.provide_label = [(label_name, (batch_size,))]
        self.batch_size = batch_size
        self.ctx_num = ctx_num 
        self.images_per_identity = images_per_identity
        self.identities = int(per_batch_size/self.images_per_identity)
        self.min_per_identity = 10
        if self.images_per_identity<=10:
          self.min_per_identity = self.images_per_identity
        self.min_per_identity = 1
        assert self.min_per_identity<=self.images_per_identity
        print(self.images_per_identity, self.identities, self.min_per_identity)
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
        self.rand_mirror = rand_mirror
        print('rand_mirror', self.rand_mirror)
        self.extra = None
        self.model = model
        if use_extra:
          self.provide_data = [(data_name, (batch_size,) + data_shape), ('extra', (batch_size, per_batch_size))]
          self.extra = np.full(self.provide_data[1][1], -1.0, dtype=np.float32)
          c = 0
          while c<batch_size:
            a = 0
            while a<per_batch_size:
              b = a+images_per_identity
              self.extra[(c+a):(c+b),a:b] = 1.0
              #print(c+a, c+b, a, b)
              a = b
            c += per_batch_size
          self.extra = nd.array(self.extra)
          #self.batch_label = nd.empty(self.provide_label[0][1])
          #per_batch_size = int(batch_size/ctx_num)
          #_label = -1
          #for i in xrange(batch_size):
          #  if i%self.images_per_identity==0:
          #    _label+=1
          #    if i%per_batch_size==0:
          #      _label = 0
          #  label = nd.array([float(_label)])
          #  self.batch_label[i][:] = label
          #print(self.batch_label)
          print(self.extra)
        else:
          self.provide_data = [(data_name, (batch_size,) + data_shape)]

        if aug_list is None:
            self.auglist = mx.image.CreateAugmenter(data_shape, **kwargs)
        else:
            self.auglist = aug_list
        print('aug size:', len(self.auglist))
        for aug in self.auglist:
          print(aug.__class__)
        self.cur = [0, 0]
        self.inited = False

    def get_extra(self):
      return self.extra

    def offline_reset(self):
      data = nd.zeros( self.provide_data[0][1] )
      label = nd.zeros( self.provide_label[0][1] )
      #label = np.zeros( self.provide_label[0][1] )
      X = None
      ba = 0
      batch_num = 0
      while ba<len(self.seq):
        batch_num+=1
        if batch_num%10==0:
          print('loading batch',batch_num, ba)
        bb = min(ba+self.batch_size, len(self.seq))
        _count = bb-ba
        for i in xrange(_count):
          key = self.seq[i+ba]
          _label, fname, bbox, landmark = self.imglist[key]
          s = self.read_image(fname)
          _data = self.imdecode(s)
          #_data = self.augmentation_transform([_data])[0]
          _npdata = _data.asnumpy()
          if landmark is not None:
            _npdata = face_preprocess.preprocess(_npdata, bbox = bbox, landmark=landmark, image_size=self.image_size)
          if self.mean is not None:
            _npdata = _npdata.astype(np.float32)
            _npdata -= self.mean
            _npdata *= 0.0078125
          nimg = np.zeros(_npdata.shape, dtype=np.float32)
          nimg[self.patch[1]:self.patch[3],self.patch[0]:self.patch[2],:] = _npdata[self.patch[1]:self.patch[3], self.patch[0]:self.patch[2], :]
          #print(_npdata.shape)
          #print(_npdata)
          _data = mx.nd.array(nimg)
          data[i][:] = self.postprocess_data(_data)
          label[i][:] = _label
        db = mx.io.DataBatch(data=(data,self.extra), label=(label,))
        self.model.forward(db, is_train=False)
        net_out = self.model.get_outputs()
        _embeddings = net_out[0].asnumpy()
        _embeddings = sklearn.preprocessing.normalize(_embeddings)
        if _count<self.batch_size:
          _embeddings = _embeddings[0:_count,:]
        #print(_embeddings.shape)
        if X is None:
          X = np.zeros( (len(self.olabels), _embeddings.shape[1]), dtype=np.float32 )
        nplabel = label.asnumpy()
        for i in xrange(_count):
          ilabel = int(nplabel[i])
          #print(ilabel, ilabel.__class__)
          X[ilabel] += _embeddings[i]
        ba = bb
      X = sklearn.preprocessing.normalize(X)
      d = X.shape[1]
      faiss_params = [20,5]
      print('start to train faiss')
      print(X.shape)
      quantizer = faiss.IndexFlatL2(d)  # the other index
      index = faiss.IndexIVFFlat(quantizer, d, faiss_params[0], faiss.METRIC_L2)
      assert not index.is_trained
      index.train(X)
      index.add(X)
      assert index.is_trained
      print('trained')
      index.nprobe = faiss_params[1]
      k = self.identities
      D, I = index.search(X, k)     # actual search
      print(I.shape)
      self.labels = []
      for i in xrange(I.shape[0]):
        #assert I[i][0]==i
        for j in xrange(k):
          _label = I[i][j]
          assert _label<len(self.olabels)
          self.labels.append(_label)
      print('labels assigned', len(self.labels))


      

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        if self.extra is not None:
          self.offline_reset()
        elif self.shuffle:
          random.shuffle(self.labels)
        self.cur = [0,0]

    def num_samples(self):
      #count = 0
      #for k,v in self.labelposting.iteritems():
      #  if len(v)<self.min_per_identity:
      #    continue
      #  count+=self.images_per_identity
      count = len(self.olabels)*self.images_per_identity*self.identities
      return count


    def next_sample(self):
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        while True:
          if self.cur[0] >= len(self.labels):
            raise StopIteration
          label = self.labels[self.cur[0]]
          posting = self.labelposting[label]
          if len(posting)<self.min_per_identity or self.cur[1] >= self.images_per_identity:
            self.cur[0]+=1
            self.cur[1] = 0
            continue
          if self.shuffle and self.cur[1]==0:
            random.shuffle(posting)
          idx = posting[self.cur[1]%len(posting)]
          self.cur[1] += 1
          label, fname, bbox, landmark = self.imglist[idx]
          return label, self.read_image(fname), bbox, landmark


    def next(self):
        if not self.inited:
          self.reset()
          self.inited = True
        """Returns the next batch of data."""
        #print('in next', self.cur, self.labelcur)
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                label, s, bbox, landmark = self.next_sample()
                _data = self.imdecode(s)
                #_data = self.augmentation_transform([_data])[0]
                _npdata = _data.asnumpy()
                if landmark is not None:
                  _npdata = face_preprocess.preprocess(_npdata, bbox = bbox, landmark=landmark, image_size=self.image_size)
                if self.rand_mirror:
                  _rd = random.randint(0,1)
                  if _rd==1:
                    for c in xrange(_npdata.shape[2]):
                      _npdata[:,:,c] = np.fliplr(_npdata[:,:,c])
                if self.mean is not None:
                  _npdata = _npdata.astype(np.float32)
                  _npdata -= self.mean
                  _npdata *= 0.0078125
                nimg = np.zeros(_npdata.shape, dtype=np.float32)
                nimg[self.patch[1]:self.patch[3],self.patch[0]:self.patch[2],:] = _npdata[self.patch[1]:self.patch[3], self.patch[0]:self.patch[2], :]
                #print(_npdata.shape)
                #print(_npdata)
                _data = mx.nd.array(nimg)
                #print(_data.shape)
                data = [_data]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                #print('aa',data[0].shape)
                #data = self.augmentation_transform(data)
                #print('bb',data[0].shape)
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    #print(datum.shape)
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i<batch_size:
                raise StopIteration

        #print('next end', batch_size, i)
        if self.extra is not None:
          return io.DataBatch([batch_data, self.extra], [batch_label], batch_size - i)
        else:
          return io.DataBatch([batch_data], [batch_label], batch_size - i)

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
        if self.patch[4]%2==0:
          img = mx.image.imdecode(s)
        else:
          img = mx.image.imdecode(s, flag=0)
          img = nd.broadcast_to(img, (img.shape[0], img.shape[1], 3))
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

class FaceImageIter5(io.DataIter):

    def __init__(self, batch_size, ctx_num, images_per_identity, data_shape, 
        path_imglist=None, path_root=None,
        shuffle=False, aug_list=None, exclude_lfw = False, mean = None,
        patch = [0,0,96,112,0],  rand_mirror = False,
        data_name='data', label_name='softmax_label', **kwargs):
        super(FaceImageIter5, self).__init__()
        assert path_imglist
        print('loading image list...')
        with open(path_imglist) as fin:
            self.imglist = {}
            self.labels = []
            self.olabels = []
            self.labelposting = {}
            self.seq = []
            key = 0
            for line in iter(fin.readline, ''):
                line = line.strip().split('\t')
                flag = int(line[0])
                if flag==0:
                  assert len(line)==17
                else:
                  assert len(line)==3
                label = nd.array([float(line[2])])
                ilabel = int(line[2])
                bbox = None
                landmark = None
                if len(line)==17:
                  bbox = np.array([int(i) for i in line[3:7]])
                  landmark = np.array([float(i) for i in line[7:17]]).reshape( (2,5) ).T
                image_path = line[1]
                if exclude_lfw:
                  _vec = image_path.split('/')
                  person_id = int(_vec[-2])
                  if person_id==166921 or person_id==1056413 or person_id==1193098:
                    continue
                self.imglist[key] = (label, image_path, bbox, landmark)
                self.seq.append(key)
                if ilabel in self.labelposting:
                  self.labelposting[ilabel].append(key)
                else:
                  self.labelposting[ilabel] = [key]
                  self.olabels.append(ilabel)
                key+=1
                #if key>=10000:
                #  break
        print('image list size', len(self.imglist))
        print('label size', len(self.olabels))
        print('last label',self.olabels[-1])

        self.path_root = path_root
        self.mean = mean
        if self.mean:
          self.mean = np.array(self.mean, dtype=np.float32).reshape(1,1,3)
        self.patch = patch 

        self.check_data_shape(data_shape)
        self.per_batch_size = int(batch_size/ctx_num)
        self.provide_label = [(label_name, (batch_size,))]
        self.batch_size = batch_size
        self.ctx_num = ctx_num 
        self.images_per_identity = images_per_identity
        self.identities = int(self.per_batch_size/self.images_per_identity)
        self.min_per_identity = 10
        if self.images_per_identity<=10:
          self.min_per_identity = self.images_per_identity
        self.min_per_identity = 1
        assert self.min_per_identity<=self.images_per_identity
        print(self.images_per_identity, self.identities, self.min_per_identity)
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
        self.rand_mirror = rand_mirror
        print('rand_mirror', self.rand_mirror)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]

        if aug_list is None:
            self.auglist = mx.image.CreateAugmenter(data_shape, **kwargs)
        else:
            self.auglist = aug_list
        print('aug size:', len(self.auglist))
        for aug in self.auglist:
          print(aug.__class__)
        self.cur = 0
        self.buffer = []
        self.reset()


    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        if self.shuffle:
          random.shuffle(self.seq)
        self.cur = 0

    def num_samples(self):
      return -1


    def next_sample(self, i_ctx):
        if self.cur >= len(self.seq):
          raise StopIteration
        if i_ctx==0:
          idx = self.seq[self.cur]
          self.cur += 1
          label, fname, bbox, landmark = self.imglist[idx]
          ilabel = int(label.asnumpy()[0])
          self.buffer = self.labelposting[ilabel]
          random.shuffle(self.buffer)
        if i_ctx<self.images_per_identity:
          pos = i_ctx%len(self.buffer)
          idx = self.buffer[pos]
        else:
          idx = self.seq[self.cur]
          self.cur += 1
        label, fname, bbox, landmark = self.imglist[idx]
        return label, self.read_image(fname), bbox, landmark


    def next(self):
        """Returns the next batch of data."""
        #print('in next', self.cur, self.labelcur)
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                i_ctx = i%self.per_batch_size
                label, s, bbox, landmark = self.next_sample(i_ctx)
                _data = self.imdecode(s)
                #_data = self.augmentation_transform([_data])[0]
                _npdata = _data.asnumpy()
                if landmark is not None:
                  _npdata = face_preprocess.preprocess(_npdata, bbox = bbox, landmark=landmark, image_size=self.image_size)
                if self.rand_mirror:
                  _rd = random.randint(0,1)
                  if _rd==1:
                    for c in xrange(_npdata.shape[2]):
                      _npdata[:,:,c] = np.fliplr(_npdata[:,:,c])
                if self.mean is not None:
                  _npdata = _npdata.astype(np.float32)
                  _npdata -= self.mean
                  _npdata *= 0.0078125
                nimg = np.zeros(_npdata.shape, dtype=np.float32)
                nimg[self.patch[1]:self.patch[3],self.patch[0]:self.patch[2],:] = _npdata[self.patch[1]:self.patch[3], self.patch[0]:self.patch[2], :]
                #print(_npdata.shape)
                #print(_npdata)
                _data = mx.nd.array(nimg)
                #print(_data.shape)
                data = [_data]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                #print('aa',data[0].shape)
                #data = self.augmentation_transform(data)
                #print('bb',data[0].shape)
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    #print(datum.shape)
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i<batch_size:
                raise StopIteration

        return io.DataBatch([batch_data], [batch_label], batch_size - i)

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
        if self.patch[4]%2==0:
          img = mx.image.imdecode(s)
        else:
          img = mx.image.imdecode(s, flag=0)
          img = nd.broadcast_to(img, (img.shape[0], img.shape[1], 3))
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
