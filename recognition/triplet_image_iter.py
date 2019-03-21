# THIS FILE IS FOR EXPERIMENTS, USE image_iter.py FOR NORMAL IMAGE LOADING.
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

class FaceImageIter(io.DataIter):

    def __init__(self, batch_size, data_shape,
                 path_imgrec = None,
                 shuffle=False, aug_list=None,
                 rand_mirror = False, cutoff = 0,
                 ctx_num = 0, images_per_identity = 0,
                 triplet_params = None,
                 mx_model = None,
                 data_name='data', label_name='softmax_label', **kwargs):
        super(FaceImageIter, self).__init__()
        assert path_imgrec
        assert shuffle
        logging.info('loading recordio %s...',
                     path_imgrec)
        path_imgidx = path_imgrec[0:-4]+".idx"
        self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
        s = self.imgrec.read_idx(0)
        header, _ = recordio.unpack(s)
        assert header.flag>0
        print('header0 label', header.label)
        self.header0 = (int(header.label[0]), int(header.label[1]))
        #assert(header.flag==1)
        self.imgidx = range(1, int(header.label[0]))
        self.id2range = {}
        self.seq_identity = range(int(header.label[0]), int(header.label[1]))
        for identity in self.seq_identity:
          s = self.imgrec.read_idx(identity)
          header, _ = recordio.unpack(s)
          a,b = int(header.label[0]), int(header.label[1])
          self.id2range[identity] = (a,b)

        print('id2range', len(self.id2range))
        self.seq = self.imgidx
        print(len(self.seq))

        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
        self.rand_mirror = rand_mirror
        print('rand_mirror', rand_mirror)
        self.cutoff = cutoff
        #self.cast_aug = mx.image.CastAug()
        #self.color_aug = mx.image.ColorJitterAug(0.4, 0.4, 0.4)
        self.ctx_num = ctx_num 
        self.per_batch_size = int(self.batch_size/self.ctx_num)
        self.images_per_identity = images_per_identity
        if self.images_per_identity>0:
          self.identities = int(self.per_batch_size/self.images_per_identity)
          self.per_identities = self.identities
          self.repeat = 3000000.0/(self.images_per_identity*len(self.id2range))
          self.repeat = int(self.repeat)
          print(self.images_per_identity, self.identities, self.repeat)
        self.mx_model = mx_model
        self.triplet_params = triplet_params
        self.triplet_mode = False
        #self.provide_label = None
        self.provide_label = [(label_name, (batch_size,))]
        if self.triplet_params is not None:
          assert self.images_per_identity>0
          assert self.mx_model is not None
          self.triplet_bag_size = self.triplet_params[0]
          self.triplet_alpha = self.triplet_params[1]
          self.triplet_max_ap = self.triplet_params[2]
          assert self.triplet_bag_size>0
          assert self.triplet_alpha>=0.0
          assert self.triplet_alpha<=1.0
          self.triplet_mode = True
          self.triplet_cur = 0
          self.triplet_seq = []
          self.triplet_reset()
          self.seq_min_size = self.batch_size*2
        self.cur = 0
        self.nbatch = 0
        self.is_init = False
        self.times = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #self.reset()



    def pairwise_dists(self, embeddings):
      nd_embedding_list = []
      for i in xrange(self.ctx_num):
        nd_embedding = mx.nd.array(embeddings, mx.gpu(i))
        nd_embedding_list.append(nd_embedding)
      nd_pdists = []
      pdists = []
      for idx in xrange(embeddings.shape[0]):
        emb_idx = idx%self.ctx_num
        nd_embedding = nd_embedding_list[emb_idx]
        a_embedding = nd_embedding[idx]
        body = mx.nd.broadcast_sub(a_embedding, nd_embedding)
        body = body*body
        body = mx.nd.sum_axis(body, axis=1)
        nd_pdists.append(body)
        if len(nd_pdists)==self.ctx_num or idx==embeddings.shape[0]-1:
          for x in nd_pdists:
            pdists.append(x.asnumpy())
          nd_pdists = []
      return pdists

    def pick_triplets(self, embeddings, nrof_images_per_class):
      emb_start_idx = 0
      triplets = []
      people_per_batch = len(nrof_images_per_class)
      #self.time_reset()
      pdists = self.pairwise_dists(embeddings)
      #self.times[3] += self.time_elapsed()

      for i in xrange(people_per_batch):
          nrof_images = int(nrof_images_per_class[i])
          for j in xrange(1,nrof_images):
              #self.time_reset()
              a_idx = emb_start_idx + j - 1
              #neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
              neg_dists_sqr = pdists[a_idx]
              #self.times[3] += self.time_elapsed()

              for pair in xrange(j, nrof_images): # For every possible positive pair.
                  p_idx = emb_start_idx + pair
                  #self.time_reset()
                  pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                  #self.times[4] += self.time_elapsed()
                  #self.time_reset()
                  neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                  if self.triplet_max_ap>0.0:
                    if pos_dist_sqr>self.triplet_max_ap:
                      continue
                  all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<self.triplet_alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                  #self.times[5] += self.time_elapsed()
                  #self.time_reset()
                  #all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                  nrof_random_negs = all_neg.shape[0]
                  if nrof_random_negs>0:
                      rnd_idx = np.random.randint(nrof_random_negs)
                      n_idx = all_neg[rnd_idx]
                      triplets.append( (a_idx, p_idx, n_idx) )
          emb_start_idx += nrof_images
      np.random.shuffle(triplets)
      return triplets

    def triplet_reset(self):
      #reset self.oseq by identities seq
      self.triplet_cur = 0
      ids = []
      for k in self.id2range:
        ids.append(k)
      random.shuffle(ids)
      self.triplet_seq = []
      for _id in ids:
        v = self.id2range[_id]
        _list = range(*v)
        random.shuffle(_list)
        if len(_list)>self.images_per_identity:
          _list = _list[0:self.images_per_identity]
        self.triplet_seq += _list
      print('triplet_seq', len(self.triplet_seq))
      assert len(self.triplet_seq)>=self.triplet_bag_size

    def time_reset(self):
      self.time_now = datetime.datetime.now()

    def time_elapsed(self):
      time_now = datetime.datetime.now()
      diff = time_now - self.time_now
      return diff.total_seconds()


    def select_triplets(self):
      self.seq = []
      while len(self.seq)<self.seq_min_size:
        self.time_reset()
        embeddings = None
        bag_size = self.triplet_bag_size
        batch_size = self.batch_size
        #data = np.zeros( (bag_size,)+self.data_shape )
        #label = np.zeros( (bag_size,) )
        tag = []
        #idx = np.zeros( (bag_size,) )
        print('eval %d images..'%bag_size, self.triplet_cur)
        print('triplet time stat', self.times)
        if self.triplet_cur+bag_size>len(self.triplet_seq):
          self.triplet_reset()
          #bag_size = min(bag_size, len(self.triplet_seq))
          print('eval %d images..'%bag_size, self.triplet_cur)
        self.times[0] += self.time_elapsed()
        self.time_reset()
        #print(data.shape)
        data = nd.zeros( self.provide_data[0][1] )
        label = None
        if self.provide_label is not None:
          label = nd.zeros( self.provide_label[0][1] )
        ba = 0
        while True:
          bb = min(ba+batch_size, bag_size)
          if ba>=bb:
            break
          _count = bb-ba
          #data = nd.zeros( (_count,)+self.data_shape )
          #_batch = self.data_iter.next()
          #_data = _batch.data[0].asnumpy()
          #print(_data.shape)
          #_label = _batch.label[0].asnumpy()
          #data[ba:bb,:,:,:] = _data
          #label[ba:bb] = _label
          for i in xrange(ba, bb):
            #print(ba, bb, self.triplet_cur, i, len(self.triplet_seq))
            _idx = self.triplet_seq[i+self.triplet_cur]
            s = self.imgrec.read_idx(_idx)
            header, img = recordio.unpack(s)
            img = self.imdecode(img)
            data[i-ba][:] = self.postprocess_data(img)
            _label = header.label
            if not isinstance(_label, numbers.Number):
              _label = _label[0]
            if label is not None:
              label[i-ba][:] = _label
            tag.append( ( int(_label), _idx) )
            #idx[i] = _idx

          db = mx.io.DataBatch(data=(data,))
          self.mx_model.forward(db, is_train=False)
          net_out = self.mx_model.get_outputs()
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
        assert len(tag)==bag_size
        self.triplet_cur+=bag_size
        embeddings = sklearn.preprocessing.normalize(embeddings)
        self.times[1] += self.time_elapsed()
        self.time_reset()
        nrof_images_per_class = [1]
        for i in xrange(1, bag_size):
          if tag[i][0]==tag[i-1][0]:
            nrof_images_per_class[-1]+=1
          else:
            nrof_images_per_class.append(1)
          
        triplets = self.pick_triplets(embeddings, nrof_images_per_class) # shape=(T,3)
        print('found triplets', len(triplets))
        ba = 0
        while True:
          bb = ba+self.per_batch_size//3
          if bb>len(triplets):
            break
          _triplets = triplets[ba:bb]
          for i in xrange(3):
            for triplet in _triplets:
              _pos = triplet[i]
              _idx = tag[_pos][1]
              self.seq.append(_idx)
          ba = bb
        self.times[2] += self.time_elapsed()

    def hard_mining_reset(self):
      #import faiss
      from annoy import AnnoyIndex
      data = nd.zeros( self.provide_data[0][1] )
      label = nd.zeros( self.provide_label[0][1] )
      #label = np.zeros( self.provide_label[0][1] )
      X = None
      ba = 0
      batch_num = 0
      while ba<len(self.oseq):
        batch_num+=1
        if batch_num%10==0:
          print('loading batch',batch_num, ba)
        bb = min(ba+self.batch_size, len(self.oseq))
        _count = bb-ba
        for i in xrange(_count):
          idx = self.oseq[i+ba]
          s = self.imgrec.read_idx(idx)
          header, img = recordio.unpack(s)
          img = self.imdecode(img)
          data[i][:] = self.postprocess_data(img)
          label[i][:] = header.label
        db = mx.io.DataBatch(data=(data,self.data_extra), label=(label,))
        self.mx_model.forward(db, is_train=False)
        net_out = self.mx_model.get_outputs()
        embedding = net_out[0].asnumpy()
        nembedding = sklearn.preprocessing.normalize(embedding)
        if _count<self.batch_size:
          nembedding = nembedding[0:_count,:]
        if X is None:
          X = np.zeros( (len(self.id2range), nembedding.shape[1]), dtype=np.float32 )
        nplabel = label.asnumpy()
        for i in xrange(_count):
          ilabel = int(nplabel[i])
          #print(ilabel, ilabel.__class__)
          X[ilabel] += nembedding[i]
        ba = bb
      X = sklearn.preprocessing.normalize(X)
      d = X.shape[1]
      t = AnnoyIndex(d, metric='euclidean')
      for i in xrange(X.shape[0]):
        t.add_item(i, X[i])
      print('start to build index')
      t.build(20)
      print(X.shape)
      k = self.per_identities
      self.seq = []
      for i in xrange(X.shape[0]):
        nnlist = t.get_nns_by_item(i, k)
        assert nnlist[0]==i
        for _label in nnlist:
          assert _label<len(self.id2range)
          _id = self.header0[0]+_label
          v = self.id2range[_id]
          _list = range(*v)
          if len(_list)<self.images_per_identity:
            random.shuffle(_list)
          else:
            _list = np.random.choice(_list, self.images_per_identity, replace=False)
          for i in xrange(self.images_per_identity):
            _idx = _list[i%len(_list)]
            self.seq.append(_idx)
      #faiss_params = [20,5]
      #quantizer = faiss.IndexFlatL2(d)  # the other index
      #index = faiss.IndexIVFFlat(quantizer, d, faiss_params[0], faiss.METRIC_L2)
      #assert not index.is_trained
      #index.train(X)
      #index.add(X)
      #assert index.is_trained
      #print('trained')
      #index.nprobe = faiss_params[1]
      #D, I = index.search(X, k)     # actual search
      #print(I.shape)
      #self.seq = []
      #for i in xrange(I.shape[0]):
      #  #assert I[i][0]==i
      #  for j in xrange(k):
      #    _label = I[i][j]
      #    assert _label<len(self.id2range)
      #    _id = self.header0[0]+_label
      #    v = self.id2range[_id]
      #    _list = range(*v)
      #    if len(_list)<self.images_per_identity:
      #      random.shuffle(_list)
      #    else:
      #      _list = np.random.choice(_list, self.images_per_identity, replace=False)
      #    for i in xrange(self.images_per_identity):
      #      _idx = _list[i%len(_list)]
      #      self.seq.append(_idx)

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        self.cur = 0
        if self.images_per_identity>0:
          if self.triplet_mode:
            self.select_triplets()
          elif not self.hard_mining:
            self.seq = []
            idlist = []
            for _id,v in self.id2range.iteritems():
              idlist.append((_id,range(*v)))
            for r in xrange(self.repeat):
              if r%10==0:
                print('repeat', r)
              if self.shuffle:
                random.shuffle(idlist)
              for item in idlist:
                _id = item[0]
                _list = item[1]
                #random.shuffle(_list)
                if len(_list)<self.images_per_identity:
                  random.shuffle(_list)
                else:
                  _list = np.random.choice(_list, self.images_per_identity, replace=False)
                for i in xrange(self.images_per_identity):
                  _idx = _list[i%len(_list)]
                  self.seq.append(_idx)
          else:
            self.hard_mining_reset()
          print('seq len', len(self.seq))
        else:
          if self.shuffle:
              random.shuffle(self.seq)
        if self.seq is None and self.imgrec is not None:
            self.imgrec.reset()

    def num_samples(self):
      return len(self.seq)

    def next_sample(self):
      while True:
        if self.cur >= len(self.seq):
            raise StopIteration
        idx = self.seq[self.cur]
        self.cur += 1
        s = self.imgrec.read_idx(idx)
        header, img = recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
          label = label[0]
        return label, img, None, None

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
        if not self.is_init:
          self.reset()
          self.is_init = True
        """Returns the next batch of data."""
        #print('in next', self.cur, self.labelcur)
        self.nbatch+=1
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        if self.provide_label is not None:
          batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                label, s, bbox, landmark = self.next_sample()
                _data = self.imdecode(s)
                if self.rand_mirror:
                  _rd = random.randint(0,1)
                  if _rd==1:
                    _data = mx.ndarray.flip(data=_data, axis=1)
                if self.cutoff>0:
                  centerh = random.randint(0, _data.shape[0]-1)
                  centerw = random.randint(0, _data.shape[1]-1)
                  half = self.cutoff//2
                  starth = max(0, centerh-half)
                  endh = min(_data.shape[0], centerh+half)
                  startw = max(0, centerw-half)
                  endw = min(_data.shape[1], centerw+half)
                  _data = _data.astype('float32')
                  #print(starth, endh, startw, endw, _data.shape)
                  _data[starth:endh, startw:endw, :] = 127.5
                #_npdata = _data.asnumpy()
                #if landmark is not None:
                #  _npdata = face_preprocess.preprocess(_npdata, bbox = bbox, landmark=landmark, image_size=self.image_size)
                #if self.rand_mirror:
                #  _npdata = self.mirror_aug(_npdata)
                #if self.mean is not None:
                #  _npdata = _npdata.astype(np.float32)
                #  _npdata -= self.mean
                #  _npdata *= 0.0078125
                #nimg = np.zeros(_npdata.shape, dtype=np.float32)
                #nimg[self.patch[1]:self.patch[3],self.patch[0]:self.patch[2],:] = _npdata[self.patch[1]:self.patch[3], self.patch[0]:self.patch[2], :]
                #_data = mx.nd.array(nimg)
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
                    if self.provide_label is not None:
                      batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i<batch_size:
                raise StopIteration

        #print('next end', batch_size, i)
        _label = None
        if self.provide_label is not None:
          _label = [batch_label]
        return io.DataBatch([batch_data], _label, batch_size - i)

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
        img = mx.image.imdecode(s) #mx.ndarray
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
    assert len(iter_list)>0
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


