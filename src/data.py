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
import multiprocessing

logger = logging.getLogger()

def pick_triplets_impl(q_in, q_out):
  more = True
  while more:
      deq = q_in.get()
      if deq is None:
        more = False
      else:
        embeddings, emb_start_idx, nrof_images, alpha = deq
        print('running', emb_start_idx, nrof_images, os.getpid())
        for j in xrange(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                #all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    #triplets.append( (a_idx, p_idx, n_idx) )
                    q_out.put( (a_idx, p_idx, n_idx) )
        #emb_start_idx += nrof_images
  print('exit',os.getpid())

class FaceImageIter(io.DataIter):

    def __init__(self, batch_size, data_shape,
                 path_imgrec = None,
                 shuffle=False, aug_list=None, mean = None,
                 rand_mirror = False, cutoff = 0,
                 c2c_threshold = 0.0, output_c2c = 0, c2c_mode = -10, limit = 0,
                 ctx_num = 0, images_per_identity = 0, data_extra = None, hard_mining = False, 
                 triplet_params = None, coco_mode = False,
                 mx_model = None,
                 data_name='data', label_name='softmax_label', **kwargs):
        super(FaceImageIter, self).__init__()
        assert path_imgrec
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4]+".idx"
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            self.idx2cos = {}
            self.idx2flag = {}
            self.idx2meancos = {}
            self.c2c_auto = False
            #if output_c2c or c2c_threshold>0.0 or c2c_mode>=-5:
            #  path_c2c = os.path.join(os.path.dirname(path_imgrec), 'c2c')
            #  print(path_c2c)
            #  if os.path.exists(path_c2c):
            #    for line in open(path_c2c, 'r'):
            #      vec = line.strip().split(',')
            #      idx = int(vec[0])
            #      self.idx2cos[idx] = float(vec[1])
            #      self.idx2flag[idx] = 1
            #      if len(vec)>2:
            #        self.idx2flag[idx] = int(vec[2])
            #  else:
            #    self.c2c_auto = True
            #    self.c2c_step = 10000
            if header.flag>0:
              print('header0 label', header.label)
              self.header0 = (int(header.label[0]), int(header.label[1]))
              #assert(header.flag==1)
              self.imgidx = range(1, int(header.label[0]))
              if c2c_mode==0:
                imgidx2 = []
                for idx in self.imgidx:
                  c = self.idx2cos[idx]
                  f = self.idx2flag[idx]
                  if f!=1:
                    continue
                  imgidx2.append(idx)
                print('idx count', len(self.imgidx), len(imgidx2))
                self.imgidx = imgidx2
              elif c2c_mode==1:
                imgidx2 = []
                tmp = []
                for idx in self.imgidx:
                  c = self.idx2cos[idx]
                  f = self.idx2flag[idx]
                  if f==1:
                    imgidx2.append(idx)
                  else:
                    tmp.append( (idx, c) )
                tmp = sorted(tmp, key = lambda x:x[1])
                tmp = tmp[250000:300000]
                for _t in tmp:
                  imgidx2.append(_t[0])
                print('idx count', len(self.imgidx), len(imgidx2))
                self.imgidx = imgidx2
              elif c2c_mode==2:
                imgidx2 = []
                tmp = []
                for idx in self.imgidx:
                  c = self.idx2cos[idx]
                  f = self.idx2flag[idx]
                  if f==1:
                    imgidx2.append(idx)
                  else:
                    tmp.append( (idx, c) )
                tmp = sorted(tmp, key = lambda x:x[1])
                tmp = tmp[200000:300000]
                for _t in tmp:
                  imgidx2.append(_t[0])
                print('idx count', len(self.imgidx), len(imgidx2))
                self.imgidx = imgidx2
              elif c2c_mode==-2:
                imgidx2 = []
                for idx in self.imgidx:
                  c = self.idx2cos[idx]
                  f = self.idx2flag[idx]
                  if f==2:
                    continue
                  if c<0.73:
                    continue
                  imgidx2.append(idx)
                print('idx count', len(self.imgidx), len(imgidx2))
                self.imgidx = imgidx2
              elif c2c_threshold>0.0:
                imgidx2 = []
                for idx in self.imgidx:
                  c = self.idx2cos[idx]
                  f = self.idx2flag[idx]
                  if c<c2c_threshold:
                    continue
                  imgidx2.append(idx)
                print(len(self.imgidx), len(imgidx2))
                self.imgidx = imgidx2
              self.id2range = {}
              self.seq_identity = range(int(header.label[0]), int(header.label[1]))
              c2c_stat = [0,0]
              for identity in self.seq_identity:
                s = self.imgrec.read_idx(identity)
                header, _ = recordio.unpack(s)
                a,b = int(header.label[0]), int(header.label[1])
                self.id2range[identity] = (a,b)
                count = b-a
                if count>=output_c2c:
                  c2c_stat[1]+=1
                else:
                  c2c_stat[0]+=1
                for ii in xrange(a,b):
                  self.idx2flag[ii] = count
                if len(self.idx2cos)>0:
                  m = 0.0
                  for ii in xrange(a,b):
                    m+=self.idx2cos[ii]
                  m/=(b-a)
                  for ii in xrange(a,b):
                    self.idx2meancos[ii] = m
                  #self.idx2meancos[identity] = m

              print('id2range', len(self.id2range))
              print(len(self.idx2cos), len(self.idx2meancos), len(self.idx2flag))
              print('c2c_stat', c2c_stat)
              if limit>0 and limit<len(self.imgidx):
                random.seed(727)
                prob = float(limit)/len(self.imgidx)
                new_imgidx = []
                new_ids = 0
                for identity in self.seq_identity:
                  s = self.imgrec.read_idx(identity)
                  header, _ = recordio.unpack(s)
                  a,b = int(header.label[0]), int(header.label[1])
                  found = False
                  for _idx in xrange(a,b):
                    if random.random()<prob:
                      found = True
                      new_imgidx.append(_idx)
                  if found:
                    new_ids+=1
                self.imgidx = new_imgidx
                print('new ids', new_ids)
                random.seed(None)
                #random.Random(727).shuffle(self.imgidx)
                #self.imgidx = self.imgidx[0:limit]
            else:
              self.imgidx = list(self.imgrec.keys)
            if shuffle:
              self.seq = self.imgidx
              self.oseq = self.imgidx
              print(len(self.seq))
            else:
              self.seq = None

        self.mean = mean
        self.nd_mean = None
        if self.mean:
          self.mean = np.array(self.mean, dtype=np.float32).reshape(1,1,3)
          self.nd_mean = mx.nd.array(self.mean).reshape((1,1,3))

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
        self.c2c_threshold = c2c_threshold
        self.output_c2c = output_c2c
        self.per_batch_size = int(self.batch_size/self.ctx_num)
        self.images_per_identity = images_per_identity
        if self.images_per_identity>0:
          self.identities = int(self.per_batch_size/self.images_per_identity)
          self.per_identities = self.identities
          self.repeat = 3000000.0/(self.images_per_identity*len(self.id2range))
          self.repeat = int(self.repeat)
          print(self.images_per_identity, self.identities, self.repeat)
        self.data_extra = None
        if data_extra is not None:
          self.data_extra = nd.array(data_extra)
          self.provide_data = [(data_name, (batch_size,) + data_shape), ('extra', data_extra.shape)]
        self.hard_mining = hard_mining
        self.mx_model = mx_model
        if self.hard_mining:
          assert self.images_per_identity>0
          assert self.mx_model is not None
        self.triplet_params = triplet_params
        self.triplet_mode = False
        self.coco_mode = coco_mode
        if len(label_name)>0:
          if output_c2c:
            self.provide_label = [(label_name, (batch_size,2))]
          else:
            self.provide_label = [(label_name, (batch_size,))]
        else:
          self.provide_label = []
        print(self.provide_label[0][1])
        if self.coco_mode:
          assert self.triplet_params is None
          assert self.images_per_identity>0
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
          self.triplet_oseq_cur = 0
          self.triplet_oseq_reset()
          self.seq_min_size = self.batch_size*2
        self.cur = 0
        self.nbatch = 0
        self.is_init = False
        self.times = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #self.reset()


    def ____pick_triplets(self, embeddings, nrof_images_per_class):
      emb_start_idx = 0
      people_per_batch = len(nrof_images_per_class)
      nrof_threads = 8
      q_in = multiprocessing.Queue()
      q_out = multiprocessing.Queue()
      processes = [multiprocessing.Process(target=pick_triplets_impl, args=(q_in, q_out)) \
                      for i in range(nrof_threads)]
      for p in processes:
          p.start()
      
      # VGG Face: Choosing good triplets is crucial and should strike a balance between
      #  selecting informative (i.e. challenging) examples and swamping training with examples that
      #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
      #  the image n at random, but only between the ones that violate the triplet loss margin. The
      #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
      #  choosing the maximally violating example, as often done in structured output learning.

      for i in xrange(people_per_batch):
          nrof_images = int(nrof_images_per_class[i])
          job = (embeddings, emb_start_idx, nrof_images, self.triplet_alpha)
          emb_start_idx+=nrof_images
          q_in.put(job)
      for i in xrange(nrof_threads):
        q_in.put(None)
      print('joining')
      for p in processes:
          p.join()
      print('joined')
      q_out.put(None)

      triplets = []
      more = True
      while more:
        triplet = q_out.get()
        if triplet is None:
          more = False
        else:
          triplets.append(triplets)
      np.random.shuffle(triplets)
      return triplets

    #cal pairwise dists on single gpu
    def _pairwise_dists(self, embeddings):
      nd_embedding = mx.nd.array(embeddings, mx.gpu(0))
      pdists = []
      for idx in xrange(embeddings.shape[0]):
        a_embedding = nd_embedding[idx]
        body = mx.nd.broadcast_sub(a_embedding, nd_embedding)
        body = body*body
        body = mx.nd.sum_axis(body, axis=1)
        ret = body.asnumpy()
        #print(ret.shape)
        pdists.append(ret)
      return pdists

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

    def __pick_triplets(self, embeddings, nrof_images_per_class):
      emb_start_idx = 0
      triplets = []
      people_per_batch = len(nrof_images_per_class)
      
      for i in xrange(people_per_batch):
          nrof_images = int(nrof_images_per_class[i])
          if nrof_images<2:
            continue
          for j in xrange(1,nrof_images):
              a_idx = emb_start_idx + j - 1
              pcount = nrof_images-1
              dists_a2all = np.sum(np.square(embeddings[a_idx] - embeddings), 1) #(N,)
              #print(a_idx, dists_a2all.shape)
              ba = emb_start_idx
              bb = emb_start_idx+nrof_images
              sorted_idx = np.argsort(dists_a2all)
              #print('assert', sorted_idx[0], a_idx)
              #assert sorted_idx[0]==a_idx
              #for idx in sorted_idx:
              #  print(idx, dists_a2all[idx])
              p2n_map = {}
              pfound = 0
              for idx in sorted_idx:
                if idx==a_idx: #is anchor
                  continue
                if idx<bb and idx>=ba: #is pos
                  p2n_map[idx] = [dists_a2all[idx], []] #ap, [neg_list]
                  pfound+=1
                else: # is neg
                  an = dists_a2all[idx]
                  if pfound==pcount and len(p2n_map)==0:
                    break
                  to_del = []
                  for p_idx in p2n_map:
                    v = p2n_map[p_idx]
                    an_ap = an - v[0]
                    if an_ap<self.triplet_alpha:
                      v[1].append(idx)
                    else:
                      #output
                      if len(v[1])>0:
                        n_idx = random.choice(v[1])
                        triplets.append( (a_idx, p_idx, n_idx) )
                      to_del.append(p_idx)
                  for _del in to_del:
                    del p2n_map[_del]
              for p_idx,v in p2n_map.iteritems():
                if len(v[1])>0:
                  n_idx = random.choice(v[1])
                  triplets.append( (a_idx, p_idx, n_idx) )
          emb_start_idx += nrof_images
      np.random.shuffle(triplets)
      return triplets

    def triplet_oseq_reset(self):
      #reset self.oseq by identities seq
      self.triplet_oseq_cur = 0
      ids = []
      for k in self.id2range:
        ids.append(k)
      random.shuffle(ids)
      self.oseq = []
      for _id in ids:
        v = self.id2range[_id]
        _list = range(*v)
        random.shuffle(_list)
        if len(_list)>self.images_per_identity:
          _list = _list[0:self.images_per_identity]
        self.oseq += _list
      print('oseq', len(self.oseq))

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
        print('eval %d images..'%bag_size, self.triplet_oseq_cur)
        print('triplet time stat', self.times)
        if self.triplet_oseq_cur+bag_size>len(self.oseq):
          self.triplet_oseq_reset()
          print('eval %d images..'%bag_size, self.triplet_oseq_cur)
        self.times[0] += self.time_elapsed()
        self.time_reset()
        #print(data.shape)
        data = nd.zeros( self.provide_data[0][1] )
        label = nd.zeros( self.provide_label[0][1] )
        ba = 0
        while True:
          bb = min(ba+batch_size, bag_size)
          if ba>=bb:
            break
          #_batch = self.data_iter.next()
          #_data = _batch.data[0].asnumpy()
          #print(_data.shape)
          #_label = _batch.label[0].asnumpy()
          #data[ba:bb,:,:,:] = _data
          #label[ba:bb] = _label
          for i in xrange(ba, bb):
            _idx = self.oseq[i+self.triplet_oseq_cur]
            s = self.imgrec.read_idx(_idx)
            header, img = recordio.unpack(s)
            img = self.imdecode(img)
            data[i-ba][:] = self.postprocess_data(img)
            label[i-ba][:] = header.label
            tag.append( ( int(header.label), _idx) )
            #idx[i] = _idx

          db = mx.io.DataBatch(data=(data,), label=(label,))
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
        self.triplet_oseq_cur+=bag_size
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

    def triplet_reset(self):
      self.select_triplets()

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
    def reset_c2c(self):
      self.select_triplets()
      for identity,v in self.id2range.iteritems():
        _list = range(*v)
      
        for idx in _list:
          s = imgrec.read_idx(idx)
          ocontents.append(s)
        embeddings = None
        #print(len(ocontents))
        ba = 0
        while True:
          bb = min(ba+args.batch_size, len(ocontents))
          if ba>=bb:
            break
          _batch_size = bb-ba
          _batch_size2 = max(_batch_size, args.ctx_num)
          data = nd.zeros( (_batch_size2,3, image_size[0], image_size[1]) )
          label = nd.zeros( (_batch_size2,) )
          count = bb-ba
          ii=0
          for i in xrange(ba, bb):
            header, img = mx.recordio.unpack(ocontents[i])
            img = mx.image.imdecode(img)
            img = nd.transpose(img, axes=(2, 0, 1))
            data[ii][:] = img
            label[ii][:] = header.label
            ii+=1
          while ii<_batch_size2:
            data[ii][:] = data[0][:]
            label[ii][:] = label[0][:]
            ii+=1
          db = mx.io.DataBatch(data=(data,), label=(label,))
          self.mx_model.forward(db, is_train=False)
          net_out = self.mx_model.get_outputs()
          net_out = net_out[0].asnumpy()
          model.forward(db, is_train=False)
          net_out = model.get_outputs()
          net_out = net_out[0].asnumpy()
          if embeddings is None:
            embeddings = np.zeros( (len(ocontents), net_out.shape[1]))
          embeddings[ba:bb,:] = net_out[0:_batch_size,:]
          ba = bb
        embeddings = sklearn.preprocessing.normalize(embeddings)
        embedding = np.mean(embeddings, axis=0, keepdims=True)
        embedding = sklearn.preprocessing.normalize(embedding)
        sims = np.dot(embeddings, embedding).flatten()
        assert len(sims)==len(_list)
        for i in xrange(len(_list)):
          _idx = _list[i]
          self.idx2cos[_idx] = sims[i]

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        if self.c2c_auto:
          self.reset_c2c()
        self.cur = 0
        if self.images_per_identity>0:
          if self.triplet_mode:
            self.triplet_reset()
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
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        if self.seq is not None:
          while True:
            if self.cur >= len(self.seq):
                raise StopIteration
            idx = self.seq[self.cur]
            self.cur += 1
            if self.imgrec is not None:
              s = self.imgrec.read_idx(idx)
              header, img = recordio.unpack(s)
              label = header.label
              if self.output_c2c:
                count = self.idx2flag[idx]
                if self.output_c2c==1:
                  v = np.random.uniform(0.4, 0.5)
                elif self.output_c2c==2:
                  v = np.random.uniform(0.4, 0.5)
                  if count>=self.output_c2c:
                    v = np.random.uniform(0.3, 0.4)
                elif self.output_c2c==3:
                  v = (9.5 - math.log(2.0+count))/10.0
                  v = min(max(v, 0.3), 0.5)
                elif self.output_c2c==4:
                  mu = 0.0
                  sigma = 0.1
                  mrange = [0.4,0.5]
                  v = numpy.random.normal(mu, sigma)
                  v = math.abs(v)*-1.0+mrange[1]
                  v = max(v, mrange[0])
                elif self.output_c2c==5:
                  v = np.random.uniform(0.41, 0.51)
                  if count>=175:
                    v = np.random.uniform(0.37, 0.47)
                elif self.output_c2c==6:
                  v = np.random.uniform(0.41, 0.51)
                  if count>=175:
                    v = np.random.uniform(0.38, 0.48)
                else:
                  assert False

                label = [label, v]
              else:
                if not isinstance(label, numbers.Number):
                  label = label[0]
              return label, img, None, None
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
                if self.nd_mean is not None:
                    _data = _data.astype('float32')
                    _data -= self.nd_mean
                    _data *= 0.0078125
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
                      if not self.coco_mode:
                        if len(batch_label.shape)==1:
                          batch_label[i][:] = label
                        else:
                          for ll in xrange(batch_label.shape[1]):
                            v = label[ll]
                            if ll>0:
                              #c2c = v
                              #_param = [0.5, 0.4, 0.85, 0.75]
                              #_a = (_param[1]-_param[0])/(_param[3]-_param[2])
                              #m = _param[1]+_a*(c2c-_param[3])
                              #m = min(_param[0], max(_param[1],m))
                              #v = math.cos(m)
                              #v = v*v
                              m = v
                              v = math.cos(m)
                              v = v*v
                              #print('m', i,m,v)

                            batch_label[i][ll] = v
                      else:
                        batch_label[i][:] = (i%self.per_batch_size)//self.images_per_identity
                    i += 1
        except StopIteration:
            if i<batch_size:
                raise StopIteration

        #print('next end', batch_size, i)
        _label = None
        if self.provide_label is not None:
          _label = [batch_label]
        if self.data_extra is not None:
          return io.DataBatch([batch_data, self.data_extra], _label, batch_size - i)
        else:
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


