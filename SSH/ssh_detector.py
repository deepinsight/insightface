from __future__ import print_function
import sys
import cv2
import mxnet as mx
from mxnet import ndarray as nd
import numpy as np
import numpy.random as npr
from distutils.util import strtobool

from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper


class SSHDetector:
  def __init__(self, prefix, epoch, ctx_id=0, test_mode=False):
    self.ctx_id = ctx_id
    self.ctx = mx.gpu(self.ctx_id)
    self.fpn_keys = []
    fpn_stride = []
    fpn_base_size = []
    self._feat_stride_fpn = [32, 16, 8]

    for s in self._feat_stride_fpn:
        self.fpn_keys.append('stride%s'%s)
        fpn_stride.append(int(s))
        fpn_base_size.append(16)

    self._scales = np.array([32,16,8,4,2,1])
    self._ratios = np.array([1.0]*len(self._feat_stride_fpn))
    self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(base_size=fpn_base_size, scales=self._scales, ratios=self._ratios)))
    self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
    self._rpn_pre_nms_top_n = 1000
    #self._rpn_post_nms_top_n = rpn_post_nms_top_n
    #self.score_threshold = 0.05
    self.nms_threshold = 0.3
    self._bbox_pred = nonlinear_pred
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    self.nms = gpu_nms_wrapper(self.nms_threshold, self.ctx_id)
    self.pixel_means = np.array([103.939, 116.779, 123.68]) #BGR

    if not test_mode:
      image_size = (640, 640)
      self.model = mx.mod.Module(symbol=sym, context=self.ctx, label_names = None)
      self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))], for_training=False)
      self.model.set_params(arg_params, aux_params)
    else:
      from rcnn.core.module import MutableModule
      image_size = (2400, 2400)
      data_shape = [('data', (1,3,image_size[0], image_size[1]))]
      self.model = MutableModule(symbol=sym, data_names=['data'], label_names=None,
                                context=self.ctx, max_data_shapes=data_shape)
      self.model.bind(data_shape, None, for_training=False)
      self.model.set_params(arg_params, aux_params)


  def detect(self, img, threshold=0.05, scales=[1.0]):
    proposals_list = []
    scores_list = []

    for im_scale in scales:

      if im_scale!=1.0:
        im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
      else:
        im = img
      im = im.astype(np.float32)
      #self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))], for_training=False)
      im_info = [im.shape[0], im.shape[1], im_scale]
      im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
      for i in range(3):
          im_tensor[0, i, :, :] = im[:, :, 2 - i] - self.pixel_means[2 - i]
      data = nd.array(im_tensor)
      db = mx.io.DataBatch(data=(data,), provide_data=[('data', data.shape)])
      self.model.forward(db, is_train=False)
      net_out = self.model.get_outputs()
      pre_nms_topN = self._rpn_pre_nms_top_n
      #post_nms_topN = self._rpn_post_nms_top_n
      #min_size_dict = self._rpn_min_size_fpn

      for s in self._feat_stride_fpn:
          if len(scales)>1 and s==32 and im_scale==scales[-1]:
            continue
          _key = 'stride%s'%s
          stride = int(s)
          idx = 0
          if s==16:
            idx=2
          elif s==8:
            idx=4
          print('getting', im_scale, stride, idx, len(net_out), data.shape, file=sys.stderr)
          scores = net_out[idx].asnumpy()
          #print(scores.shape)
          idx+=1
          #print('scores',stride, scores.shape, file=sys.stderr)
          scores = scores[:, self._num_anchors['stride%s'%s]:, :, :]
          bbox_deltas = net_out[idx].asnumpy()

          #if DEBUG:
          #    print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
          #    print 'scale: {}'.format(im_info[2])

          _height, _width = int(im_info[0] / stride), int(im_info[1] / stride)
          height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

          A = self._num_anchors['stride%s'%s]
          K = height * width

          anchors = anchors_plane(height, width, stride, self._anchors_fpn['stride%s'%s].astype(np.float32))
          #print((height, width), (_height, _width), anchors.shape, bbox_deltas.shape, scores.shape, file=sys.stderr)
          anchors = anchors.reshape((K * A, 4))

          #print('pre', bbox_deltas.shape, height, width)
          bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
          #print('after', bbox_deltas.shape, height, width)
          bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

          scores = self._clip_pad(scores, (height, width))
          scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

          #print(anchors.shape, bbox_deltas.shape, A, K, file=sys.stderr)
          proposals = self._bbox_pred(anchors, bbox_deltas)
          #proposals = anchors

          proposals = clip_boxes(proposals, im_info[:2])

          #keep = self._filter_boxes(proposals, min_size_dict['stride%s'%s] * im_info[2])
          #proposals = proposals[keep, :]
          #scores = scores[keep]
          #print('333', proposals.shape)

          scores_ravel = scores.ravel()
          order = scores_ravel.argsort()[::-1]
          if pre_nms_topN > 0:
              order = order[:pre_nms_topN]
          proposals = proposals[order, :]
          scores = scores[order]

          proposals /= im_scale

          proposals_list.append(proposals)
          scores_list.append(scores)

    proposals = np.vstack(proposals_list)
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    #if config.TEST.SCORE_THRESH>0.0:
    #  _count = np.sum(scores_ravel>config.TEST.SCORE_THRESH)
    #  order = order[:_count]
    #if pre_nms_topN > 0:
    #    order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    det = np.hstack((proposals, scores)).astype(np.float32)

    #if np.shape(det)[0] == 0:
    #    print("Something wrong with the input image(resolution is too low?), generate fake proposals for it.")
    #    proposals = np.array([[1.0, 1.0, 2.0, 2.0]]*post_nms_topN, dtype=np.float32)
    #    scores = np.array([[0.9]]*post_nms_topN, dtype=np.float32)
    #    det = np.array([[1.0, 1.0, 2.0, 2.0, 0.9]]*post_nms_topN, dtype=np.float32)

    
    if self.nms_threshold<1.0:
      keep = self.nms(det)
      det = det[keep, :]
    if threshold>0.0:
      keep = np.where(det[:, 4] >= threshold)[0]
      det = det[keep, :]
    return det

  @staticmethod
  def _filter_boxes(boxes, min_size):
      """ Remove all boxes with any side smaller than min_size """
      ws = boxes[:, 2] - boxes[:, 0] + 1
      hs = boxes[:, 3] - boxes[:, 1] + 1
      keep = np.where((ws >= min_size) & (hs >= min_size))[0]
      return keep

  @staticmethod
  def _clip_pad(tensor, pad_shape):
      """
      Clip boxes of the pad area.
      :param tensor: [n, c, H, W]
      :param pad_shape: [h, w]
      :return: [n, c, h, w]
      """
      H, W = tensor.shape[2:]
      h, w = pad_shape

      if h < H or w < W:
        tensor = tensor[:, :, :h, :w].copy()

      return tensor
