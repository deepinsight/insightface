"""
RPN:
data =
    {'data': [num_images, c, h, w],
     'im_info': [num_images, 4] (optional)}
label =
    {'gt_boxes': [num_boxes, 5] (optional),
     'label': [batch_size, 1] <- [batch_size, num_anchors, feat_height, feat_width],
     'bbox_target': [batch_size, num_anchors, feat_height, feat_width],
     'bbox_weight': [batch_size, num_anchors, feat_height, feat_width]}
"""

from __future__ import print_function
import sys
import logging
import datetime
import numpy as np
import numpy.random as npr

from ..logger import logger
from ..config import config
from .image import get_image, tensor_vstack, get_crop_image
from ..processing.generate_anchor import generate_anchors, anchors_plane
from ..processing.bbox_transform import bbox_overlaps, bbox_transform, landmark_transform

STAT = {0:0, 8:0, 16:0, 32:0}

def get_rpn_testbatch(roidb):
    """
    return a dict of testbatch
    :param roidb: ['image', 'flipped']
    :return: data, label, im_info
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

    data = {'data': im_array,
            'im_info': im_info}
    label = {}

    return data, label, im_info

def get_rpn_batch(roidb):
    """
    prototype for rpn batch: data, im_info, gt_boxes
    :param roidb: ['image', 'flipped'] + ['gt_boxes', 'boxes', 'gt_classes']
    :return: data, label
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

    # gt boxes: (x1, y1, x2, y2, cls)
    if roidb[0]['gt_classes'].size > 0:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((roidb[0]['boxes'].shape[0], 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    else:
        gt_boxes = np.empty((0, 5), dtype=np.float32)

    data = {'data': im_array,
            'im_info': im_info}
    label = {'gt_boxes': gt_boxes}

    return data, label

def get_crop_batch(roidb):
    """
    prototype for rpn batch: data, im_info, gt_boxes
    :param roidb: ['image', 'flipped'] + ['gt_boxes', 'boxes', 'gt_classes']
    :return: data, label
    """
    #assert len(roidb) == 1, 'Single batch only'
    data_list = []
    label_list = []
    imgs, roidb = get_crop_image(roidb)
    assert len(imgs)==len(roidb)
    for i in range(len(imgs)):
      im_array = imgs[i]
      im_info = np.array([roidb[i]['im_info']], dtype=np.float32)

      # gt boxes: (x1, y1, x2, y2, cls)
      if roidb[i]['gt_classes'].size > 0:
          gt_inds = np.where(roidb[i]['gt_classes'] != 0)[0]
          gt_boxes = np.empty((roidb[i]['boxes'].shape[0], 5), dtype=np.float32)
          gt_boxes[:, 0:4] = roidb[i]['boxes'][gt_inds, :]
          gt_boxes[:, 4] = roidb[i]['gt_classes'][gt_inds]
          if config.USE_BLUR:
            gt_blur = roidb[i]['blur']
          if config.FACE_LANDMARK:
            #gt_landmarks = np.empty((roidb[i]['landmarks'].shape[0], 11), dtype=np.float32)
            gt_landmarks = roidb[i]['landmarks'][gt_inds,:,:]
          if config.HEAD_BOX:
            gt_boxes_head = np.empty((roidb[i]['boxes_head'].shape[0], 5), dtype=np.float32)
            gt_boxes_head[:, 0:4] = roidb[i]['boxes_head'][gt_inds, :]
            gt_boxes_head[:, 4] = roidb[i]['gt_classes'][gt_inds]
      else:
          gt_boxes = np.empty((0, 5), dtype=np.float32)
          if config.USE_BLUR:
            gt_blur = np.empty((0,), dtype=np.float32)
          if config.FACE_LANDMARK:
            gt_landmarks = np.empty((0, 5, 3), dtype=np.float32)
          if config.HEAD_BOX:
            gt_boxes_head = np.empty((0, 5), dtype=np.float32)

      data = {'data': im_array,
              'im_info': im_info}
      label = {'gt_boxes': gt_boxes}
      if config.USE_BLUR:
        label['gt_blur'] = gt_blur
      if config.FACE_LANDMARK:
        label['gt_landmarks'] = gt_landmarks
      if config.HEAD_BOX:
        label['gt_boxes_head'] = gt_boxes_head
      data_list.append(data)
      label_list.append(label)

    return data_list, label_list

def assign_anchor_fpn(feat_shape, gt_label, im_info, landmark=False, prefix='face', select_stride=0):
    """
    assign ground truth boxes to anchor positions
    :param feat_shape: infer output shape
    :param gt_boxes: assign ground truth
    :param im_info: filter out anchors overlapped with edges
    :return: tuple
    labels: of shape (batch_size, 1) <- (batch_size, num_anchors, feat_height, feat_width)
    bbox_targets: of shape (batch_size, num_anchors * 4, feat_height, feat_width)
    bbox_weights: mark the assigned anchors
    """
    def _unmap(data, count, inds, fill=0):
        """" unmap a subset inds of data into original data of size count """
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    global STAT
    DEBUG = False

    im_info = im_info[0]
    gt_boxes = gt_label['gt_boxes']
    # clean up boxes
    nonneg = np.where(gt_boxes[:, 4] != -1)[0]
    gt_boxes = gt_boxes[nonneg]
    if config.USE_BLUR:
      gt_blur = gt_label['gt_blur']
      gt_blur = gt_blur[nonneg]
    if landmark:
      gt_landmarks = gt_label['gt_landmarks']
      gt_landmarks = gt_landmarks[nonneg]
      assert gt_boxes.shape[0]==gt_landmarks.shape[0]
    #scales = np.array(scales, dtype=np.float32)
    feat_strides = config.RPN_FEAT_STRIDE
    bbox_pred_len = 4
    landmark_pred_len = 10
    if config.USE_BLUR:
      gt_boxes[:,4] = gt_blur
      bbox_pred_len = 5
    if config.USE_OCCLUSION:
      landmark_pred_len = 15

    anchors_list = []
    anchors_num_list = []
    inds_inside_list = []
    feat_infos = []
    A_list = []
    for i in range(len(feat_strides)):
        stride = feat_strides[i]
        sstride = str(stride)
        base_size = config.RPN_ANCHOR_CFG[sstride]['BASE_SIZE']
        allowed_border = config.RPN_ANCHOR_CFG[sstride]['ALLOWED_BORDER']
        ratios = config.RPN_ANCHOR_CFG[sstride]['RATIOS']
        scales = config.RPN_ANCHOR_CFG[sstride]['SCALES']
        base_anchors = generate_anchors(base_size=base_size, ratios=list(ratios), scales=np.array(scales, dtype=np.float32), stride = stride, dense_anchor = config.DENSE_ANCHOR)
        num_anchors = base_anchors.shape[0]
        feat_height, feat_width = feat_shape[i][-2:]
        feat_stride = feat_strides[i]
        feat_infos.append([feat_height, feat_width])

        A = num_anchors
        A_list.append(A)
        K = feat_height * feat_width

        all_anchors = anchors_plane(feat_height, feat_width, feat_stride, base_anchors)
        all_anchors = all_anchors.reshape((K * A, 4))
        #print('anchor0', stride, all_anchors[0])

        total_anchors = int(K * A)
        anchors_num_list.append(total_anchors)
        # only keep anchors inside the image
        inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                               (all_anchors[:, 1] >= -allowed_border) &
                               (all_anchors[:, 2] < im_info[1] + allowed_border) &
                               (all_anchors[:, 3] < im_info[0] + allowed_border))[0]
        if DEBUG:
            print('total_anchors', total_anchors)
            print('inds_inside', len(inds_inside))

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        #print('AA', anchors.shape, len(inds_inside))

        anchors_list.append(anchors)
        inds_inside_list.append(inds_inside)

    # Concat anchors from each level
    anchors = np.concatenate(anchors_list)
    for i in range(1, len(inds_inside_list)):
        inds_inside_list[i] = inds_inside_list[i] + sum(anchors_num_list[:i])
    inds_inside = np.concatenate(inds_inside_list)
    total_anchors = sum(anchors_num_list)
    #print('total_anchors', anchors.shape[0], len(inds_inside), file=sys.stderr)

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)
    #print('BB', anchors.shape, len(inds_inside))
    #print('gt_boxes', gt_boxes.shape, file=sys.stderr)

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        #print('AAA', argmax_overlaps.shape)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        if config.TRAIN.RPN_FORCE_POSITIVE:
          labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
        labels[max_overlaps >= config.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    else:
        labels[:] = 0
    fg_inds = np.where(labels == 1)[0]
    #print('fg count', len(fg_inds))

    # subsample positive labels if we have too many
    if config.TRAIN.RPN_ENABLE_OHEM==0:
      fg_inds = np.where(labels == 1)[0]
      num_fg = int(config.TRAIN.RPN_FG_FRACTION * config.TRAIN.RPN_BATCH_SIZE)
      if len(fg_inds) > num_fg:
          disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
          if DEBUG:
              disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
          labels[disable_inds] = -1

      # subsample negative labels if we have too many
      num_bg = config.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
      bg_inds = np.where(labels == 0)[0]
      if len(bg_inds) > num_bg:
          disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
          if DEBUG:
              disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
          labels[disable_inds] = -1

      #fg_inds = np.where(labels == 1)[0]
      #num_fg = len(fg_inds)
      #num_bg = num_fg*int(1.0/config.TRAIN.RPN_FG_FRACTION-1)

      #bg_inds = np.where(labels == 0)[0]
      #if len(bg_inds) > num_bg:
      #    disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
      #    if DEBUG:
      #        disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
      #    labels[disable_inds] = -1
    else:
      fg_inds = np.where(labels == 1)[0]
      num_fg = len(fg_inds)
      bg_inds = np.where(labels == 0)[0]
      num_bg = len(bg_inds)

    #print('anchor stat', num_fg, num_bg)


    bbox_targets = np.zeros((len(inds_inside), bbox_pred_len), dtype=np.float32)
    if gt_boxes.size > 0:
        #print('GT', gt_boxes.shape, gt_boxes[argmax_overlaps, :4].shape)
        bbox_targets[:,:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :])
        #bbox_targets[:,4] = gt_blur

    bbox_weights = np.zeros((len(inds_inside), bbox_pred_len), dtype=np.float32)
    #bbox_weights[labels == 1, :] = np.array(config.TRAIN.RPN_BBOX_WEIGHTS)
    bbox_weights[labels == 1, 0:4] = 1.0
    if bbox_pred_len>4:
      bbox_weights[labels == 1, 4:bbox_pred_len] = 0.1

    if landmark:
      landmark_targets = np.zeros((len(inds_inside), landmark_pred_len), dtype=np.float32)
      #landmark_weights = np.zeros((len(inds_inside), 10), dtype=np.float32)
      landmark_weights = np.zeros((len(inds_inside), landmark_pred_len), dtype=np.float32)
      #landmark_weights[labels == 1, :] = np.array(config.TRAIN.RPN_LANDMARK_WEIGHTS)
      if landmark_pred_len==10:
        landmark_weights[labels == 1, :] = 1.0
      elif landmark_pred_len==15:
        v = [1.0, 1.0, 0.1] * 5
        assert len(v)==15
        landmark_weights[labels == 1, :] = np.array(v)
      else:
        assert False
      #TODO here
      if gt_landmarks.size > 0:
        #print('AAA',argmax_overlaps)
        a_landmarks = gt_landmarks[argmax_overlaps,:,:]
        landmark_targets[:] = landmark_transform(anchors, a_landmarks)
        invalid = np.where(a_landmarks[:,0,2]<0.0)[0]
        #assert len(invalid)==0
        #landmark_weights[invalid, :] = np.array(config.TRAIN.RPN_INVALID_LANDMARK_WEIGHTS)
        landmark_weights[invalid, :] = 0.0

    #if DEBUG:
    #    _sums = bbox_targets[labels == 1, :].sum(axis=0)
    #    _squared_sums = (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
    #    _counts = np.sum(labels == 1)
    #    means = _sums / (_counts + 1e-14)
    #    stds = np.sqrt(_squared_sums / _counts - means ** 2)
    #    print 'means', means
    #    print 'stdevs', stds
    # map up to original set of anchors
    #print(labels.shape, total_anchors, inds_inside.shape, inds_inside[0], inds_inside[-1])
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_weights = _unmap(bbox_weights, total_anchors, inds_inside, fill=0)
    if landmark:
      landmark_targets = _unmap(landmark_targets, total_anchors, inds_inside, fill=0)
      landmark_weights = _unmap(landmark_weights, total_anchors, inds_inside, fill=0)
    #print('CC', anchors.shape, len(inds_inside))

    #if DEBUG:
    #    if gt_boxes.size > 0:
    #        print 'rpn: max max_overlaps', np.max(max_overlaps)
    #    print 'rpn: num_positives', np.sum(labels == 1)
    #    print 'rpn: num_negatives', np.sum(labels == 0)
    #    _fg_sum = np.sum(labels == 1)
    #    _bg_sum = np.sum(labels == 0)
    #    _count = 1
    #    print 'rpn: num_positive avg', _fg_sum / _count
    #    print 'rpn: num_negative avg', _bg_sum / _count

    # resahpe
    label_list = list()
    bbox_target_list = list()
    bbox_weight_list = list()
    if landmark:
      landmark_target_list = list()
      landmark_weight_list = list()
    anchors_num_range = [0] + anchors_num_list
    label = {}
    for i in range(len(feat_strides)):
        stride = feat_strides[i]
        feat_height, feat_width = feat_infos[i]
        A = A_list[i]
        _label = labels[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
        if select_stride>0 and stride!=select_stride:
          #print('set', stride, select_stride)
          _label[:] = -1
        #print('_label', _label.shape, select_stride)
        #_fg_inds = np.where(_label == 1)[0]
        #n_fg = len(_fg_inds)
        #STAT[0]+=1
        #STAT[stride]+=n_fg
        #if STAT[0]%100==0:
        #  print('rpn_stat', STAT, file=sys.stderr)
        bbox_target = bbox_targets[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
        bbox_weight = bbox_weights[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
        if landmark:
          landmark_target = landmark_targets[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
          landmark_weight = landmark_weights[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]

        _label = _label.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
        _label = _label.reshape((1, A * feat_height * feat_width))
        bbox_target = bbox_target.reshape((1, feat_height*feat_width, A * bbox_pred_len)).transpose(0, 2, 1)
        bbox_weight = bbox_weight.reshape((1, feat_height*feat_width, A * bbox_pred_len)).transpose((0, 2, 1))
        label['%s_label_stride%d'%(prefix, stride)] = _label
        label['%s_bbox_target_stride%d'%(prefix,stride)] = bbox_target
        label['%s_bbox_weight_stride%d'%(prefix,stride)] = bbox_weight
        if landmark:
          landmark_target = landmark_target.reshape((1, feat_height*feat_width, A * landmark_pred_len)).transpose(0, 2, 1)
          landmark_weight = landmark_weight.reshape((1, feat_height*feat_width, A * landmark_pred_len)).transpose((0, 2, 1))
          label['%s_landmark_target_stride%d'%(prefix,stride)] = landmark_target
          label['%s_landmark_weight_stride%d'%(prefix,stride)] = landmark_weight
        #print('in_rpn', stride,_label.shape, bbox_target.shape, bbox_weight.shape, file=sys.stderr)
        label_list.append(_label)
        #print('DD', _label.shape)
        bbox_target_list.append(bbox_target)
        bbox_weight_list.append(bbox_weight)
        if landmark:
          landmark_target_list.append(landmark_target)
          landmark_weight_list.append(landmark_weight)

    label_concat = np.concatenate(label_list, axis=1)
    bbox_target_concat = np.concatenate(bbox_target_list, axis=2)
    bbox_weight_concat = np.concatenate(bbox_weight_list, axis=2)
    #fg_inds = np.where(label_concat[0] == 1)[0]
    #print('fg_inds_in_rpn2', fg_inds, file=sys.stderr)

    label.update({'%s_label'%prefix: label_concat,
            '%s_bbox_target'%prefix: bbox_target_concat,
            '%s_bbox_weight'%prefix: bbox_weight_concat}
            )
    if landmark:
      landmark_target_concat = np.concatenate(landmark_target_list, axis=2)
      landmark_weight_concat = np.concatenate(landmark_weight_list, axis=2)
      label['%s_landmark_target'%prefix] = landmark_target_concat
      label['%s_landmark_weight'%prefix] = landmark_weight_concat
    return label


class AA:
  def __init__(self, feat_shape):
    self.feat_shape = feat_shape
    feat_strides = config.RPN_FEAT_STRIDE
    anchors_list = []
    anchors_num_list = []
    inds_inside_list = []
    feat_infos = []
    A_list = []
    DEBUG = False
    for i in range(len(feat_strides)):
        stride = feat_strides[i]
        sstride = str(stride)
        base_size = config.RPN_ANCHOR_CFG[sstride]['BASE_SIZE']
        allowed_border = config.RPN_ANCHOR_CFG[sstride]['ALLOWED_BORDER']
        ratios = config.RPN_ANCHOR_CFG[sstride]['RATIOS']
        scales = config.RPN_ANCHOR_CFG[sstride]['SCALES']
        base_anchors = generate_anchors(base_size=base_size, ratios=list(ratios), scales=np.array(scales, dtype=np.float32), stride = stride, dense_anchor = config.DENSE_ANCHOR)
        num_anchors = base_anchors.shape[0]
        feat_height, feat_width = feat_shape[i][-2:]
        feat_stride = feat_strides[i]
        feat_infos.append([feat_height, feat_width])

        A = num_anchors
        A_list.append(A)
        K = feat_height * feat_width

        all_anchors = anchors_plane(feat_height, feat_width, feat_stride, base_anchors)
        all_anchors = all_anchors.reshape((K * A, 4))
        #print('anchor0', stride, all_anchors[0])

        total_anchors = int(K * A)
        anchors_num_list.append(total_anchors)
        # only keep anchors inside the image
        inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                               (all_anchors[:, 1] >= -allowed_border) &
                               (all_anchors[:, 2] < config.SCALES[0][1] + allowed_border) &
                               (all_anchors[:, 3] < config.SCALES[0][1] + allowed_border))[0]
        if DEBUG:
            print('total_anchors', total_anchors)
            print('inds_inside', len(inds_inside))

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        #print('AA', anchors.shape, len(inds_inside))

        anchors_list.append(anchors)
        inds_inside_list.append(inds_inside)
    anchors = np.concatenate(anchors_list)
    for i in range(1, len(inds_inside_list)):
        inds_inside_list[i] = inds_inside_list[i] + sum(anchors_num_list[:i])
    inds_inside = np.concatenate(inds_inside_list)
    #self.anchors_list = anchors_list
    #self.inds_inside_list = inds_inside_list
    self.anchors = anchors
    self.inds_inside = inds_inside
    self.anchors_num_list = anchors_num_list
    self.feat_infos = feat_infos
    self.A_list = A_list
    self._times = [0.0, 0.0, 0.0, 0.0]

  @staticmethod
  def _unmap(data, count, inds, fill=0):
      """" unmap a subset inds of data into original data of size count """
      if len(data.shape) == 1:
          ret = np.empty((count,), dtype=np.float32)
          ret.fill(fill)
          ret[inds] = data
      else:
          ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
          ret.fill(fill)
          ret[inds, :] = data
      return ret

  def assign_anchor_fpn(self, gt_label, im_info, landmark=False, prefix='face', select_stride=0):

    #ta = datetime.datetime.now()
    im_info = im_info[0]
    gt_boxes = gt_label['gt_boxes']
    # clean up boxes
    nonneg = np.where(gt_boxes[:, 4] != -1)[0]
    gt_boxes = gt_boxes[nonneg]
    if config.USE_BLUR:
      gt_blur = gt_label['gt_blur']
      gt_blur = gt_blur[nonneg]
    if landmark:
      gt_landmarks = gt_label['gt_landmarks']
      gt_landmarks = gt_landmarks[nonneg]
      assert gt_boxes.shape[0]==gt_landmarks.shape[0]
    #scales = np.array(scales, dtype=np.float32)
    feat_strides = config.RPN_FEAT_STRIDE
    bbox_pred_len = 4
    landmark_pred_len = 10
    if config.USE_BLUR:
      gt_boxes[:,4] = gt_blur
      bbox_pred_len = 5
    if config.USE_OCCLUSION:
      landmark_pred_len = 15

    #anchors_list = self.anchors_list
    #inds_inside_list = self.inds_inside_list
    anchors = self.anchors
    inds_inside = self.inds_inside
    anchors_num_list = self.anchors_num_list
    feat_infos = self.feat_infos
    A_list = self.A_list

    total_anchors = sum(anchors_num_list)
    #print('total_anchors', anchors.shape[0], len(inds_inside), file=sys.stderr)

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)
    #print('BB', anchors.shape, len(inds_inside))
    #print('gt_boxes', gt_boxes.shape, file=sys.stderr)
    #tb = datetime.datetime.now()
    #self._times[0] += (tb-ta).total_seconds()
    #ta = datetime.datetime.now()

    if gt_boxes.size > 0:
        # overlap between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        #print('AAA', argmax_overlaps.shape)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        if config.TRAIN.RPN_FORCE_POSITIVE:
          labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
        labels[max_overlaps >= config.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if config.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    else:
        labels[:] = 0
    fg_inds = np.where(labels == 1)[0]
    #print('fg count', len(fg_inds))

    # subsample positive labels if we have too many
    if config.TRAIN.RPN_ENABLE_OHEM==0:
      fg_inds = np.where(labels == 1)[0]
      num_fg = int(config.TRAIN.RPN_FG_FRACTION * config.TRAIN.RPN_BATCH_SIZE)
      if len(fg_inds) > num_fg:
          disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
          if DEBUG:
              disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
          labels[disable_inds] = -1

      # subsample negative labels if we have too many
      num_bg = config.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
      bg_inds = np.where(labels == 0)[0]
      if len(bg_inds) > num_bg:
          disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
          if DEBUG:
              disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
          labels[disable_inds] = -1

      #fg_inds = np.where(labels == 1)[0]
      #num_fg = len(fg_inds)
      #num_bg = num_fg*int(1.0/config.TRAIN.RPN_FG_FRACTION-1)

      #bg_inds = np.where(labels == 0)[0]
      #if len(bg_inds) > num_bg:
      #    disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
      #    if DEBUG:
      #        disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
      #    labels[disable_inds] = -1
    else:
      fg_inds = np.where(labels == 1)[0]
      num_fg = len(fg_inds)
      bg_inds = np.where(labels == 0)[0]
      num_bg = len(bg_inds)

    #print('anchor stat', num_fg, num_bg)


    bbox_targets = np.zeros((len(inds_inside), bbox_pred_len), dtype=np.float32)
    if gt_boxes.size > 0:
        #print('GT', gt_boxes.shape, gt_boxes[argmax_overlaps, :4].shape)
        bbox_targets[:,:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :])
        #bbox_targets[:,4] = gt_blur
    #tb = datetime.datetime.now()
    #self._times[1] += (tb-ta).total_seconds()
    #ta = datetime.datetime.now()

    bbox_weights = np.zeros((len(inds_inside), bbox_pred_len), dtype=np.float32)
    #bbox_weights[labels == 1, :] = np.array(config.TRAIN.RPN_BBOX_WEIGHTS)
    bbox_weights[labels == 1, 0:4] = 1.0
    if bbox_pred_len>4:
      bbox_weights[labels == 1, 4:bbox_pred_len] = 0.1

    if landmark:
      landmark_targets = np.zeros((len(inds_inside), landmark_pred_len), dtype=np.float32)
      #landmark_weights = np.zeros((len(inds_inside), 10), dtype=np.float32)
      landmark_weights = np.zeros((len(inds_inside), landmark_pred_len), dtype=np.float32)
      #landmark_weights[labels == 1, :] = np.array(config.TRAIN.RPN_LANDMARK_WEIGHTS)
      if landmark_pred_len==10:
        landmark_weights[labels == 1, :] = 1.0
      elif landmark_pred_len==15:
        v = [1.0, 1.0, 0.1] * 5
        assert len(v)==15
        landmark_weights[labels == 1, :] = np.array(v)
      else:
        assert False
      #TODO here
      if gt_landmarks.size > 0:
        #print('AAA',argmax_overlaps)
        a_landmarks = gt_landmarks[argmax_overlaps,:,:]
        landmark_targets[:] = landmark_transform(anchors, a_landmarks)
        invalid = np.where(a_landmarks[:,0,2]<0.0)[0]
        #assert len(invalid)==0
        #landmark_weights[invalid, :] = np.array(config.TRAIN.RPN_INVALID_LANDMARK_WEIGHTS)
        landmark_weights[invalid, :] = 0.0
    #tb = datetime.datetime.now()
    #self._times[2] += (tb-ta).total_seconds()
    #ta = datetime.datetime.now()

    #if DEBUG:
    #    _sums = bbox_targets[labels == 1, :].sum(axis=0)
    #    _squared_sums = (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
    #    _counts = np.sum(labels == 1)
    #    means = _sums / (_counts + 1e-14)
    #    stds = np.sqrt(_squared_sums / _counts - means ** 2)
    #    print 'means', means
    #    print 'stdevs', stds
    # map up to original set of anchors
    #print(labels.shape, total_anchors, inds_inside.shape, inds_inside[0], inds_inside[-1])
    labels = AA._unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = AA._unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_weights = AA._unmap(bbox_weights, total_anchors, inds_inside, fill=0)
    if landmark:
      landmark_targets = AA._unmap(landmark_targets, total_anchors, inds_inside, fill=0)
      landmark_weights = AA._unmap(landmark_weights, total_anchors, inds_inside, fill=0)
    #print('CC', anchors.shape, len(inds_inside))

    #if DEBUG:
    #    if gt_boxes.size > 0:
    #        print 'rpn: max max_overlaps', np.max(max_overlaps)
    #    print 'rpn: num_positives', np.sum(labels == 1)
    #    print 'rpn: num_negatives', np.sum(labels == 0)
    #    _fg_sum = np.sum(labels == 1)
    #    _bg_sum = np.sum(labels == 0)
    #    _count = 1
    #    print 'rpn: num_positive avg', _fg_sum / _count
    #    print 'rpn: num_negative avg', _bg_sum / _count

    # resahpe
    label_list = list()
    bbox_target_list = list()
    bbox_weight_list = list()
    if landmark:
      landmark_target_list = list()
      landmark_weight_list = list()
    anchors_num_range = [0] + anchors_num_list
    label = {}
    for i in range(len(feat_strides)):
        stride = feat_strides[i]
        feat_height, feat_width = feat_infos[i]
        A = A_list[i]
        _label = labels[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
        if select_stride>0 and stride!=select_stride:
          #print('set', stride, select_stride)
          _label[:] = -1
        #print('_label', _label.shape, select_stride)
        #_fg_inds = np.where(_label == 1)[0]
        #n_fg = len(_fg_inds)
        #STAT[0]+=1
        #STAT[stride]+=n_fg
        #if STAT[0]%100==0:
        #  print('rpn_stat', STAT, file=sys.stderr)
        bbox_target = bbox_targets[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
        bbox_weight = bbox_weights[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
        if landmark:
          landmark_target = landmark_targets[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]
          landmark_weight = landmark_weights[sum(anchors_num_range[:i+1]):sum(anchors_num_range[:i+1])+anchors_num_range[i+1]]

        _label = _label.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
        _label = _label.reshape((1, A * feat_height * feat_width))
        bbox_target = bbox_target.reshape((1, feat_height*feat_width, A * bbox_pred_len)).transpose(0, 2, 1)
        bbox_weight = bbox_weight.reshape((1, feat_height*feat_width, A * bbox_pred_len)).transpose((0, 2, 1))
        label['%s_label_stride%d'%(prefix, stride)] = _label
        label['%s_bbox_target_stride%d'%(prefix,stride)] = bbox_target
        label['%s_bbox_weight_stride%d'%(prefix,stride)] = bbox_weight
        if landmark:
          landmark_target = landmark_target.reshape((1, feat_height*feat_width, A * landmark_pred_len)).transpose(0, 2, 1)
          landmark_weight = landmark_weight.reshape((1, feat_height*feat_width, A * landmark_pred_len)).transpose((0, 2, 1))
          label['%s_landmark_target_stride%d'%(prefix,stride)] = landmark_target
          label['%s_landmark_weight_stride%d'%(prefix,stride)] = landmark_weight
        #print('in_rpn', stride,_label.shape, bbox_target.shape, bbox_weight.shape, file=sys.stderr)
        label_list.append(_label)
        #print('DD', _label.shape)
        bbox_target_list.append(bbox_target)
        bbox_weight_list.append(bbox_weight)
        if landmark:
          landmark_target_list.append(landmark_target)
          landmark_weight_list.append(landmark_weight)

    label_concat = np.concatenate(label_list, axis=1)
    bbox_target_concat = np.concatenate(bbox_target_list, axis=2)
    bbox_weight_concat = np.concatenate(bbox_weight_list, axis=2)
    #fg_inds = np.where(label_concat[0] == 1)[0]
    #print('fg_inds_in_rpn2', fg_inds, file=sys.stderr)

    label.update({'%s_label'%prefix: label_concat,
            '%s_bbox_target'%prefix: bbox_target_concat,
            '%s_bbox_weight'%prefix: bbox_weight_concat}
            )
    if landmark:
      landmark_target_concat = np.concatenate(landmark_target_list, axis=2)
      landmark_weight_concat = np.concatenate(landmark_weight_list, axis=2)
      label['%s_landmark_target'%prefix] = landmark_target_concat
      label['%s_landmark_weight'%prefix] = landmark_weight_concat
    #tb = datetime.datetime.now()
    #self._times[3] += (tb-ta).total_seconds()
    #ta = datetime.datetime.now()
    #print(self._times)
    return label
