import mxnet as mx
import numpy as np
import math
import cv2
from config import config

class LossValueMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(LossValueMetric, self).__init__(
        'lossvalue', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []

  def update(self, labels, preds):
    loss = preds[0].asnumpy()[0]
    self.sum_metric += loss
    self.num_inst += 1.0

class NMEMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(NMEMetric, self).__init__(
        'NME', axis=self.axis,
        output_names=None, label_names=None)
    #self.losses = []
    self.count = 0

  def cal_nme(self, label, pred_label):
    nme = []
    for b in xrange(pred_label.shape[0]):
      record = [None]*6
      item = []
      if label.ndim==4:
          _heatmap = label[b][36]
          if np.count_nonzero(_heatmap)==0:
              continue
      else:#ndim==3
          #print(label[b])
          if np.count_nonzero(label[b])==0:
              continue
      for p in xrange(pred_label.shape[1]):
        if label.ndim==4:
            heatmap_gt = label[b][p]
            ind_gt = np.unravel_index(np.argmax(heatmap_gt, axis=None), heatmap_gt.shape)
            ind_gt = np.array(ind_gt)
        else:
            ind_gt = label[b][p]
            #ind_gt = ind_gt.astype(np.int)
            #print(ind_gt)
        heatmap_pred = pred_label[b][p]
        heatmap_pred = cv2.resize(heatmap_pred, (config.input_img_size, config.input_img_size))
        ind_pred = np.unravel_index(np.argmax(heatmap_pred, axis=None), heatmap_pred.shape)
        ind_pred = np.array(ind_pred)
        #print(ind_gt.shape)
        #print(ind_pred)
        if p==36:
            #print('b', b, p, ind_gt, np.count_nonzero(heatmap_gt))
            record[0] = ind_gt
        elif p==39:
            record[1] = ind_gt
        elif p==42:
            record[2] = ind_gt
        elif p==45:
            record[3] = ind_gt
        if record[4] is None or record[5] is None:
            record[4] = ind_gt
            record[5] = ind_gt
        else:
            record[4] = np.minimum(record[4], ind_gt)
            record[5] = np.maximum(record[5], ind_gt)
        #print(ind_gt.shape, ind_pred.shape)
        value = np.sqrt(np.sum(np.square(ind_gt - ind_pred)))
        item.append(value)
      _nme = np.mean(item)
      if config.landmark_type=='2d':
          left_eye = (record[0]+record[1])/2
          right_eye = (record[2]+record[3])/2
          _dist = np.sqrt(np.sum(np.square(left_eye - right_eye)))
          #print('eye dist', _dist, left_eye, right_eye)
          _nme /= _dist
      else:
          #_dist = np.sqrt(float(label.shape[2]*label.shape[3]))
          _dist = np.sqrt(np.sum(np.square(record[5] - record[4])))
          #print(_dist)
          _nme /= _dist
      nme.append(_nme)
    return np.mean(nme)

  def update(self, labels, preds):
    self.count+=1
    label = labels[0].asnumpy()
    pred_label = preds[-1].asnumpy()
    nme = self.cal_nme(label, pred_label)

    #print('nme', nme)
    #nme = np.mean(nme)
    self.sum_metric += np.mean(nme)
    self.num_inst += 1.0
