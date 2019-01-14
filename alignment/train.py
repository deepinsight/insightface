from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse
from data import FaceSegIter
import mxnet as mx
import mxnet.optimizer as optimizer
import numpy as np
import os
import sys
import math
import random
import cv2
from config import config, default, generate_config
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbol'))
import sym_heatmap
#import sym_fc
#from symbol import fc
from optimizer import ONadam


args = None
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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

  def update(self, labels, preds):
    self.count+=1
    preds = [preds[-1]]
    for label, pred_label in zip(labels, preds):
        label = label.asnumpy()
        pred_label = pred_label.asnumpy()
        #print('label', np.count_nonzero(label[0][36]))
        #print('acc',label.shape, pred_label.shape)
        #print(label.ndim)

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
        #print('nme', nme)
        #nme = np.mean(nme)

    if len(nme)>0:
        self.sum_metric += np.mean(nme)
        self.num_inst += 1.0

def main(args):
  _seed = 727
  random.seed(_seed)
  np.random.seed(_seed)
  mx.random.seed(_seed)
  ctx = []
  cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
  if len(cvd)>0:
    for i in xrange(len(cvd.split(','))):
      ctx.append(mx.gpu(i))
  if len(ctx)==0:
    ctx = [mx.cpu()]
    print('use cpu')
  else:
    print('gpu num:', len(ctx))
  #ctx = [mx.gpu(0)]
  args.ctx_num = len(ctx)

  args.batch_size = args.per_batch_size*args.ctx_num
  config.per_batch_size = args.per_batch_size



  print('Call with', args, config)
  train_iter = FaceSegIter(path_imgrec = os.path.join(config.dataset_path, 'train.rec'),
      batch_size = args.batch_size,
      per_batch_size = args.per_batch_size,
      aug_level = 1,
      exf = args.exf,
      args = args,
      )

  data_shape = train_iter.get_data_shape()
  #label_shape = train_iter.get_label_shape()
  sym = sym_heatmap.get_symbol(num_classes=config.num_classes)
  if len(args.pretrained)==0:
      #data_shape_dict = {'data' : (args.per_batch_size,)+data_shape, 'softmax_label' : (args.per_batch_size,)+label_shape}
      data_shape_dict = train_iter.get_shape_dict()
      arg_params, aux_params = sym_heatmap.init_weights(sym, data_shape_dict)
  else:
      vec = args.pretrained.split(',')
      print('loading', vec)
      _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
      #sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)

  model = mx.mod.Module(
      context       = ctx,
      symbol        = sym,
      label_names   = train_iter.get_label_names(),
  )
  #lr = 1.0e-3
  #lr = 2.5e-4
  _rescale_grad = 1.0/args.ctx_num
  #_rescale_grad = 1.0/args.batch_size
  #lr = args.lr
  #opt = optimizer.Nadam(learning_rate=args.lr, wd=args.wd, rescale_grad=_rescale_grad, clip_gradient=5.0)
  if args.optimizer=='onadam':
    opt = ONadam(learning_rate=args.lr, wd=args.wd, rescale_grad=_rescale_grad, clip_gradient=5.0)
  elif args.optimizer=='nadam':
    opt = optimizer.Nadam(learning_rate=args.lr, rescale_grad=_rescale_grad)
  elif args.optimizer=='rmsprop':
    opt = optimizer.RMSProp(learning_rate=args.lr, rescale_grad=_rescale_grad)
  elif args.optimizer=='adam':
    opt = optimizer.Adam(learning_rate=args.lr, rescale_grad=_rescale_grad)
  else:
    opt = optimizer.SGD(learning_rate=args.lr, momentum=0.9, wd=args.wd, rescale_grad=_rescale_grad)
  initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
  _cb = mx.callback.Speedometer(args.batch_size, args.frequent)
  _metric = LossValueMetric()
  #_metric = NMEMetric()
  #_metric2 = AccMetric()
  #eval_metrics = [_metric, _metric2]
  eval_metrics = [_metric]
  lr_steps = [int(x) for x in args.lr_step.split(',')]
  print('lr-steps', lr_steps)
  global_step = [0]

  def val_test():
    all_layers = sym.get_internals()
    vsym = all_layers['heatmap_output']
    vmodel = mx.mod.Module(symbol=vsym, context=ctx, label_names = None)
    #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    vmodel.bind(data_shapes=[('data', (args.batch_size,)+data_shape)])
    arg_params, aux_params = model.get_params()
    vmodel.set_params(arg_params, aux_params)
    for target in config.val_targets:
        _file = os.path.join(config.dataset_path, '%s.rec'%target)
        if not os.path.exists(_file):
            continue
        val_iter = FaceSegIter(path_imgrec = _file,
          batch_size = args.batch_size,
          #batch_size = 4,
          aug_level = 0,
          args = args,
          )
        _metric = NMEMetric()
        val_metric = mx.metric.create(_metric)
        val_metric.reset()
        val_iter.reset()
        for i, eval_batch in enumerate(val_iter):
          #print(eval_batch.data[0].shape, eval_batch.label[0].shape)
          batch_data = mx.io.DataBatch(eval_batch.data)
          model.forward(batch_data, is_train=False)
          model.update_metric(val_metric, eval_batch.label)
        nme_value = val_metric.get_name_value()[0][1]
        print('[%d][%s]NME: %f'%(global_step[0], target, nme_value))
  
  def _batch_callback(param):
    _cb(param)
    global_step[0]+=1
    mbatch = global_step[0]
    for _lr in lr_steps:
      if mbatch==_lr:
        opt.lr *= 0.2
        print('lr change to', opt.lr)
        break
    if mbatch%1000==0:
      print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)
    if mbatch>0 and mbatch%args.verbose==0:
      val_test()
      if args.ckpt==1:
        msave = mbatch//args.verbose
        print('saving', msave)
        arg, aux = model.get_params()
        mx.model.save_checkpoint(args.prefix, msave, model.symbol, arg, aux)
    if mbatch==lr_steps[-1]:
      if args.ckpt==2:
        #msave = mbatch//args.verbose
        msave = 1
        print('saving', msave)
        arg, aux = model.get_params()
        mx.model.save_checkpoint(args.prefix, msave, model.symbol, arg, aux)
      sys.exit(0)

  train_iter = mx.io.PrefetchingIter(train_iter)

  model.fit(train_iter,
      begin_epoch        = 0,
      num_epoch          = 9999,
      #eval_data          = val_iter,
      eval_data          = None,
      eval_metric        = eval_metrics,
      kvstore            = 'device',
      optimizer          = opt,
      initializer        = initializer,
      arg_params         = arg_params,
      aux_params         = aux_params,
      allow_missing      = True,
      batch_end_callback = _batch_callback,
      epoch_end_callback = None,
      )

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train face alignment')
  # general
  parser.add_argument('--network', help='network name', default=default.network, type=str)
  parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
  args, rest = parser.parse_known_args()
  generate_config(args.network, args.dataset)
  parser.add_argument('--prefix', default=default.prefix, help='directory to save model.')
  parser.add_argument('--pretrained', default=default.pretrained, help='')
  parser.add_argument('--optimizer', default='nadam', help='')
  parser.add_argument('--lr', type=float, default=default.lr, help='')
  parser.add_argument('--wd', type=float, default=default.wd, help='')
  parser.add_argument('--per-batch-size', type=int, default=default.per_batch_size, help='')
  parser.add_argument('--lr-step', help='learning rate steps (in epoch)', default=default.lr_step, type=str)
  parser.add_argument('--ckpt', type=int, default=1, help='')
  parser.add_argument('--norm', type=int, default=0, help='')
  parser.add_argument('--exf', type=int, default=1, help='')
  parser.add_argument('--frequent', type=int, default=default.frequent, help='')
  parser.add_argument('--verbose', type=int, default=default.verbose, help='')
  args = parser.parse_args()
  main(args)

