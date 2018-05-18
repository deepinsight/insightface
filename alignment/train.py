from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import hg
import hg2 as hg
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
    #label = preds[1].asnumpy()
    #out0 = preds[1].asnumpy()[0][0]
    #l = 10
    #out0 = out0[20:20+l, 20:20+l]
    #out1 = preds[2].asnumpy()[0][0]
    #out1 = out1[20:20+l, 20:20+l]

    #m = preds[3].asnumpy()[0]
    #theta = np.arcsin(m[3])
    #theta = theta/math.pi*180
    #print(out0)
    #print('')
    #print(out1)
    #print('')
    #print(m, theta)
    #print(label[0])
    #for i in xrange(gt_label.shape[0]):
    #    label0 = gt_label[i][0]
    #    c = np.count_nonzero(label0)
    #    ind = np.unravel_index(np.argmax(label0, axis=None), label0.shape)
    #    print('A', i, ind, label0.shape, c)



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
        #print('acc',label.shape, pred_label.shape)

        nme = []
        for b in xrange(pred_label.shape[0]):
          for p in xrange(pred_label.shape[1]):
            heatmap_gt = label[b][p]
            heatmap_pred = pred_label[b][p]
            heatmap_pred = cv2.resize(heatmap_pred, (label.shape[2], label.shape[3]))
            ind_gt = np.unravel_index(np.argmax(heatmap_gt, axis=None), heatmap_gt.shape)
            ind_pred = np.unravel_index(np.argmax(heatmap_pred, axis=None), heatmap_pred.shape)
            ind_gt = np.array(ind_gt)
            ind_pred = np.array(ind_pred)
            dist = np.sqrt(np.sum(np.square(ind_gt - ind_pred)))
            nme.append(dist)
        nme = np.mean(nme)
        nme /= np.sqrt(float(label.shape[2]*label.shape[3]))

    self.sum_metric += nme
    self.num_inst += 1.0

class NMEMetric2(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(NMEMetric2, self).__init__(
        'NME2', axis=self.axis,
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
            heatmap_pred = cv2.resize(heatmap_pred, (args.input_img_size, args.input_img_size))
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
          if args.norm_type=='2d':
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


  print('Call with', args)
  train_iter = FaceSegIter(path_imgrec = os.path.join(args.data_dir, 'train.rec'),
      batch_size = args.batch_size,
      per_batch_size = args.per_batch_size,
      aug_level = 1,
      use_coherent = args.use_coherent,
      args = args,
      )
  targets = ['ibug', 'cofw_testset', '300W', 'AFLW2000-3D']

  data_shape = train_iter.get_data_shape()
  #label_shape = train_iter.get_label_shape()
  sym = hg.get_symbol(num_classes=args.num_classes, binarize=args.binarize, label_size=args.output_label_size, input_size=args.input_img_size, use_coherent = args.use_coherent, use_dla = args.use_dla, use_N = args.use_N, use_DCN = args.use_DCN, per_batch_size = args.per_batch_size)
  if len(args.pretrained)==0:
      #data_shape_dict = {'data' : (args.per_batch_size,)+data_shape, 'softmax_label' : (args.per_batch_size,)+label_shape}
      data_shape_dict = train_iter.get_shape_dict()
      arg_params, aux_params = hg.init_weights(sym, data_shape_dict)
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
  lr = args.lr
  #_rescale_grad = 1.0
  _rescale_grad = 1.0/args.ctx_num
  #lr = args.lr
  #opt = optimizer.SGD(learning_rate=lr, momentum=0.9, wd=5.e-4, rescale_grad=_rescale_grad)
  #opt = optimizer.Adam(learning_rate=lr, wd=args.wd, rescale_grad=_rescale_grad)
  opt = optimizer.Nadam(learning_rate=lr, wd=args.wd, rescale_grad=_rescale_grad, clip_gradient=5.0)
  #opt = optimizer.RMSProp(learning_rate=lr, wd=args.wd, rescale_grad=_rescale_grad)
  initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
  _cb = mx.callback.Speedometer(args.batch_size, 10)
  _metric = LossValueMetric()
  #_metric2 = AccMetric()
  #eval_metrics = [_metric, _metric2]
  eval_metrics = [_metric]

  #lr_steps = [40000,60000,80000]
  #lr_steps = [12000,18000,22000]
  if len(args.lr_steps)==0:
    lr_steps = [16000,24000,30000]
    #lr_steps = [14000,24000,30000]
    #lr_steps = [5000,10000]
  else:
    lr_steps = [int(x) for x in args.lr_steps.split(',')]
  _a = 40//args.batch_size
  for i in xrange(len(lr_steps)):
    lr_steps[i] *= _a
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
    for target in targets:
        _file = os.path.join(args.data_dir, '%s.rec'%target)
        if not os.path.exists(_file):
            continue
        val_iter = FaceSegIter(path_imgrec = _file,
          batch_size = args.batch_size,
          #batch_size = 4,
          aug_level = 0,
          args = args,
          )
        _metric = NMEMetric2()
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
      if args.ckpt>0:
        msave = mbatch//args.verbose
        print('saving', msave)
        arg, aux = model.get_params()
        mx.model.save_checkpoint(args.prefix, msave, model.symbol, arg, aux)


  model.fit(train_iter,
      begin_epoch        = 0,
      num_epoch          = args.end_epoch,
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
  parser = argparse.ArgumentParser(description='Train face 3d')
  # general
  parser.add_argument('--data-dir', default='./data', help='')
  parser.add_argument('--prefix', default='./models/test',
      help='directory to save model.')
  parser.add_argument('--pretrained', default='',
      help='')
  parser.add_argument('--lr-steps', default='', type=str, help='')
  parser.add_argument('--verbose', type=int, default=2000, help='')
  parser.add_argument('--retrain', action='store_true', default=False,
      help='true means continue training.')
  parser.add_argument('--binarize', action='store_true', default=False, help='')
  parser.add_argument('--end-epoch', type=int, default=87,
      help='training epoch size.')
  parser.add_argument('--per-batch-size', type=int, default=20, help='')
  parser.add_argument('--num-classes', type=int, default=68, help='')
  parser.add_argument('--input-img-size', type=int, default=128, help='')
  parser.add_argument('--output-label-size', type=int, default=64, help='')
  parser.add_argument('--lr', type=float, default=2.5e-4, help='')
  parser.add_argument('--wd', type=float, default=5e-4, help='')
  parser.add_argument('--ckpt', type=int, default=1, help='')
  parser.add_argument('--norm-type', type=str, default='2d', help='')
  parser.add_argument('--use-coherent', type=int, default=1, help='')
  parser.add_argument('--use-dla', type=int, default=1, help='')
  parser.add_argument('--use-N', type=int, default=3, help='')
  parser.add_argument('--use-DCN', type=int, default=2, help='')
  args = parser.parse_args()
  main(args)

