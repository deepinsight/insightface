from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import logging
import pickle
import numpy as np
import sklearn
from data import FaceImageIter
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
#import face_image
import fresnet
import fmobilenet 


logger = logging.getLogger()
logger.setLevel(logging.INFO)

AGE=100


args = None


class AccMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(AccMetric, self).__init__(
        'acc', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []
    self.count = 0

  def update(self, labels, preds):
    self.count+=1
    label = labels[0].asnumpy()[:,0:1]
    pred_label = preds[-1].asnumpy()[:,0:2]
    pred_label = np.argmax(pred_label, axis=self.axis)
    pred_label = pred_label.astype('int32').flatten()
    label = label.astype('int32').flatten()
    assert label.shape==pred_label.shape
    self.sum_metric += (pred_label.flat == label.flat).sum()
    self.num_inst += len(pred_label.flat)

class LossValueMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(LossValueMetric, self).__init__(
        'lossvalue', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []

  def update(self, labels, preds):
    loss = preds[-1].asnumpy()[0]
    self.sum_metric += loss
    self.num_inst += 1.0
    gt_label = preds[-2].asnumpy()
    #print(gt_label)

class MAEMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(MAEMetric, self).__init__(
        'MAE', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []
    self.count = 0

  def update(self, labels, preds):
    self.count+=1
    label = labels[0].asnumpy()
    label_age = np.count_nonzero(label[:,1:], axis=1)
    pred_age = np.zeros( label_age.shape, dtype=np.int)
    #pred_age = np.zeros( label_age.shape, dtype=np.float32)
    pred = preds[-1].asnumpy()
    for i in xrange(AGE):
      _pred = pred[:,2+i*2:4+i*2]
      _pred = np.argmax(_pred, axis=1)
      #pred = pred[:,1]
      pred_age += _pred
    #pred_age = pred_age.astype(np.int)
    mae = np.mean(np.abs(label_age - pred_age))
    self.sum_metric += mae
    self.num_inst += 1.0

class CUMMetric(mx.metric.EvalMetric):
  def __init__(self, n=5):
    self.axis = 1
    self.n = n
    super(CUMMetric, self).__init__(
        'CUM_%d'%n, axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []
    self.count = 0

  def update(self, labels, preds):
    self.count+=1
    label = labels[0].asnumpy()
    label_age = np.count_nonzero(label[:,1:], axis=1)
    pred_age = np.zeros( label_age.shape, dtype=np.int)
    pred = preds[-1].asnumpy()
    for i in xrange(AGE):
      _pred = pred[:,2+i*2:4+i*2]
      _pred = np.argmax(_pred, axis=1)
      #pred = pred[:,1]
      pred_age += _pred
    diff = np.abs(label_age - pred_age)
    cum = np.sum( (diff<self.n) )
    self.sum_metric += cum
    self.num_inst += len(label_age)

def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--data-dir', default='', help='training set directory')
  parser.add_argument('--prefix', default='../model/model', help='directory to save model.')
  parser.add_argument('--pretrained', default='', help='pretrained model to load')
  parser.add_argument('--ckpt', type=int, default=1, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
  parser.add_argument('--loss-type', type=int, default=4, help='loss type')
  parser.add_argument('--verbose', type=int, default=2000, help='do verification testing and model saving every verbose batches')
  parser.add_argument('--max-steps', type=int, default=0, help='max training batches')
  parser.add_argument('--end-epoch', type=int, default=100000, help='training epoch size.')
  parser.add_argument('--network', default='r50', help='specify network')
  parser.add_argument('--image-size', default='112,112', help='specify input image height and width')
  parser.add_argument('--version-input', type=int, default=1, help='network input config')
  parser.add_argument('--version-output', type=str, default='GAP', help='network embedding output config')
  parser.add_argument('--version-act', type=str, default='prelu', help='network activation config')
  parser.add_argument('--multiplier', type=float, default=1.0, help='')
  parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
  parser.add_argument('--lr-steps', type=str, default='', help='steps of lr changing')
  parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
  parser.add_argument('--bn-mom', type=float, default=0.9, help='bn mom')
  parser.add_argument('--mom', type=float, default=0.9, help='momentum')
  parser.add_argument('--per-batch-size', type=int, default=128, help='batch size in each context')
  parser.add_argument('--rand-mirror', type=int, default=1, help='if do random mirror in training')
  parser.add_argument('--cutoff', type=int, default=0, help='cut off aug')
  parser.add_argument('--color', type=int, default=0, help='color jittering aug')
  parser.add_argument('--ce-loss', default=False, action='store_true', help='if output ce loss')
  args = parser.parse_args()
  return args


def get_symbol(args, arg_params, aux_params):
  data_shape = (args.image_channel,args.image_h,args.image_w)
  image_shape = ",".join([str(x) for x in data_shape])
  margin_symbols = []
  if args.network[0]=='m':
    fc1 = fmobilenet.get_symbol(AGE*2+2, 
        multiplier = args.multiplier,
        version_input=args.version_input, 
        version_output=args.version_output)
  else:
    fc1 = fresnet.get_symbol(AGE*2+2, args.num_layers,
        version_input=args.version_input, 
        version_output=args.version_output)
  label = mx.symbol.Variable('softmax_label')
  gender_label = mx.symbol.slice_axis(data = label, axis=1, begin=0, end=1)
  gender_label = mx.symbol.reshape(gender_label, shape=(args.per_batch_size,))
  gender_fc1 = mx.symbol.slice_axis(data = fc1, axis=1, begin=0, end=2)
  #gender_fc7 = mx.sym.FullyConnected(data=gender_fc1, num_hidden=2, name='gender_fc7')
  gender_softmax = mx.symbol.SoftmaxOutput(data=gender_fc1, label = gender_label, name='gender_softmax', normalization='valid', use_ignore=True, ignore_label = 9999)
  outs = [gender_softmax]
  for i in range(AGE):
    age_label = mx.symbol.slice_axis(data = label, axis=1, begin=i+1, end=i+2)
    age_label = mx.symbol.reshape(age_label, shape=(args.per_batch_size,))
    age_fc1 = mx.symbol.slice_axis(data = fc1, axis=1, begin=2+i*2, end=4+i*2)
    #age_fc7 = mx.sym.FullyConnected(data=age_fc1, num_hidden=2, name='age_fc7_%i'%i)
    age_softmax = mx.symbol.SoftmaxOutput(data=age_fc1, label = age_label, name='age_softmax_%d'%i, normalization='valid', grad_scale=1)
    outs.append(age_softmax)
  outs.append(mx.sym.BlockGrad(fc1))

  out = mx.symbol.Group(outs)
  return (out, arg_params, aux_params)

def train_net(args):
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
    prefix = args.prefix
    prefix_dir = os.path.dirname(prefix)
    if not os.path.exists(prefix_dir):
      os.makedirs(prefix_dir)
    end_epoch = args.end_epoch
    args.ctx_num = len(ctx)
    args.num_layers = int(args.network[1:])
    print('num_layers', args.num_layers)
    if args.per_batch_size==0:
      args.per_batch_size = 128
    args.batch_size = args.per_batch_size*args.ctx_num
    args.rescale_threshold = 0
    args.image_channel = 3

    data_dir_list = args.data_dir.split(',')
    assert len(data_dir_list)==1
    data_dir = data_dir_list[0]
    path_imgrec = None
    path_imglist = None
    image_size = [int(x) for x in args.image_size.split(',')]
    assert len(image_size)==2
    assert image_size[0]==image_size[1]
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    print('image_size', image_size)
    path_imgrec = os.path.join(data_dir, "train.rec")
    path_imgrec_val = os.path.join(data_dir, "val.rec")


    print('Called with argument:', args)
    data_shape = (args.image_channel,image_size[0],image_size[1])
    mean = None

    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd
    base_mom = args.mom
    if len(args.pretrained)==0:
      arg_params = None
      aux_params = None
      sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
    else:
      vec = args.pretrained.split(',')
      print('loading', vec)
      _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
      sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)

    #label_name = 'softmax_label'
    #label_shape = (args.batch_size,)
    model = mx.mod.Module(
        context       = ctx,
        symbol        = sym,
    )
    val_dataiter = None

    train_dataiter = FaceImageIter(
        batch_size           = args.batch_size,
        data_shape           = data_shape,
        path_imgrec          = path_imgrec,
        shuffle              = True,
        rand_mirror          = args.rand_mirror,
        mean                 = mean,
        cutoff               = args.cutoff,
        color_jittering      = args.color,
    )
    val_dataiter = FaceImageIter(
        batch_size           = args.batch_size,
        data_shape           = data_shape,
        path_imgrec          = path_imgrec_val,
        shuffle              = False,
        rand_mirror          = False,
        mean                 = mean,
    )

    metric = mx.metric.CompositeEvalMetric([AccMetric(), MAEMetric(), CUMMetric()])

    if args.network[0]=='r' or args.network[0]=='y':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    elif args.network[0]=='i' or args.network[0]=='x':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2) #inception
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    _rescale = 1.0/args.ctx_num
    opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    #opt = optimizer.Nadam(learning_rate=base_lr, wd=base_wd, rescale_grad=_rescale)
    som = 20
    _cb = mx.callback.Speedometer(args.batch_size, som)
    lr_steps = [int(x) for x in args.lr_steps.split(',')]

    global_step = [0]
    def _batch_callback(param):
      _cb(param)
      global_step[0]+=1
      mbatch = global_step[0]
      for _lr in lr_steps:
        if mbatch==_lr:
          opt.lr *= 0.1
          print('lr change to', opt.lr)
          break
      if mbatch%1000==0:
        print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)
      if mbatch==lr_steps[-1]:
        arg, aux = model.get_params()
        all_layers = model.symbol.get_internals()
        _sym = all_layers['fc1_output']
        mx.model.save_checkpoint(args.prefix, 0, _sym, arg, aux)
        sys.exit(0)

    epoch_cb = None
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)
    print('start fitting')

    model.fit(train_dataiter,
        begin_epoch        = begin_epoch,
        num_epoch          = end_epoch,
        eval_data          = val_dataiter,
        eval_metric        = metric,
        kvstore            = 'device',
        optimizer          = opt,
        #optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        allow_missing      = True,
        batch_end_callback = _batch_callback,
        epoch_end_callback = epoch_cb )

def main():
    #time.sleep(3600*6.5)
    global args
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

