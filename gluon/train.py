from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import logging
import time
import pickle
import numpy as np
import sklearn
from image_iter import FaceImageIter
from age_iter import FaceImageIter as FaceImageIterAge
#from image_iter import FaceImageIterList
import mxnet as mx
from mxnet import gluon
from mxnet import profiler
from mxnet.gluon import nn
from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet.test_utils import get_mnist_iterator
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
import argparse
import mxnet.optimizer as optimizer
#sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'eval'))
import verification
#sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'common'))
import face_image
#sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'blocks'))
import fresnet
from UDD import *
#import finception_resnet_v2
#import fmobilenet 
#import fmobilenetv2
#import fmobilefacenet
#import fxception
#import fdensenet
#import fdpn
#import fnasnet
#import spherenet
#sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))
#import center_loss


logger = logging.getLogger()
logger.setLevel(logging.INFO)

AGE = 100

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
    #preds = [preds[1]] #use softmax output
    for label, pred_label in zip(labels, preds):
        if pred_label.shape != label.shape:
            pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.asnumpy().astype('int32').flatten()
        label = label.asnumpy()
        if label.ndim==2:
          label = label[:,0]
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
    label_age = np.count_nonzero(label, axis=1)
    pred_age = np.zeros( label_age.shape, dtype=np.int)
    #pred_age = np.zeros( label_age.shape, dtype=np.float32)
    pred = preds[0].asnumpy()
    for i in xrange(AGE):
      _pred = pred[:,i*2:(i*2+2)]
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
    label_age = np.count_nonzero(label, axis=1)
    pred_age = np.zeros( label_age.shape, dtype=np.int)
    pred = preds[0].asnumpy()
    for i in xrange(AGE):
      _pred = pred[:,i*2:(i*2+2)]
      _pred = np.argmax(_pred, axis=1)
      #pred = pred[:,1]
      pred_age += _pred
    diff = np.abs(label_age - pred_age)
    cum = np.sum( (diff<self.n) )
    self.sum_metric += cum
    self.num_inst += len(label_age)

def parse_args():
  global args
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--data-dir', default='', help='training set directory')
  parser.add_argument('--gender-data-dir', default='', help='training set directory')
  parser.add_argument('--age-data-dir', default='', help='training set directory')
  parser.add_argument('--prefix', default='../model/model', help='directory to save model.')
  parser.add_argument('--pretrained', default='', help='pretrained model to load')
  parser.add_argument('--ckpt', type=int, default=1, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
  parser.add_argument('--loss-type', type=int, default=4, help='loss type')
  parser.add_argument('--verbose', type=int, default=2000, help='do verification testing and model saving every verbose batches')
  parser.add_argument('--max-steps', type=int, default=0, help='max training batches')
  parser.add_argument('--end-epoch', type=int, default=100000, help='training epoch size.')
  parser.add_argument('--network', default='r50', help='specify network')
  parser.add_argument('--version-output', type=str, default='E', help='network embedding output config')
  parser.add_argument('--version-unit', type=int, default=1, help='resnet unit config')
  parser.add_argument('--version-act', type=str, default='relu', help='network activation config')
  parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
  parser.add_argument('--lr-steps', type=str, default='', help='steps of lr changing')
  parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
  parser.add_argument('--fc7-wd-mult', type=float, default=1.0, help='weight decay mult for fc7')
  parser.add_argument('--bn-mom', type=float, default=0.9, help='bn mom')
  parser.add_argument('--mom', type=float, default=0.9, help='momentum')
  parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
  parser.add_argument('--per-batch-size', type=int, default=128, help='batch size in each context')
  parser.add_argument('--margin-m', type=float, default=0.5, help='margin for loss')
  parser.add_argument('--margin-s', type=float, default=64.0, help='scale for feature')
  parser.add_argument('--margin-a', type=float, default=1.0, help='')
  parser.add_argument('--margin-b', type=float, default=0.0, help='')
  parser.add_argument('--rand-mirror', type=int, default=1, help='if do random mirror in training')
  parser.add_argument('--cutoff', type=int, default=0, help='cut off aug')
  parser.add_argument('--eval', type=str, default='lfw,cfp_fp,agedb_30', help='verification targets')
  parser.add_argument('--task', type=str, default='', help='')
  parser.add_argument('--mode', type=str, default='gluon', help='')
  args = parser.parse_args()
  return args

def get_model():
  #print('init resnet', args.num_layers)
  if args.task=='':
    if args.margin_a>0.0:
      return ArcMarginBlock(args, prefix='')
    else:
      return DenseBlock(args, prefix='')
  else:#AGE or GENDER
    return GABlock(args, prefix='')


#def get_symbol(args, arg_params, aux_params):
#  data_shape = (args.image_channel,args.image_h,args.image_w)
#  image_shape = ",".join([str(x) for x in data_shape])
#  margin_symbols = []
#  if args.network[0]=='d':
#    embedding = fdensenet.get_symbol(args.emb_size, args.num_layers,
#        version_se=args.version_se, version_input=args.version_input, 
#        version_output=args.version_output, version_unit=args.version_unit)
#  elif args.network[0]=='m':
#    print('init mobilenet', args.num_layers)
#    if args.num_layers==1:
#      embedding = fmobilenet.get_symbol(args.emb_size, 
#          version_se=args.version_se, version_input=args.version_input, 
#          version_output=args.version_output, version_unit=args.version_unit)
#    else:
#      embedding = fmobilenetv2.get_symbol(args.emb_size)
#  elif args.network[0]=='i':
#    print('init inception-resnet-v2', args.num_layers)
#    embedding = finception_resnet_v2.get_symbol(args.emb_size,
#        version_se=args.version_se, version_input=args.version_input, 
#        version_output=args.version_output, version_unit=args.version_unit)
#  elif args.network[0]=='x':
#    print('init xception', args.num_layers)
#    embedding = fxception.get_symbol(args.emb_size,
#        version_se=args.version_se, version_input=args.version_input, 
#        version_output=args.version_output, version_unit=args.version_unit)
#  elif args.network[0]=='p':
#    print('init dpn', args.num_layers)
#    embedding = fdpn.get_symbol(args.emb_size, args.num_layers,
#        version_se=args.version_se, version_input=args.version_input, 
#        version_output=args.version_output, version_unit=args.version_unit)
#  elif args.network[0]=='n':
#    print('init nasnet', args.num_layers)
#    embedding = fnasnet.get_symbol(args.emb_size)
#  elif args.network[0]=='s':
#    print('init spherenet', args.num_layers)
#    embedding = spherenet.get_symbol(args.emb_size, args.num_layers)
#  elif args.network[0]=='y':
#    print('init mobilefacenet', args.num_layers)
#    embedding = fmobilefacenet.get_symbol(args.emb_size, bn_mom = args.bn_mom, wd_mult = args.fc7_wd_mult)
#  else:
#    print('init resnet', args.num_layers)
#    embedding = fresnet.get_symbol(args.emb_size, args.num_layers, 
#        version_se=args.version_se, version_input=args.version_input, 
#        version_output=args.version_output, version_unit=args.version_unit,
#        version_act=args.version_act)
#  all_label = mx.symbol.Variable('softmax_label')
#  gt_label = all_label
#  extra_loss = None
#  _weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, args.emb_size), lr_mult=1.0, wd_mult=args.fc7_wd_mult)
#  if args.loss_type==0: #softmax
#    _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
#    fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=args.num_classes, name='fc7')
#  elif args.loss_type==1: #sphere
#    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
#    fc7 = mx.sym.LSoftmax(data=embedding, label=gt_label, num_hidden=args.num_classes,
#                          weight = _weight,
#                          beta=args.beta, margin=args.margin, scale=args.scale,
#                          beta_min=args.beta_min, verbose=1000, name='fc7')
#  elif args.loss_type==2:
#    s = args.margin_s
#    m = args.margin_m
#    assert(s>0.0)
#    assert(m>0.0)
#    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
#    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s
#    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.num_classes, name='fc7')
#    s_m = s*m
#    gt_one_hot = mx.sym.one_hot(gt_label, depth = args.num_classes, on_value = s_m, off_value = 0.0)
#    fc7 = fc7-gt_one_hot
#  elif args.loss_type==4:
#    s = args.margin_s
#    m = args.margin_m
#    assert s>0.0
#    assert m>=0.0
#    assert m<(math.pi/2)
#    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
#    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s
#    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.num_classes, name='fc7')
#    zy = mx.sym.pick(fc7, gt_label, axis=1)
#    cos_t = zy/s
#    cos_m = math.cos(m)
#    sin_m = math.sin(m)
#    mm = math.sin(math.pi-m)*m
#    #threshold = 0.0
#    threshold = math.cos(math.pi-m)
#    if args.easy_margin:
#      cond = mx.symbol.Activation(data=cos_t, act_type='relu')
#    else:
#      cond_v = cos_t - threshold
#      cond = mx.symbol.Activation(data=cond_v, act_type='relu')
#    body = cos_t*cos_t
#    body = 1.0-body
#    sin_t = mx.sym.sqrt(body)
#    new_zy = cos_t*cos_m
#    b = sin_t*sin_m
#    new_zy = new_zy - b
#    new_zy = new_zy*s
#    if args.easy_margin:
#      zy_keep = zy
#    else:
#      zy_keep = zy - s*mm
#    new_zy = mx.sym.where(cond, new_zy, zy_keep)
#
#    diff = new_zy - zy
#    diff = mx.sym.expand_dims(diff, 1)
#    gt_one_hot = mx.sym.one_hot(gt_label, depth = args.num_classes, on_value = 1.0, off_value = 0.0)
#    body = mx.sym.broadcast_mul(gt_one_hot, diff)
#    fc7 = fc7+body
#  elif args.loss_type==5:
#    s = args.margin_s
#    m = args.margin_m
#    assert s>0.0
#    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
#    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s
#    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.num_classes, name='fc7')
#    if args.margin_a!=1.0 or args.margin_m!=0.0 or args.margin_b!=0.0:
#      if args.margin_a==1.0 and args.margin_m==0.0:
#        s_m = s*args.margin_b
#        gt_one_hot = mx.sym.one_hot(gt_label, depth = args.num_classes, on_value = s_m, off_value = 0.0)
#        fc7 = fc7-gt_one_hot
#      else:
#        zy = mx.sym.pick(fc7, gt_label, axis=1)
#        cos_t = zy/s
#        t = mx.sym.arccos(cos_t)
#        if args.margin_a!=1.0:
#          t = t*args.margin_a
#        if args.margin_m>0.0:
#          t = t+args.margin_m
#        body = mx.sym.cos(t)
#        if args.margin_b>0.0:
#          body = body - args.margin_b
#        new_zy = body*s
#        diff = new_zy - zy
#        diff = mx.sym.expand_dims(diff, 1)
#        gt_one_hot = mx.sym.one_hot(gt_label, depth = args.num_classes, on_value = 1.0, off_value = 0.0)
#        body = mx.sym.broadcast_mul(gt_one_hot, diff)
#        fc7 = fc7+body
#  out_list = [mx.symbol.BlockGrad(embedding)]
#  softmax = mx.symbol.SoftmaxOutput(data=fc7, label = gt_label, name='softmax', normalization='valid')
#  out_list.append(softmax)
#  out = mx.symbol.Group(out_list)
#  return (out, arg_params, aux_params)
#
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
    args.image_channel = 3

    data_dir = args.data_dir
    if args.task=='gender':
      data_dir = args.gender_data_dir
    elif args.task=='age':
      data_dir = args.age_data_dir
    print('data dir', data_dir)
    path_imgrec = None
    path_imglist = None
    prop = face_image.load_property(data_dir)
    args.num_classes = prop.num_classes
    image_size = prop.image_size
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    print('image_size', image_size)
    assert(args.num_classes>0)
    print('num_classes', args.num_classes)
    path_imgrec = os.path.join(data_dir, "train.rec")


    print('Called with argument:', args)
    data_shape = (args.image_channel,image_size[0],image_size[1])
    mean = None

    begin_epoch = 0
    net = get_model()
    #if args.task=='':
    #  test_net = get_model_test(net)
    #print(net.__class__)
    #net = net0[0]
    if args.network[0]=='r' or args.network[0]=='y':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    elif args.network[0]=='i' or args.network[0]=='x':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2) #inception
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    net.hybridize()
    if args.mode=='gluon':
      if len(args.pretrained)==0:
        pass
      else:
        net.load_params(args.pretrained, allow_missing=True, ignore_extra = True)
      net.initialize(initializer)
      net.collect_params().reset_ctx(ctx)

    val_iter = None
    if args.task=='':
      train_iter = FaceImageIter(
          batch_size           = args.batch_size,
          data_shape           = data_shape,
          path_imgrec          = path_imgrec,
          shuffle              = True,
          rand_mirror          = args.rand_mirror,
          mean                 = mean,
          cutoff               = args.cutoff,
      )
    else:
      train_iter = FaceImageIterAge(
          batch_size           = args.batch_size,
          data_shape           = data_shape,
          path_imgrec          = path_imgrec,
          task                 = args.task,
          shuffle              = True,
          rand_mirror          = args.rand_mirror,
          mean                 = mean,
          cutoff               = args.cutoff,
      )

    if args.task=='age':
      metric = CompositeEvalMetric([MAEMetric(), CUMMetric()])
    elif args.task=='gender':
      metric = CompositeEvalMetric([AccMetric()])
    else:
      metric = CompositeEvalMetric([AccMetric()])

    ver_list = []
    ver_name_list = []
    if args.task=='':
      for name in args.eval.split(','):
        path = os.path.join(data_dir,name+".bin")
        if os.path.exists(path):
          data_set = verification.load_bin(path, image_size)
          ver_list.append(data_set)
          ver_name_list.append(name)
          print('ver', name)

    def ver_test(nbatch):
      results = []
      for i in xrange(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], net, ctx, batch_size = args.batch_size)
        print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
        #print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
        print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
        results.append(acc2)
      return results

    def val_test(nbatch=0):
      acc = 0.0
      #if args.task=='age':
      if len(args.age_data_dir)>0:
        val_iter = FaceImageIterAge(
            batch_size           = args.batch_size,
            data_shape           = data_shape,
            path_imgrec          = os.path.join(args.age_data_dir, 'val.rec'),
            task                 = args.task,
            shuffle              = False,
            rand_mirror          = False,
            mean                 = mean,
        )
        _metric = MAEMetric()
        val_metric = mx.metric.create(_metric)
        val_metric.reset()
        _metric2 = CUMMetric()
        val_metric2 = mx.metric.create(_metric2)
        val_metric2.reset()
        val_iter.reset()
        for batch in val_iter:
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            for x in data:
                outputs.append(net(x)[2])
            val_metric.update(label, outputs)
            val_metric2.update(label, outputs)
        _value = val_metric.get_name_value()[0][1]
        print('[%d][VMAE]: %f'%(nbatch, _value))
        _value = val_metric2.get_name_value()[0][1]
        if args.task=='age':
          acc = _value
        print('[%d][VCUM]: %f'%(nbatch, _value))
      if len(args.gender_data_dir)>0:
        val_iter = FaceImageIterAge(
            batch_size           = args.batch_size,
            data_shape           = data_shape,
            path_imgrec          = os.path.join(args.gender_data_dir, 'val.rec'),
            task                 = args.task,
            shuffle              = False,
            rand_mirror          = False,
            mean                 = mean,
        )
        _metric = AccMetric()
        val_metric = mx.metric.create(_metric)
        val_metric.reset()
        val_iter.reset()
        for batch in val_iter:
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            for x in data:
                outputs.append(net(x)[1])
            val_metric.update(label, outputs)
        _value = val_metric.get_name_value()[0][1]
        if args.task=='gender':
          acc = _value
        print('[%d][VACC]: %f'%(nbatch, _value))
      return acc


    total_time = 0
    num_epochs = 0
    best_acc = [0]
    highest_acc = [0.0, 0.0]  #lfw and target
    global_step = [0]
    save_step = [0]
    if len(args.lr_steps)==0:
      lr_steps = [100000, 140000, 160000]
      p = 512.0/args.batch_size
      for l in xrange(len(lr_steps)):
        lr_steps[l] = int(lr_steps[l]*p)
    else:
      lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)

    kv = mx.kv.create('device')
    #kv = mx.kv.create('local')
    #_rescale = 1.0/args.ctx_num
    #opt = optimizer.SGD(learning_rate=args.lr, momentum=args.mom, wd=args.wd, rescale_grad=_rescale)
    #opt = optimizer.SGD(learning_rate=args.lr, momentum=args.mom, wd=args.wd)
    if args.mode=='gluon':
      trainer = gluon.Trainer(net.collect_params(), 'sgd', 
              {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.mom, 'multi_precision': True},
              kvstore=kv)
    else:
      _rescale = 1.0/args.ctx_num
      opt = optimizer.SGD(learning_rate=args.lr, momentum=args.mom, wd=args.wd, rescale_grad=_rescale)
      _cb = mx.callback.Speedometer(args.batch_size, 20)
      arg_params = None
      aux_params = None
      data = mx.sym.var('data')
      label = mx.sym.var('softmax_label')
      if args.margin_a>0.0:
        fc7 = net(data, label)
      else:
        fc7 = net(data)
      #sym = mx.symbol.SoftmaxOutput(data=fc7, label = label, name='softmax', normalization='valid')
      ceop = gluon.loss.SoftmaxCrossEntropyLoss()
      loss = ceop(fc7, label) 
      #loss = loss/args.per_batch_size
      loss = mx.sym.mean(loss)
      sym = mx.sym.Group( [mx.symbol.BlockGrad(fc7), mx.symbol.MakeLoss(loss, name='softmax')] )

    def _batch_callback():
      mbatch = global_step[0]
      global_step[0]+=1
      for _lr in lr_steps:
        if mbatch==_lr:
          args.lr *= 0.1
          if args.mode=='gluon':
            trainer.set_learning_rate(args.lr)
          else:
            opt.lr  = args.lr
          print('lr change to', args.lr)
          break

      #_cb(param)
      if mbatch%1000==0:
        print('lr-batch-epoch:',args.lr, mbatch)

      if mbatch>0 and mbatch%args.verbose==0:
        save_step[0]+=1
        msave = save_step[0]
        do_save = False
        is_highest = False
        if args.task=='age' or args.task=='gender':
          acc = val_test(mbatch)
          if acc>=highest_acc[-1]:
            highest_acc[-1] = acc
            is_highest = True
            do_save = True
        else:
          acc_list = ver_test(mbatch)
          if len(acc_list)>0:
            lfw_score = acc_list[0]
            if lfw_score>highest_acc[0]:
              highest_acc[0] = lfw_score
              if lfw_score>=0.998:
                do_save = True
            if acc_list[-1]>=highest_acc[-1]:
              highest_acc[-1] = acc_list[-1]
              if lfw_score>=0.99:
                do_save = True
                is_highest = True
        if args.ckpt==0:
          do_save = False
        elif args.ckpt>1:
          do_save = True
        if do_save:
          print('saving', msave)
          #print('saving gluon params')
          fname = os.path.join(args.prefix, 'model-gluon.params')
          net.save_params(fname)
          fname = os.path.join(args.prefix, 'model')
          net.export(fname, msave)
          #arg, aux = model.get_params()
          #mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
        print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))
      if args.max_steps>0 and mbatch>args.max_steps:
        sys.exit(0)

    def _batch_callback_sym(param):
      _cb(param)
      _batch_callback()


    if args.mode!='gluon':
      model = mx.mod.Module(
          context       = ctx,
          symbol        = sym,
      )
      model.fit(train_iter,
          begin_epoch        = 0,
          num_epoch          = args.end_epoch,
          eval_data          = None,
          eval_metric        = metric,
          kvstore            = 'device',
          optimizer          = opt,
          initializer        = initializer,
          arg_params         = arg_params,
          aux_params         = aux_params,
          allow_missing      = True,
          batch_end_callback = _batch_callback_sym,
          epoch_end_callback = None )
    else:
      loss_weight = 1.0
      if args.task=='age':
        loss_weight = 1.0/AGE
      #loss = gluon.loss.SoftmaxCrossEntropyLoss(weight = loss_weight)
      loss = nd.SoftmaxOutput
      #loss = gluon.loss.SoftmaxCrossEntropyLoss()
      while True:
          #trainer = update_learning_rate(opt.lr, trainer, epoch, opt.lr_factor, lr_steps)
          tic = time.time()
          train_iter.reset()
          metric.reset()
          btic = time.time()
          for i, batch in enumerate(train_iter):
              _batch_callback()
              #data = gluon.utils.split_and_load(batch.data[0].astype(opt.dtype), ctx_list=ctx, batch_axis=0)
              #label = gluon.utils.split_and_load(batch.label[0].astype(opt.dtype), ctx_list=ctx, batch_axis=0)
              data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
              label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
              outputs = []
              Ls = []
              with ag.record():
                  for x, y in zip(data, label):
                      #print(y.asnumpy())
                      if args.task=='':
                        if args.margin_a>0.0:
                          z = net(x,y)
                        else:
                          z = net(x)
                        #print(z[0].shape, z[1].shape)
                      else:
                        z = net(x)
                      if args.task=='gender':
                        L = loss(z[1], y)
                        #L = L/args.per_batch_size
                        Ls.append(L)
                        outputs.append(z[1])
                      elif args.task=='age':
                        for k in xrange(AGE):
                          _z = nd.slice_axis(z[2], axis=1, begin=k*2, end=k*2+2)
                          _y = nd.slice_axis(y, axis=1, begin=k, end=k+1)
                          _y = nd.flatten(_y)
                          L = loss(_z, _y)
                          #L = L/args.per_batch_size
                          #L /= AGE
                          Ls.append(L)
                        outputs.append(z[2])
                      else:
                        L = loss(z, y)
                        #L = L/args.per_batch_size
                        Ls.append(L)
                        outputs.append(z)
                      # store the loss and do backward after we have done forward
                      # on all GPUs for better speed on multiple GPUs.
                  ag.backward(Ls)
              #trainer.step(batch.data[0].shape[0], ignore_stale_grad=True)
              #trainer.step(args.ctx_num)
              n = batch.data[0].shape[0]
              #print(n,n)
              trainer.step(n)
              metric.update(label, outputs)
              if i>0 and i%20==0:
                  name, acc = metric.get()
                  if len(name)==2:
                    logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f, %s=%f'%(
                                   num_epochs, i, args.batch_size/(time.time()-btic), name[0], acc[0], name[1], acc[1]))
                  else:
                    logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f'%(
                                   num_epochs, i, args.batch_size/(time.time()-btic), name[0], acc[0]))
                  #metric.reset()
              btic = time.time()

          epoch_time = time.time()-tic

          # First epoch will usually be much slower than the subsequent epics,
          # so don't factor into the average
          if num_epochs > 0:
            total_time = total_time + epoch_time

          #name, acc = metric.get()
          #logger.info('[Epoch %d] training: %s=%f, %s=%f'%(num_epochs, name[0], acc[0], name[1], acc[1]))
          logger.info('[Epoch %d] time cost: %f'%(num_epochs, epoch_time))
          num_epochs = num_epochs + 1
          #name, val_acc = test(ctx, val_data)
          #logger.info('[Epoch %d] validation: %s=%f, %s=%f'%(epoch, name[0], val_acc[0], name[1], val_acc[1]))

          # save model if meet requirements
          #save_checkpoint(epoch, val_acc[0], best_acc)
      if num_epochs > 1:
          print('Average epoch time: {}'.format(float(total_time)/(num_epochs - 1)))



def main():
    #time.sleep(3600*6.5)
    global args
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

