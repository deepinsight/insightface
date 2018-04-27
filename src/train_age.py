from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import copy
import logging
import pickle
import numpy as np
from age_iter import FaceImageIter
from age_iter import FaceImageIterList
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_image
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
import fresnet
import finception_resnet_v2
import fmobilenet 
import fmobilenetv2
import fmobilefacenet
import fxception
import fdensenet
import fdpn
import fnasnet
import spherenet
import verification
import sklearn
#sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))
#import center_loss


logger = logging.getLogger()
logger.setLevel(logging.INFO)


args = None

AGE = 100

USE_FR = False
USE_GENDER = False
USE_AGE = True


class AccMetric(mx.metric.EvalMetric):
  def __init__(self, pred_idx = 1, label_idx = 0, name='acc'):
    self.axis = 1
    self.pred_idx = pred_idx
    self.label_idx = label_idx
    super(AccMetric, self).__init__(
        name, axis=self.axis,
        output_names=None, label_names=None)
    self.name = name
    self.losses = []
    self.count = 0

  def update(self, labels, preds):
    self.count+=1

    #print('label num', len(labels))
    preds = [preds[self.pred_idx]] #use softmax output
    for label, pred_label in zip(labels, preds):
        if pred_label.shape != label.shape:
            pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.asnumpy().astype('int32').flatten()
        label = label.asnumpy()
        if label.ndim==2:
          label = label[:,self.label_idx]
        label = label.astype('int32').flatten()
        print(self.name, label)
        assert label.shape==pred_label.shape
        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)

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
    label = label[:,(AGE*-1):]
    label_age = np.count_nonzero(label, axis=1)
    pred_age = np.zeros( label_age.shape, dtype=np.int)
    for i in xrange(-1*AGE, -1):
        pred = preds[i].asnumpy()
        pred = np.argmax(pred, axis=1)
        pred_age += pred
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
    label = label[:,(AGE*-1):]
    label_age = np.count_nonzero(label, axis=1)
    pred_age = np.zeros( label_age.shape, dtype=np.int)
    for i in xrange(-1*AGE, -1):
        pred = preds[i].asnumpy()
        pred = np.argmax(pred, axis=1)
        pred_age += pred
    diff = np.abs(label_age - pred_age)
    cum = np.sum( (diff<self.n) )

    self.sum_metric += cum
    self.num_inst += len(label_age)

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
  parser.add_argument('--version-se', type=int, default=0, help='whether to use se in network')
  parser.add_argument('--version-input', type=int, default=1, help='network input config')
  parser.add_argument('--version-output', type=str, default='E', help='network embedding output config')
  parser.add_argument('--version-unit', type=int, default=3, help='resnet unit config')
  parser.add_argument('--version-act', type=str, default='prelu', help='network activation config')
  parser.add_argument('--use-deformable', type=int, default=0, help='use deformable cnn in network')
  parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
  parser.add_argument('--lr-steps', type=str, default='', help='steps of lr changing')
  parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
  parser.add_argument('--mom', type=float, default=0.9, help='momentum')
  parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
  parser.add_argument('--per-batch-size', type=int, default=128, help='batch size in each context')
  parser.add_argument('--margin-m', type=float, default=0.5, help='margin for loss')
  parser.add_argument('--margin-s', type=float, default=64.0, help='scale for feature')
  parser.add_argument('--margin-a', type=float, default=1.0, help='')
  parser.add_argument('--margin-b', type=float, default=0.0, help='')
  parser.add_argument('--easy-margin', type=int, default=0, help='')
  parser.add_argument('--rand-mirror', type=int, default=1, help='if do random mirror in training')
  parser.add_argument('--cutoff', type=int, default=0, help='cut off aug')
  parser.add_argument('--target', type=str, default='lfw,cfp_fp,agedb_30', help='verification targets')
  parser.add_argument('--ignore-label', type=int, default=-1, help='ignore label')
  args = parser.parse_args()
  return args


def get_softmax(args, embedding, nembedding, gt_label, name):
    s = args.margin_s
    m = args.margin_m
    assert s>0.0
    if args.margin_a==0.0:
        _weight = mx.symbol.Variable(name+"_weight", shape=(args.num_classes, args.emb_size), lr_mult=1.0)
        _bias = mx.symbol.Variable(name+'_bias', lr_mult=2.0, wd_mult=0.0)
        fc = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=args.num_classes, name=name)
    else:
        _weight = mx.symbol.Variable(name+"_weight", shape=(args.num_classes, args.emb_size), lr_mult=1.0)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        fc = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.num_classes, name=name)
        if args.margin_a!=1.0 or args.margin_m!=0.0 or args.margin_b!=0.0:
          if args.margin_a==1.0 and args.margin_m==0.0:
            s_m = s*args.margin_b
            gt_one_hot = mx.sym.one_hot(gt_label, depth = args.num_classes, on_value = s_m, off_value = 0.0)
            fc = fc-gt_one_hot
          else:
            zy = mx.sym.pick(fc, gt_label, axis=1)
            cos_t = zy/s
            t = mx.sym.arccos(cos_t)
            if args.margin_a!=1.0:
              t = t*args.margin_a
            if args.margin_m>0.0:
              t = t+args.margin_m
            body = mx.sym.cos(t)
            if args.margin_b>0.0:
              body = body - args.margin_b
            new_zy = body*s
            diff = new_zy - zy
            diff = mx.sym.expand_dims(diff, 1)
            gt_one_hot = mx.sym.one_hot(gt_label, depth = args.num_classes, on_value = 1.0, off_value = 0.0)
            body = mx.sym.broadcast_mul(gt_one_hot, diff)
            fc = fc+body
    if args.ignore_label==0:
        softmax = mx.symbol.SoftmaxOutput(data=fc, label = gt_label, name=name+'_softmax', normalization='valid', grad_scale = args.grad_scale)
    else:
        softmax = mx.symbol.SoftmaxOutput(data=fc, label = gt_label, name=name+'_softmax', normalization='valid', use_ignore=True, ignore_label=args.ignore_label, grad_scale = args.grad_scale)
    return softmax

def get_symbol(args, arg_params, aux_params):
  data_shape = (args.image_channel,args.image_h,args.image_w)
  image_shape = ",".join([str(x) for x in data_shape])
  margin_symbols = []
  if args.network[0]=='d':
    embedding = fdensenet.get_symbol(args.emb_size, args.num_layers,
        version_se=args.version_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit)
  elif args.network[0]=='m':
    print('init mobilenet', args.num_layers)
    if args.num_layers==1:
      embedding = fmobilenet.get_symbol(args.emb_size, 
          version_se=args.version_se, version_input=args.version_input, 
          version_output=args.version_output, version_unit=args.version_unit)
    else:
      embedding = fmobilenetv2.get_symbol(args.emb_size)
  elif args.network[0]=='i':
    print('init inception-resnet-v2', args.num_layers)
    embedding = finception_resnet_v2.get_symbol(args.emb_size,
        version_se=args.version_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit)
  elif args.network[0]=='x':
    print('init xception', args.num_layers)
    embedding = fxception.get_symbol(args.emb_size,
        version_se=args.version_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit)
  elif args.network[0]=='p':
    print('init dpn', args.num_layers)
    embedding = fdpn.get_symbol(args.emb_size, args.num_layers,
        version_se=args.version_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit)
  elif args.network[0]=='n':
    print('init nasnet', args.num_layers)
    embedding = fnasnet.get_symbol(args.emb_size)
  elif args.network[0]=='s':
    print('init spherenet', args.num_layers)
    embedding = spherenet.get_symbol(args.emb_size, args.num_layers)
  elif args.network[0]=='y':
    print('init mobilefacenet', args.num_layers)
    embedding = fmobilefacenet.get_symbol(args.emb_size)
  else:
    print('init resnet', args.num_layers)
    embedding = fresnet.get_symbol(args.emb_size, args.num_layers, 
        version_se=args.version_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit,
        version_act=args.version_act)
  all_label = mx.symbol.Variable('softmax_label')
  gt_label = all_label
  extra_loss = None
  s = args.margin_s
  #m = args.margin_m
  assert s>0.0
  nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s
  out_list = [mx.symbol.BlockGrad(embedding)]

  _args = copy.deepcopy(args)

  if USE_FR:
      _args.grad_scale = 1.0
      fr_label = mx.symbol.slice_axis(all_label, axis=1, begin=0, end=1)
      fr_label = mx.symbol.reshape(fr_label, (args.per_batch_size,))
      fr_softmax = get_softmax(_args, embedding, nembedding, fr_label, 'fc7')
      out_list.append(fr_softmax)

  if USE_GENDER:
      _args.grad_scale = 0.2
      _args.margin_a = 0.0
      _args.num_classes = 2
      gender_label = mx.symbol.slice_axis(all_label, axis=1, begin=1, end=2)
      gender_label = mx.symbol.reshape(gender_label, (args.per_batch_size,))
      gender_softmax = get_softmax(_args, embedding, nembedding, gender_label, 'fc8')
      out_list.append(gender_softmax)

  if USE_AGE:
      _args.grad_scale = 0.01
      _args.margin_a = 0.0
      _args.num_classes = 2
      for i in xrange(AGE):
          age_label = mx.symbol.slice_axis(all_label, axis=1, begin=2+i, end=3+i)
          age_label = mx.symbol.reshape(age_label, (args.per_batch_size,))
          age_softmax = get_softmax(_args, embedding, nembedding, age_label, 'fc9_%d'%(i))
          out_list.append(age_softmax)

  out = mx.symbol.Group(out_list)
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
    args.num_classes = 0
    image_size = (112,112)
    if os.path.exists(os.path.join(data_dir, 'property')):
      prop = face_image.load_property(data_dir)
      args.num_classes = prop.num_classes
      image_size = prop.image_size
      assert(args.num_classes>0)
      print('num_classes', args.num_classes)
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    print('image_size', image_size)
    path_imgrec = os.path.join(data_dir, "train.rec")

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
    if args.network[0]=='s':
      data_shape_dict = {'data' : (args.per_batch_size,)+data_shape}
      spherenet.init_weights(sym, data_shape_dict, args.num_layers)

    #label_name = 'softmax_label'
    #label_shape = (args.batch_size,)
    model = mx.mod.Module(
        context       = ctx,
        symbol        = sym,
    )

    train_dataiter = FaceImageIter(
        batch_size           = args.batch_size,
        data_shape           = data_shape,
        path_imgrec          = path_imgrec,
        shuffle              = True,
        rand_mirror          = args.rand_mirror,
        mean                 = mean,
        cutoff               = args.cutoff,
    )
    val_rec = os.path.join(data_dir, "val.rec")
    val_iter = None
    if os.path.exists(val_rec):
        val_iter = FaceImageIter(
            batch_size           = args.batch_size,
            data_shape           = data_shape,
            path_imgrec          = val_rec,
            shuffle              = False,
            rand_mirror          = False,
            mean                 = mean,
        )

    eval_metrics = []
    if USE_FR:
      _metric = AccMetric(pred_idx=1, label_idx=0)
      eval_metrics.append(_metric)
      if USE_GENDER:
          _metric = AccMetric(pred_idx=2, label_idx=1, name='gender')
          eval_metrics.append(_metric)
    elif USE_GENDER:
      _metric = AccMetric(pred_idx=1, label_idx=1, name='gender')
      eval_metrics.append(_metric)
    if USE_AGE:
      _metric = MAEMetric()
      eval_metrics.append(_metric)
      _metric = CUMMetric()
      eval_metrics.append(_metric)

    if args.network[0]=='r':
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

    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
      path = os.path.join(data_dir,name+".bin")
      if os.path.exists(path):
        data_set = verification.load_bin(path, image_size)
        ver_list.append(data_set)
        ver_name_list.append(name)
        print('ver', name)



    def ver_test(nbatch):
      results = []
      for i in xrange(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, args.batch_size, 10, None, None)
        print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
        #print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
        print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
        results.append(acc2)
      return results

    def val_test():
      _metric = MAEMetric()
      val_metric = mx.metric.create(_metric)
      val_metric.reset()
      _metric2 = CUMMetric()
      val_metric2 = mx.metric.create(_metric2)
      val_metric2.reset()
      val_iter.reset()
      for i, eval_batch in enumerate(val_iter):
        model.forward(eval_batch, is_train=False)
        model.update_metric(val_metric, eval_batch.label)
        model.update_metric(val_metric2, eval_batch.label)
      _value = val_metric.get_name_value()[0][1]
      print('MAE: %f'%(_value))
      _value = val_metric2.get_name_value()[0][1]
      print('CUM: %f'%(_value))


    highest_acc = [0.0, 0.0]  #lfw and target
    #for i in xrange(len(ver_list)):
    #  highest_acc.append(0.0)
    global_step = [0]
    save_step = [0]
    if len(args.lr_steps)==0:
      lr_steps = [40000, 60000, 80000]
      if args.loss_type>=1 and args.loss_type<=7:
        lr_steps = [100000, 140000, 160000]
      p = 512.0/args.batch_size
      for l in xrange(len(lr_steps)):
        lr_steps[l] = int(lr_steps[l]*p)
    else:
      lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)
    def _batch_callback(param):
      #global global_step
      global_step[0]+=1
      mbatch = global_step[0]
      for _lr in lr_steps:
        if mbatch==_lr:
          opt.lr *= 0.1
          print('lr change to', opt.lr)
          break

      _cb(param)
      if mbatch%1000==0:
        print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)

      if mbatch>=0 and mbatch%args.verbose==0:
        if val_iter is not None:
            val_test()
        acc_list = ver_test(mbatch)
        save_step[0]+=1
        msave = save_step[0]
        do_save = False
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
        if args.ckpt==0:
          do_save = False
        elif args.ckpt>1:
          do_save = True
        if do_save:
          print('saving', msave)
          arg, aux = model.get_params()
          mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
        print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))
      if args.max_steps>0 and mbatch>args.max_steps:
        sys.exit(0)

    epoch_cb = None

    model.fit(train_dataiter,
        begin_epoch        = begin_epoch,
        num_epoch          = end_epoch,
        eval_data          = None,
        eval_metric        = eval_metrics,
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

