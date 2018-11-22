# THIS FILE IS FOR EXPERIMENTS, USE train_softmax.py FOR NORMAL TRAINING.
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
from triplet_image_iter import FaceImageIter
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_image
from noise_sgd import NoiseSGD
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
import fresnet
import finception_resnet_v2
import fmobilenet 
import fmobilenetv2
import fxception
import fdensenet
import fdpn
import fnasnet
import spherenet
#import lfw
import verification
import sklearn
sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))
import center_loss


logger = logging.getLogger()
logger.setLevel(logging.INFO)


args = None


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
  parser.add_argument('--ckpt', type=int, default=3, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
  parser.add_argument('--network', default='r50', help='specify network')
  parser.add_argument('--version-se', type=int, default=0, help='whether to use se in network')
  parser.add_argument('--version-input', type=int, default=1, help='network input config')
  parser.add_argument('--version-output', type=str, default='E', help='network embedding output config')
  parser.add_argument('--version-unit', type=int, default=3, help='resnet unit config')
  parser.add_argument('--version-act', type=str, default='prelu', help='network activation config')
  parser.add_argument('--end-epoch', type=int, default=100000, help='training epoch size.')
  parser.add_argument('--noise-sgd', type=float, default=0.0, help='')
  parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
  parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
  parser.add_argument('--mom', type=float, default=0.9, help='momentum')
  parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
  parser.add_argument('--per-batch-size', type=int, default=128, help='batch size in each context')
  parser.add_argument('--images-per-identity', type=int, default=5, help='')
  parser.add_argument('--triplet-bag-size', type=int, default=3600, help='')
  parser.add_argument('--triplet-alpha', type=float, default=0.3, help='')
  parser.add_argument('--triplet-max-ap', type=float, default=0.0, help='')
  parser.add_argument('--verbose', type=int, default=2000, help='')
  parser.add_argument('--loss-type', type=int, default=1, help='')
  parser.add_argument('--use-deformable', type=int, default=0, help='')
  parser.add_argument('--rand-mirror', type=int, default=1, help='')
  parser.add_argument('--cutoff', type=int, default=0, help='')
  parser.add_argument('--lr-steps', type=str, default='', help='')
  parser.add_argument('--max-steps', type=int, default=0, help='')
  parser.add_argument('--target', type=str, default='lfw,cfp_fp,agedb_30', help='')
  args = parser.parse_args()
  return args


def get_symbol(args, arg_params, aux_params, sym_embedding=None):
  if sym_embedding is None:
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
    else:
      print('init resnet', args.num_layers)
      embedding = fresnet.get_symbol(args.emb_size, args.num_layers, 
          version_se=args.version_se, version_input=args.version_input, 
          version_output=args.version_output, version_unit=args.version_unit,
          version_act=args.version_act)
  else:
    embedding = sym_embedding

  assert args.loss_type==1 or args.loss_type==2
  gt_label = mx.symbol.Variable('softmax_label')
  nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
  anchor = mx.symbol.slice_axis(nembedding, axis=0, begin=0, end=args.per_batch_size//3)
  positive = mx.symbol.slice_axis(nembedding, axis=0, begin=args.per_batch_size//3, end=2*args.per_batch_size//3)
  negative = mx.symbol.slice_axis(nembedding, axis=0, begin=2*args.per_batch_size//3, end=args.per_batch_size)
  if args.loss_type==1:
    ap = anchor - positive
    an = anchor - negative
    ap = ap*ap
    an = an*an
    ap = mx.symbol.sum(ap, axis=1, keepdims=1) #(T,1)
    an = mx.symbol.sum(an, axis=1, keepdims=1) #(T,1)
    triplet_loss = mx.symbol.Activation(data = (ap-an+args.triplet_alpha), act_type='relu')
    triplet_loss = mx.symbol.mean(triplet_loss)
  elif args.loss_type==2:
    ap = anchor*positive
    an = anchor*negative
    ap = mx.symbol.sum(ap, axis=1, keepdims=1) #(T,1)
    an = mx.symbol.sum(an, axis=1, keepdims=1) #(T,1)
    ap = mx.sym.arccos(ap)
    an = mx.sym.arccos(an)
    triplet_loss = mx.symbol.Activation(data = (ap-an+args.triplet_alpha), act_type='relu')
    triplet_loss = mx.symbol.mean(triplet_loss)
  #triplet_loss = mx.symbol.sum(triplet_loss)/(args.per_batch_size//3)
  triplet_loss = mx.symbol.MakeLoss(triplet_loss)
  out_list = [mx.symbol.BlockGrad(embedding)]
  out_list.append(mx.sym.BlockGrad(gt_label))
  out_list.append(triplet_loss)
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
    args.image_channel = 3


    data_dir_list = args.data_dir.split(',')
    assert len(data_dir_list)==1
    data_dir = data_dir_list[0]
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

    #path_imglist = "/raid5data/dplearn/MS-Celeb-Aligned/lst2"
    path_imgrec = os.path.join(data_dir, "train.rec")

    assert args.images_per_identity>=2
    assert args.triplet_bag_size%args.batch_size==0

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
      sym, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
      all_layers = sym.get_internals()
      sym = all_layers['fc1_output']
      sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params, sym_embedding = sym)

    data_extra = None
    hard_mining = False
    triplet_params = [args.triplet_bag_size, args.triplet_alpha, args.triplet_max_ap]
    model = mx.mod.Module(
        context       = ctx,
        symbol        = sym,
        #data_names = ('data',),
        #label_names = None,
        #label_names = ('softmax_label',),
    )
    label_shape = (args.batch_size,)

    val_dataiter = None

    train_dataiter = FaceImageIter(
        batch_size           = args.batch_size,
        data_shape           = data_shape,
        path_imgrec          = path_imgrec,
        shuffle              = True,
        rand_mirror          = args.rand_mirror,
        mean                 = mean,
        cutoff               = args.cutoff,
        ctx_num              = args.ctx_num,
        images_per_identity  = args.images_per_identity,
        triplet_params       = triplet_params,
        mx_model             = model,
    )

    _metric = LossValueMetric()
    eval_metrics = [mx.metric.create(_metric)]

    if args.network[0]=='r':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    elif args.network[0]=='i' or args.network[0]=='x':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2) #inception
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    _rescale = 1.0/args.ctx_num
    if args.noise_sgd>0.0:
      print('use noise sgd')
      opt = NoiseSGD(scale = args.noise_sgd, learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    else:
      opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    som = 2
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
        acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, args.batch_size, 10, None, label_shape)
        print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
        #print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
        print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
        results.append(acc2)
      return results


    highest_acc = [0.0, 0.0]  #lfw and target
    #for i in xrange(len(ver_list)):
    #  highest_acc.append(0.0)
    global_step = [0]
    save_step = [0]
    if len(args.lr_steps)==0:
      lr_steps = [1000000000]
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
        acc_list = ver_test(mbatch)
        save_step[0]+=1
        msave = save_step[0]
        do_save = False
        is_highest = False
        if len(acc_list)>0:
          #lfw_score = acc_list[0]
          #if lfw_score>highest_acc[0]:
          #  highest_acc[0] = lfw_score
          #  if lfw_score>=0.998:
          #    do_save = True
          score = sum(acc_list)
          if acc_list[-1]>=highest_acc[-1]:
            if acc_list[-1]>highest_acc[-1]:
              is_highest = True
            else:
              if score>=highest_acc[0]:
                is_highest = True
                highest_acc[0] = score
            highest_acc[-1] = acc_list[-1]
            #if lfw_score>=0.99:
            #  do_save = True
        if is_highest:
          do_save = True
        if args.ckpt==0:
          do_save = False
        elif args.ckpt==2:
          do_save = True
        elif args.ckpt==3:
          msave = 1

        if do_save:
          print('saving', msave)
          arg, aux = model.get_params()
          mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
        print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))
      if args.max_steps>0 and mbatch>args.max_steps:
        sys.exit(0)

    #epoch_cb = mx.callback.do_checkpoint(prefix, 1)
    epoch_cb = None

    model.fit(train_dataiter,
        begin_epoch        = begin_epoch,
        num_epoch          = end_epoch,
        eval_data          = val_dataiter,
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

