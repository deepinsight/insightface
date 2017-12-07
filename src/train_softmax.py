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
from data import FaceIter
from data import FaceImageIter
from data import FaceImageIter2
from data import FaceImageIter4
from data import FaceImageIter5
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
#sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
import fresnet
import finception_resnet_v2
import fmobilenet 
import fxception
import fdensenet
#import lfw
import verification
import sklearn


logger = logging.getLogger()
logger.setLevel(logging.INFO)




class AccMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(AccMetric, self).__init__(
        'acc', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []

  def update(self, labels, preds):
    #loss = preds[2].asnumpy()[0]
    #if len(self.losses)==20:
    #  print('ce loss', sum(self.losses)/len(self.losses))
    #  self.losses = []
    #self.losses.append(loss)
    preds = [preds[1]] #use softmax output
    for label, pred_label in zip(labels, preds):
        #print(pred_label)
        #print(label.shape, pred_label.shape)
        if pred_label.shape != label.shape:
            pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.asnumpy().astype('int32').flatten()
        label = label.asnumpy().astype('int32').flatten()
        #print(label)
        #print('label',label)
        #print('pred_label', pred_label)
        assert label.shape==pred_label.shape
        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)

def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--data-dir', default='',
      help='')
  parser.add_argument('--prefix', default='../model/model',
      help='directory to save model.')
  parser.add_argument('--pretrained', default='../model/resnet-152',
      help='')
  parser.add_argument('--network', default='s20', help='')
  parser.add_argument('--use-se', action='store_true', default=False, help='')
  parser.add_argument('--version-input', type=int, default=1, help='')
  parser.add_argument('--version-output', type=str, default='A', help='')
  parser.add_argument('--version-unit', type=int, default=1, help='')
  parser.add_argument('--load-epoch', type=int, default=0,
      help='load epoch.')
  parser.add_argument('--end-epoch', type=int, default=1000,
      help='training epoch size.')
  parser.add_argument('--retrain', action='store_true', default=False,
      help='true means continue training.')
  parser.add_argument('--lr', type=float, default=0.1,
      help='')
  parser.add_argument('--wd', type=float, default=0.0005,
      help='')
  parser.add_argument('--images-per-identity', type=int, default=16,
      help='')
  parser.add_argument('--embedding-dim', type=int, default=512,
      help='')
  parser.add_argument('--per-batch-size', type=int, default=0,
      help='')
  parser.add_argument('--margin', type=int, default=4,
      help='')
  parser.add_argument('--beta', type=float, default=1000.,
      help='')
  parser.add_argument('--beta-min', type=float, default=5.,
      help='')
  parser.add_argument('--beta-freeze', type=int, default=0,
      help='')
  parser.add_argument('--gamma', type=float, default=0.12,
      help='')
  parser.add_argument('--power', type=float, default=1.0,
      help='')
  parser.add_argument('--scale', type=float, default=0.9993,
      help='')
  parser.add_argument('--verbose', type=int, default=2000,
      help='')
  parser.add_argument('--loss-type', type=int, default=1,
      help='')
  parser.add_argument('--incay', type=float, default=0.0,
      help='feature incay')
  parser.add_argument('--use-deformable', type=int, default=0,
      help='')
  parser.add_argument('--image-size', type=str, default='112,96',
      help='')
  parser.add_argument('--patch', type=str, default='0_0_96_112_0',
      help='')
  parser.add_argument('--lr-steps', type=str, default='',
      help='')
  args = parser.parse_args()
  return args


def get_symbol(args, arg_params, aux_params):
  if args.retrain:
    new_args = arg_params
  else:
    new_args = None
  data_shape = (args.image_channel,args.image_h,args.image_w)
  image_shape = ",".join([str(x) for x in data_shape])
  if args.network[0]=='d':
    embedding = fdensenet.get_symbol(512, args.num_layers,
        use_se=args.use_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit)
  elif args.network[0]=='m':
    print('init mobilenet', args.num_layers)
    embedding = fmobilenet.get_symbol(512, 
        use_se=args.use_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit)
  elif args.network[0]=='i':
    print('init inception-resnet-v2', args.num_layers)
    embedding = finception_resnet_v2.get_symbol(512)
  elif args.network[0]=='x':
    print('init xception', args.num_layers)
    embedding = fxception.get_xception_symbol(512,
        use_se=args.use_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit)
  else:
    print('init resnet', args.num_layers)
    embedding = fresnet.get_symbol(512, args.num_layers, 
        use_se=args.use_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit)
  gt_label = mx.symbol.Variable('softmax_label')
  assert args.loss_type>=0
  extra_loss = None
  if args.loss_type==0:
    _weight = mx.symbol.Variable('fc7_weight')
    _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
    fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=args.num_classes, name='fc7')
  elif args.loss_type==1:
    _weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, 512), lr_mult=1.0)
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
    fc7 = mx.sym.LSoftmax(data=embedding, label=gt_label, num_hidden=args.num_classes,
                          weight = _weight,
                          beta=args.beta, margin=args.margin, scale=args.scale,
                          beta_min=args.beta_min, verbose=100, name='fc7')
  elif args.loss_type==10:
    _weight = mx.symbol.Variable('fc7_weight')
    _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
    fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=args.num_classes, name='fc7')
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
    params = [1.2, 0.3, 1.0]
    n1 = mx.sym.expand_dims(nembedding, axis=1)
    n2 = mx.sym.expand_dims(nembedding, axis=0)
    body = mx.sym.broadcast_sub(n1, n2) #N,N,C
    body = body * body
    body = mx.sym.sum(body, axis=2) # N,N
    #body = mx.sym.sqrt(body)
    body = body - params[0]
    mask = mx.sym.Variable('extra')
    body = body*mask
    body = body+params[1]
    #body = mx.sym.maximum(body, 0.0)
    body = mx.symbol.Activation(data=body, act_type='relu')
    body = mx.sym.sum(body)
    body = body/(args.per_batch_size*args.per_batch_size-args.per_batch_size)
    extra_loss = mx.symbol.MakeLoss(body, grad_scale=params[2])
  elif args.loss_type==11:
    _weight = mx.symbol.Variable('fc7_weight')
    _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
    fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=args.num_classes, name='fc7')
    params = [0.9, 0.2]
    nembedding = mx.symbol.slice_axis(embedding, axis=0, begin=0, end=args.images_per_identity)
    nembedding = mx.symbol.L2Normalization(nembedding, mode='instance', name='fc1n')
    n1 = mx.sym.expand_dims(nembedding, axis=1)
    n2 = mx.sym.expand_dims(nembedding, axis=0)
    body = mx.sym.broadcast_sub(n1, n2) #N,N,C
    body = body * body
    body = mx.sym.sum(body, axis=2) # N,N
    body = body - params[0]
    body = mx.symbol.Activation(data=body, act_type='relu')
    body = mx.sym.sum(body)
    n = args.images_per_identity
    body = body/(n*n-n)
    extra_loss = mx.symbol.MakeLoss(body, grad_scale=params[1])
    #extra_loss = None
  else:
    #embedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*float(args.loss_type)
    embedding = embedding * 5
    _weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, 512), lr_mult=1.0)
    _weight = mx.symbol.L2Normalization(_weight, mode='instance') * 2
    fc7 = mx.sym.LSoftmax(data=embedding, label=gt_label, num_hidden=args.num_classes,
                          weight = _weight,
                          beta=args.beta, margin=args.margin, scale=args.scale,
                          beta_min=args.beta_min, verbose=100, name='fc7')

    #fc7 = mx.sym.Custom(data=embedding, label=gt_label, weight=_weight, num_hidden=args.num_classes,
    #                       beta=args.beta, margin=args.margin, scale=args.scale,
    #                       op_type='ASoftmax', name='fc7')
  softmax = mx.symbol.SoftmaxOutput(data=fc7, label = gt_label, name='softmax', normalization='valid')
  if args.loss_type<=1 and args.incay>0.0:
    params = [1.e-10]
    sel = mx.symbol.argmax(data = fc7, axis=1)
    sel = (sel==gt_label)
    norm = embedding*embedding
    norm = mx.symbol.sum(norm, axis=1)
    norm = norm+params[0]
    feature_incay = sel/norm
    feature_incay = mx.symbol.mean(feature_incay) * args.incay
    extra_loss = mx.symbol.MakeLoss(feature_incay)
  #out = softmax
  #l2_embedding = mx.symbol.L2Normalization(embedding)

  #ce = mx.symbol.softmax_cross_entropy(fc7, gt_label, name='softmax_ce')/args.per_batch_size
  #out = mx.symbol.Group([mx.symbol.BlockGrad(embedding), softmax, mx.symbol.BlockGrad(ce)])
  if extra_loss is not None:
    out = mx.symbol.Group([mx.symbol.BlockGrad(embedding), softmax, extra_loss])
  else:
    out = mx.symbol.Group([mx.symbol.BlockGrad(embedding), softmax])
  return (out, new_args, aux_params)

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
    pretrained = args.pretrained
    load_epoch = args.load_epoch
    args.ctx_num = len(ctx)
    args.num_layers = int(args.network[1:])
    print('num_layers', args.num_layers)
    if args.per_batch_size==0:
      args.per_batch_size = 128
      if args.network[0]=='r':
        args.per_batch_size = 128
      else:
        if args.num_layers>=64:
          args.per_batch_size = 120
      if args.ctx_num==2:
        args.per_batch_size *= 2
      elif args.ctx_num==3:
        args.per_batch_size = 170
      if args.network[0]=='m':
        args.per_batch_size = 128
    args.batch_size = args.per_batch_size*args.ctx_num
    args.rescale_threshold = 0
    args.image_channel = 3
    ppatch = [int(x) for x in args.patch.split('_')]
    image_size = [int(x) for x in args.image_size.split(',')]
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    assert len(ppatch)==5


    os.environ['BETA'] = str(args.beta)
    args.use_val = False
    path_imgrec = None
    path_imglist = None
    val_rec = None

    for line in open(os.path.join(args.data_dir, 'property')):
      args.num_classes = int(line.strip())
    assert(args.num_classes>0)
    print('num_classes', args.num_classes)

    #path_imglist = "/raid5data/dplearn/MS-Celeb-Aligned/lst2"
    path_imgrec = os.path.join(args.data_dir, "train.rec")
    val_rec = os.path.join(args.data_dir, "val.rec")
    if os.path.exists(val_rec):
      args.use_val = True
    else:
      val_rec = None
    #args.num_classes = 10572 #webface
    #args.num_classes = 81017
    #args.num_classes = 82395



    if args.loss_type==1 and args.num_classes>40000:
      args.beta_freeze = 5000
      args.gamma = 0.06

    print('Called with argument:', args)

    data_shape = (args.image_channel,image_size[0],image_size[1])
    mean = None

    if args.use_val:
      val_dataiter = FaceImageIter(
          batch_size           = args.batch_size,
          data_shape           = data_shape,
          path_imgrec          = val_rec,
          #path_imglist         = val_path,
          shuffle              = False,
          rand_mirror          = False,
          mean                 = mean,
      )
    else:
      val_dataiter = None



    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd
    base_mom = 0.9
    if not args.retrain:
      arg_params = None
      aux_params = None
      sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
    else:
      _, arg_params, aux_params = mx.model.load_checkpoint(pretrained, load_epoch)
      sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)


    if args.loss_type!=10:
      model = mx.mod.Module(
          context       = ctx,
          symbol        = sym,
      )
    else:
      data_names = ('data', 'extra')
      model = mx.mod.Module(
          context       = ctx,
          symbol        = sym,
          data_names    = data_names,
      )


    if args.loss_type<=9:
      train_dataiter = FaceImageIter(
          batch_size           = args.batch_size,
          data_shape           = data_shape,
          path_imgrec          = path_imgrec,
          shuffle              = True,
          rand_mirror          = True,
          mean                 = mean,
      )
    elif args.loss_type==10:
      train_dataiter = FaceImageIter4(
          batch_size           = args.batch_size,
          ctx_num              = args.ctx_num,
          images_per_identity  = args.images_per_identity,
          data_shape           = data_shape,
          path_imglist         = path_imglist,
          shuffle              = True,
          rand_mirror          = True,
          mean                 = mean,
          patch                = ppatch,
          use_extra            = True,
          model                = model,
      )
    elif args.loss_type==11:
      train_dataiter = FaceImageIter5(
          batch_size           = args.batch_size,
          ctx_num              = args.ctx_num,
          images_per_identity  = args.images_per_identity,
          data_shape           = data_shape,
          path_imglist         = path_imglist,
          shuffle              = True,
          rand_mirror          = True,
          mean                 = mean,
          patch                = ppatch,
      )

    _acc = AccMetric()
    eval_metrics = [mx.metric.create(_acc)]

    if args.network[0]=='r':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    elif args.network[0]=='i' or args.network[0]=='x':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2) #inception
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    _rescale = 1.0/args.ctx_num
    opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    _cb = mx.callback.Speedometer(args.batch_size, 10)

    ver_list = []
    ver_name_list = []
    for name in ['lfw','cfp_ff','cfp_fp','agedb_30']:
      path = os.path.join(args.data_dir,name+".bin")
      if os.path.exists(path):
        data_set = verification.load_bin(path, image_size)
        ver_list.append(data_set)
        ver_name_list.append(name)
        print('ver', name)



    def ver_test(nbatch):
      results = []
      for i in xrange(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, args.batch_size)
        print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
        print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
        print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
        results.append(acc2)
      return results


    def val_test():
      acc = AccMetric()
      val_metric = mx.metric.create(acc)
      val_metric.reset()
      val_dataiter.reset()
      for i, eval_batch in enumerate(val_dataiter):
        model.forward(eval_batch, is_train=False)
        model.update_metric(val_metric, eval_batch.label)
      acc_value = val_metric.get_name_value()[0][1]
      print('VACC: %f'%(acc_value))


    highest_acc = []
    for i in xrange(len(ver_list)):
      highest_acc.append(0.0)
    global_step = [0]
    save_step = [0]
    if len(args.lr_steps)==0:
      lr_steps = [40000, 60000, 80000]
      if args.loss_type==1:
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
        if mbatch==args.beta_freeze+_lr:
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
        lfw_score = acc_list[0]
        for i in xrange(len(acc_list)):
          acc = acc_list[i]
          if acc>=highest_acc[i]:
            highest_acc[i] = acc
            if lfw_score>=0.99:
              do_save = True
        if args.loss_type==1 and mbatch>lr_steps[-1] and mbatch%10000==0:
          do_save = True
        if do_save:
          print('saving', msave, acc)
          if val_dataiter is not None:
            val_test()
          arg, aux = model.get_params()
          mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
          #if acc>=highest_acc[0]:
          #  lfw_npy = "%s-lfw-%04d" % (prefix, msave)
          #  X = np.concatenate(embeddings_list, axis=0)
          #  print('saving lfw npy', X.shape)
          #  np.save(lfw_npy, X)
        #print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[0]))
      if mbatch<=args.beta_freeze:
        _beta = args.beta
      else:
        move = max(0, mbatch-args.beta_freeze)
        _beta = max(args.beta_min, args.beta*math.pow(1+args.gamma*move, -1.0*args.power))
      #print('beta', _beta)
      os.environ['BETA'] = str(_beta)

    #epoch_cb = mx.callback.do_checkpoint(prefix, 1)
    epoch_cb = None



    #def _epoch_callback(epoch, sym, arg_params, aux_params):
    #  print('epoch-end', epoch)

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
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

