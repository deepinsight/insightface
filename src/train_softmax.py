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
from data import FaceImageIter
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
import fxception
import fdensenet
import fdpn
#import lfw
import verification
import sklearn
sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))
import center_loss


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
  parser.add_argument('--data-dir', default='',
      help='')
  parser.add_argument('--prefix', default='../model/model',
      help='directory to save model.')
  parser.add_argument('--pretrained', default='',
      help='')
  parser.add_argument('--retrain', action='store_true', default=False,
      help='true means continue training.')
  parser.add_argument('--network', default='s20', help='')
  parser.add_argument('--version-se', type=int, default=1, help='')
  parser.add_argument('--version-input', type=int, default=1, help='')
  parser.add_argument('--version-output', type=str, default='E', help='')
  parser.add_argument('--version-unit', type=int, default=3, help='')
  parser.add_argument('--end-epoch', type=int, default=100000,
      help='training epoch size.')
  parser.add_argument('--lr', type=float, default=0.1,
      help='')
  parser.add_argument('--wd', type=float, default=0.0005,
      help='')
  parser.add_argument('--mom', type=float, default=0.9,
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
  parser.add_argument('--center-alpha', type=float, default=0.5, help='')
  parser.add_argument('--center-scale', type=float, default=0.003, help='')
  parser.add_argument('--images-per-identity', type=int, default=0, help='')
  parser.add_argument('--triplet-bag-size', type=int, default=3600, help='')
  parser.add_argument('--triplet-alpha', type=float, default=0.2, help='')
  parser.add_argument('--verbose', type=int, default=2000, help='')
  parser.add_argument('--loss-type', type=int, default=1,
      help='')
  parser.add_argument('--incay', type=float, default=0.0,
      help='feature incay')
  parser.add_argument('--use-deformable', type=int, default=0,
      help='')
  parser.add_argument('--patch', type=str, default='0_0_96_112_0',
      help='')
  parser.add_argument('--lr-steps', type=str, default='',
      help='')
  parser.add_argument('--target', type=str, default='lfw,cfp_ff,cfp_fp,agedb_30', help='')
  args = parser.parse_args()
  return args


def get_symbol(args, arg_params, aux_params):
  data_shape = (args.image_channel,args.image_h,args.image_w)
  image_shape = ",".join([str(x) for x in data_shape])
  if args.network[0]=='d':
    embedding = fdensenet.get_symbol(512, args.num_layers,
        version_se=args.version_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit)
  elif args.network[0]=='m':
    print('init mobilenet', args.num_layers)
    embedding = fmobilenet.get_symbol(512, 
        version_se=args.version_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit)
  elif args.network[0]=='i':
    print('init inception-resnet-v2', args.num_layers)
    embedding = finception_resnet_v2.get_symbol(512,
        version_se=args.version_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit)
  elif args.network[0]=='x':
    print('init xception', args.num_layers)
    embedding = fxception.get_symbol(512,
        version_se=args.version_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit)
  elif args.network[0]=='p':
    print('init dpn', args.num_layers)
    embedding = fdpn.get_symbol(512, args.num_layers,
        version_se=args.version_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit)
  else:
    print('init resnet', args.num_layers)
    embedding = fresnet.get_symbol(512, args.num_layers, 
        version_se=args.version_se, version_input=args.version_input, 
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
                          beta_min=args.beta_min, verbose=1000, name='fc7')
  elif args.loss_type==2:
    _weight = mx.symbol.Variable('fc7_weight')
    _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
    fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=args.num_classes, name='fc7')
    print('center-loss', args.center_alpha, args.center_scale)
    extra_loss = mx.symbol.Custom(data=embedding, label=gt_label, name='center_loss', op_type='centerloss',\
          num_class=args.num_classes, alpha=args.center_alpha, scale=args.center_scale, batchsize=args.per_batch_size)
  elif args.loss_type==10: #marginal loss
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
    params = [1.2, 0.3, 1.0]
    n1 = mx.sym.expand_dims(nembedding, axis=1) #N,1,C
    n2 = mx.sym.expand_dims(nembedding, axis=0) #1,N,C
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
  elif args.loss_type==11: #npair loss
    params = [0.9, 0.2]
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
    nembedding = mx.sym.transpose(nembedding)
    nembedding = mx.symbol.reshape(nembedding, (512, args.per_identities, args.images_per_identity))
    nembedding = mx.sym.transpose(nembedding, axes=(2,1,0)) #2*id*512
    #nembedding = mx.symbol.reshape(nembedding, (512, args.images_per_identity, args.per_identities))
    #nembedding = mx.sym.transpose(nembedding, axes=(1,2,0)) #2*id*512
    n1 = mx.symbol.slice_axis(nembedding, axis=0, begin=0, end=1)
    n2 = mx.symbol.slice_axis(nembedding, axis=0, begin=1, end=2)
    #n1 = []
    #n2 = []
    #for i in xrange(args.per_identities):
    #  _n1 = mx.symbol.slice_axis(nembedding, axis=0, begin=2*i, end=2*i+1)
    #  _n2 = mx.symbol.slice_axis(nembedding, axis=0, begin=2*i+1, end=2*i+2)
    #  n1.append(_n1)
    #  n2.append(_n2)
    #n1 = mx.sym.concat(*n1, dim=0)
    #n2 = mx.sym.concat(*n2, dim=0)
    #rembeddings = mx.symbol.reshape(nembedding, (args.images_per_identity, args.per_identities, 512))
    #n1 = mx.symbol.slice_axis(rembeddings, axis=0, begin=0, end=1)
    #n2 = mx.symbol.slice_axis(rembeddings, axis=0, begin=1, end=2)
    n1 = mx.symbol.reshape(n1, (args.per_identities, 512))
    n2 = mx.symbol.reshape(n2, (args.per_identities, 512))
    cosine_matrix = mx.symbol.dot(lhs=n1, rhs=n2, transpose_b = True) #id*id, id=N of N-pair
    data_extra = mx.sym.Variable('extra')
    data_extra = mx.sym.slice_axis(data_extra, axis=0, begin=0, end=args.per_identities)
    mask = cosine_matrix * data_extra
    #body = mx.sym.mean(mask)
    fii = mx.sym.sum_axis(mask, axis=1)
    fij_fii = mx.sym.broadcast_sub(cosine_matrix, fii)
    fij_fii = mx.sym.exp(fij_fii)
    row = mx.sym.sum_axis(fij_fii, axis=1)
    row = mx.sym.log(row)
    body = mx.sym.mean(row)
    extra_loss = mx.sym.MakeLoss(body)
  elif args.loss_type==12: #triplet loss
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
    anchor = mx.symbol.slice_axis(nembedding, axis=0, begin=0, end=args.per_batch_size//3)
    positive = mx.symbol.slice_axis(nembedding, axis=0, begin=args.per_batch_size//3, end=2*args.per_batch_size//3)
    negative = mx.symbol.slice_axis(nembedding, axis=0, begin=2*args.per_batch_size//3, end=args.per_batch_size)
    ap = anchor - positive
    an = anchor - negative
    ap = ap*ap
    an = an*an
    ap = mx.symbol.sum(ap, axis=1, keepdims=1) #(T,1)
    an = mx.symbol.sum(an, axis=1, keepdims=1) #(T,1)
    triplet_loss = mx.symbol.Activation(data = (ap-an+args.triplet_alpha), act_type='relu')
    triplet_loss = mx.symbol.mean(triplet_loss)
    #triplet_loss = mx.symbol.sum(triplet_loss)/(args.per_batch_size//3)
    extra_loss = mx.symbol.MakeLoss(triplet_loss)
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
  out_list = [mx.symbol.BlockGrad(embedding)]
  softmax = None
  if args.loss_type<10:
    softmax = mx.symbol.SoftmaxOutput(data=fc7, label = gt_label, name='softmax', normalization='valid')
    out_list.append(softmax)
  if softmax is None:
    out_list.append(mx.sym.BlockGrad(gt_label))
  if extra_loss is not None:
    out_list.append(extra_loss)
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
      if args.loss_type==10:
        args.per_batch_size = 256
    args.batch_size = args.per_batch_size*args.ctx_num
    args.rescale_threshold = 0
    args.image_channel = 3
    ppatch = [int(x) for x in args.patch.split('_')]
    assert len(ppatch)==5


    os.environ['BETA'] = str(args.beta)
    args.use_val = False
    path_imgrec = None
    path_imglist = None
    val_rec = None
    prop = face_image.load_property(args.data_dir)
    args.num_classes = prop.num_classes
    image_size = prop.image_size
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    print('image_size', image_size)

    assert(args.num_classes>0)
    print('num_classes', args.num_classes)

    #path_imglist = "/raid5data/dplearn/MS-Celeb-Aligned/lst2"
    path_imgrec = os.path.join(args.data_dir, "train.rec")
    val_rec = os.path.join(args.data_dir, "val.rec")
    if os.path.exists(val_rec) and args.loss_type<10:
      args.use_val = True
    else:
      val_rec = None
    args.use_val = False
    #args.num_classes = 10572 #webface
    #args.num_classes = 81017
    #args.num_classes = 82395



    if args.loss_type==1 and args.num_classes>40000:
      args.beta_freeze = 5000
      args.gamma = 0.06

    if args.loss_type==11:
      args.images_per_identity = 2
    elif args.loss_type==10:
      args.images_per_identity = 16
    elif args.loss_type==12:
      args.images_per_identity = 5

    if args.loss_type<10:
      assert args.images_per_identity==0
    else:
      assert args.images_per_identity>=2
      args.per_identities = int(args.per_batch_size/args.images_per_identity)

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

    data_extra = None
    hard_mining = False
    triplet_params = None
    if args.loss_type==10:
      hard_mining = True
      _shape = (args.batch_size, args.per_batch_size)
      data_extra = np.full(_shape, -1.0, dtype=np.float32)
      c = 0
      while c<args.batch_size:
        a = 0
        while a<args.per_batch_size:
          b = a+args.images_per_identity
          data_extra[(c+a):(c+b),a:b] = 1.0
          #print(c+a, c+b, a, b)
          a = b
        c += args.per_batch_size
    elif args.loss_type==11:
      data_extra = np.zeros( (args.batch_size, args.per_identities), dtype=np.float32)
      c = 0
      while c<args.batch_size:
        for i in xrange(args.per_identities):
          data_extra[c+i][i] = 1.0
        c+=args.per_batch_size
    elif args.loss_type==12:
      triplet_params = [args.triplet_bag_size, args.triplet_alpha]

    label_name = 'softmax_label'
    if data_extra is None:
      model = mx.mod.Module(
          context       = ctx,
          symbol        = sym,
      )
    else:
      data_names = ('data', 'extra')
      #label_name = ''
      model = mx.mod.Module(
          context       = ctx,
          symbol        = sym,
          data_names    = data_names,
          label_names   = (label_name,),
      )


    train_dataiter = FaceImageIter(
        batch_size           = args.batch_size,
        data_shape           = data_shape,
        path_imgrec          = path_imgrec,
        shuffle              = True,
        rand_mirror          = True,
        mean                 = mean,
        ctx_num              = args.ctx_num,
        images_per_identity  = args.images_per_identity,
        data_extra           = data_extra,
        hard_mining          = hard_mining,
        triplet_params       = triplet_params,
        mx_model             = model,
        label_name           = label_name,
    )

    if args.loss_type<10:
      _metric = AccMetric()
    else:
      _metric = LossValueMetric()
    eval_metrics = [mx.metric.create(_metric)]

    if args.network[0]=='r':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    elif args.network[0]=='i' or args.network[0]=='x':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2) #inception
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    _rescale = 1.0/args.ctx_num
    opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    som = 20
    if args.loss_type==12:
      som = 2
    _cb = mx.callback.Speedometer(args.batch_size, som)

    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
      path = os.path.join(args.data_dir,name+".bin")
      if os.path.exists(path):
        data_set = verification.load_bin(path, image_size)
        ver_list.append(data_set)
        ver_name_list.append(name)
        print('ver', name)



    def ver_test(nbatch):
      results = []
      for i in xrange(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, args.batch_size, data_extra)
        print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
        #print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
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


    highest_acc = [0.0]
    #for i in xrange(len(ver_list)):
    #  highest_acc.append(0.0)
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
        if acc_list[-1]>=highest_acc[0]:
          highest_acc[0] = acc_list[-1]
          if lfw_score>=0.99:
            do_save = True
        #for i in xrange(len(acc_list)):
        #  acc = acc_list[i]
        #  if acc>=highest_acc[i]:
        #    highest_acc[i] = acc
        #    if lfw_score>=0.99:
        #      do_save = True
        #if args.loss_type==1 and mbatch>lr_steps[-1] and mbatch%10000==0:
        #  do_save = True
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
        print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[0]))
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

