from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import logging
import numpy as np
from data import FaceIter
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
import mxcommon.resnet_dcn as resnet_dcn
import lfw
import sklearn
from sklearn.decomposition import PCA
from center_loss import *


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
    #print(len(labels), len(preds))
    #print(preds[1].asnumpy())
    loss = preds[2].asnumpy()[0]
    if len(self.losses)==100:
      print('triplet loss', sum(self.losses)/len(self.losses))
      self.losses = []
    self.losses.append(loss)
    preds = [preds[1]] #use softmax output
    for label, pred_label in zip(labels, preds):
        #print(label.shape, pred_label.shape)
        if pred_label.shape != label.shape:
            pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.asnumpy().astype('int32').flatten()
        label = label.asnumpy().astype('int32').flatten()
        #print(label)
        #print(label, pred_label)
        assert label.shape==pred_label.shape
        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)

def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--prefix', default='../model/face',
      help='directory to save model.')
  parser.add_argument('--load-epoch', type=int, default=0,
      help='load epoch.')
  parser.add_argument('--end-epoch', type=int, default=20,
      help='training epoch size.')
  parser.add_argument('--retrain', action='store_true', default=False,
      help='true means continue training.')
  args = parser.parse_args()
  return args

def get_symbol(args, arg_params, aux_params):
  new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
  data_shape = (3,args.image_size,args.image_size)
  image_shape = ",".join([str(x) for x in data_shape])
  layers = 152
  _,_,embeddings,_ = resnet_dcn.get_cls_symbol(128, layers, image_shape, use_deformable=False)
  all_layers = embeddings.get_internals()
  #print(all_layers)
  layer_names = ['_plus10','_plus46', '_plus49']
  #layer_names = ['plus2', '_plus10','_plus46', '_plus49']
  layers = []
  for name in layer_names:
    layers.append( all_layers[name+"_output"] )
  out_sym = mx.symbol.Group(layers)
  out_name = out_sym.list_outputs()
  #arg_name = embeddings.list_arguments()
  #aux_name = embeddings.list_auxiliary_states()
  data_shape_dict = {'data' : (args.batch_size,)+data_shape}
  arg_shape, out_shape, aux_shape = out_sym.infer_shape(**data_shape_dict)
  #print(out_shape)
  out_shape_dict = dict(zip(out_name, out_shape))
  for k,v in out_shape_dict.iteritems():
    print(k,v)

  layers = []
  for i in xrange(len(layer_names)):
    name = layer_names[i]+"_output"
    _layer = all_layers[name]
    _kernel = out_shape_dict[name][2]//5
    if _kernel>1:
      layer = mx.sym.Pooling(data=_layer, kernel=(_kernel, _kernel), stride=(_kernel,_kernel), pad=(0,0), pool_type='max')
    else:
      layer = _layer
    layer = mx.symbol.Convolution(data=layer, kernel=(3, 3), pad=(1,1), num_filter=128)
    layers.append(layer)
  body = mx.symbol.concat(*layers, dim=1)
  body = mx.symbol.Convolution(data=body, kernel=(1, 1), pad=(0,0), num_filter=128)
  body = mx.sym.Pooling(data=body, global_pool=True, kernel=(5, 5), pool_type='avg', name='last_pool')
  embeddings = mx.sym.Flatten(data=body)
  _, out_shape, _= embeddings.infer_shape(**data_shape_dict)
  print(out_shape)
  #print(arg_shape)
  #sys.exit(0)
  l2_embeddings = mx.symbol.L2Normalization(embeddings)
  batch_size = args.batch_size//args.ctx_num
  anchor = mx.symbol.slice_axis(l2_embeddings, axis=0, begin=0, end=batch_size//3)
  positive = mx.symbol.slice_axis(l2_embeddings, axis=0, begin=batch_size//3, end=2*batch_size//3)
  negative = mx.symbol.slice_axis(l2_embeddings, axis=0, begin=2*batch_size//3, end=batch_size)
  ap = anchor - positive
  an = anchor - negative
  ap = ap*ap
  an = an*an
  ap = mx.symbol.sum(ap, axis=1, keepdims=1) #(T,1)
  an = mx.symbol.sum(an, axis=1, keepdims=1) #(T,1)
  loss_scale = [1.0, 0.0, 0.0]
  #triplet_loss = mx.symbol.broadcast_maximum(0.0, ap-an+args.margin) #(T,1)
  triplet_loss = mx.symbol.Activation(data = (ap-an+args.margin), act_type='relu')
  triplet_loss = mx.symbol.sum(triplet_loss)/(batch_size//3)
  triplet_loss = mx.symbol.MakeLoss(data = triplet_loss, grad_scale = loss_scale[0])
  data = mx.symbol.Variable('data')
  gt_label = mx.symbol.Variable('softmax_label')
  fc = mx.symbol.FullyConnected(data = embeddings, num_hidden = args.num_classes, name="fc2")
  softmax = mx.symbol.SoftmaxOutput(data=fc, label = gt_label, name='softmax', grad_scale = loss_scale[1])
  if loss_scale[2]>0.0:
    _center_loss = mx.symbol.Custom(data = l2_embeddings, label = gt_label, name='center_loss', op_type='centerloss'
        , num_class= args.num_classes, alpha = 0.5, scale=loss_scale[2], batchsize=batch_size)
    out = mx.symbol.Group([mx.symbol.BlockGrad(l2_embeddings), softmax, triplet_loss, _center_loss])
  else:
    out = mx.symbol.Group([mx.symbol.BlockGrad(l2_embeddings), softmax, triplet_loss])
  #out = triplet_loss
  #out = softmax
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
    end_epoch = args.end_epoch
    pretrained = '../model/resnet-152'
    load_epoch = args.load_epoch
    args.image_size = 160
    per_batch_size = 60
    args.ctx_num = len(ctx)
    args.batch_size = per_batch_size*args.ctx_num
    #args.all_batch_size = args.batch_size*args.ctx_num
    args.bag_size = 3600
    args.margin = 0.2
    args.num_classes = 10575 #webface


    data_shape = (3,args.image_size,args.image_size)

    begin_epoch = 0
    base_lr = 0.05
    base_wd = 0.0002
    base_mom = 0.0
    lr_decay = 0.98
    if not args.retrain:
      #load and initialize params
      print(pretrained)
      _, arg_params, aux_params = mx.model.load_checkpoint(pretrained, load_epoch)
      sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
      #arg_params, aux_params = load_param(pretrained, epoch, convert=True)
      data_shape_dict = {'data': (args.batch_size, 3, args.image_size, args.image_size), 'softmax_label': (args.batch_size,)}
      resnet_dcn.init_weights(sym, data_shape_dict, arg_params, aux_params)
    else:
      pretrained = args.prefix
      sym, arg_params, aux_params = mx.model.load_checkpoint(pretrained, load_epoch)
      begin_epoch = load_epoch
      end_epoch = begin_epoch+10
      base_wd = 0.00005
      lr_decay = 0.5
      base_lr = 0.015
    # infer max shape

    model = mx.mod.Module(
        context       = ctx,
        symbol        = sym,
        #label_names   = [],
        #fixed_param_prefix = fixed_param_prefix,
    )

    train_dataiter = FaceIter(
        path_imglist         = "/raid5data/dplearn/faceinsight_align_webface.lst",
        data_shape           = data_shape,
        mod                  = model,
        ctx_num              = args.ctx_num,
        batch_size           = args.batch_size,
        bag_size             = args.bag_size,
        images_per_person    = 5,
    )

    #_dice = DiceMetric()
    _acc = AccMetric()
    eval_metrics = [mx.metric.create(_acc)]

    # rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric
    #for child_metric in [fcn_loss_metric]:
    #    eval_metrics.add(child_metric)

    # callback
    #batch_end_callback = callback.Speedometer(input_batch_size, frequent=args.frequent)
    #epoch_end_callback = mx.callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True)

    # decide learning rate
    #lr_step = '10,20,30'
    #train_size = 4848
    #nrof_batch_in_epoch = int(train_size/input_batch_size)
    #print('nrof_batch_in_epoch:', nrof_batch_in_epoch)
    #lr_factor = 0.1
    #lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    #lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    #lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    #lr_iters = [int(epoch * train_size / batch_size) for epoch in lr_epoch_diff]
    #print 'lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters

    #lr_scheduler = MultiFactorScheduler(lr_iters, lr_factor)

    # optimizer
    #optimizer_params = {'momentum': 0.9,
    #                    'wd': 0.0005,
    #                    'learning_rate': base_lr,
    #                    'rescale_grad': 1.0,
    #                    'clip_gradient': None}
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
    #opt = optimizer.SGD(learning_rate=base_lr, momentum=0.9, wd=base_wd, rescale_grad=(1.0/args.batch_size))
    opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=1.0)
    #opt = optimizer.AdaGrad(learning_rate=base_lr, wd=base_wd, rescale_grad=1.0)
    _cb = mx.callback.Speedometer(args.batch_size, 10)

    lfw_dir = '/raid5data/dplearn/lfw_mtcnn'
    lfw_pairs = lfw.read_pairs(os.path.join(lfw_dir, 'pairs.txt'))
    lfw_paths, issame_list = lfw.get_paths(lfw_dir, lfw_pairs, 'png')
    imgs = []
    lfw_data_list = []
    for flip in [0,1]:
      lfw_data = nd.empty((len(lfw_paths), 3, args.image_size, args.image_size))
      i = 0
      for path in lfw_paths:
        with open(path, 'rb') as fin:
          _bin = fin.read()
          img = mx.image.imdecode(_bin)
          img = nd.transpose(img, axes=(2, 0, 1))
          if flip==1:
            img = img.asnumpy()
            for c in xrange(img.shape[0]):
              img[c,:,:] = np.fliplr(img[c,:,:])
            img = nd.array( img )
          #print(img.shape)
          lfw_data[i][:] = img
          i+=1
          if i%1000==0:
            print('loading lfw', i)
      print(lfw_data.shape)
      lfw_data_list.append(lfw_data)

    def lfw_test(nbatch):
      print('testing lfw..')
      embeddings_list = []
      for i in xrange( len(lfw_data_list) ):
        lfw_data = lfw_data_list[i]
        embeddings = None
        ba = 0
        while ba<lfw_data.shape[0]:
          bb = min(ba+args.batch_size, lfw_data.shape[0])
          _data = nd.slice_axis(lfw_data, axis=0, begin=ba, end=bb)
          _label = nd.ones( (bb-ba,) )
          db = mx.io.DataBatch(data=(_data,), label=(_label,))
          model.forward(db, is_train=False)
          net_out = model.get_outputs()
          _embeddings = net_out[0].asnumpy()
          if embeddings is None:
            embeddings = np.zeros( (lfw_data.shape[0], _embeddings.shape[1]) )
          embeddings[ba:bb,:] = _embeddings
          ba = bb
        embeddings_list.append(embeddings)

      acc_list = []
      embeddings = embeddings_list[0]
      _, _, accuracy, val, val_std, far = lfw.evaluate(embeddings, issame_list, nrof_folds=10)
      acc_list.append(np.mean(accuracy))
      print('[%d]Accuracy: %1.3f+-%1.3f' % (nbatch, np.mean(accuracy), np.std(accuracy)))
      print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
      embeddings = np.concatenate(embeddings_list, axis=1)
      embeddings = sklearn.preprocessing.normalize(embeddings)
      print(embeddings.shape)
      _, _, accuracy, val, val_std, far = lfw.evaluate(embeddings, issame_list, nrof_folds=10)
      acc_list.append(np.mean(accuracy))
      print('[%d]Accuracy-Flip: %1.3f+-%1.3f' % (nbatch, np.mean(accuracy), np.std(accuracy)))
      print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
      pca = PCA(n_components=128)
      embeddings = pca.fit_transform(embeddings)
      embeddings = sklearn.preprocessing.normalize(embeddings)
      print(embeddings.shape)
      _, _, accuracy, val, val_std, far = lfw.evaluate(embeddings, issame_list, nrof_folds=10)
      acc_list.append(np.mean(accuracy))
      print('[%d]Accuracy-PCA: %1.3f+-%1.3f' % (nbatch, np.mean(accuracy), np.std(accuracy)))
      print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
      return max(*acc_list)


    #global_step = 0
    highest_acc = [0.0]
    last_save_acc = [0.0]
    def _batch_callback(param):
      #global global_step
      mbatch = param.nbatch+1
      if mbatch % 4000 == 0:
        opt.lr *= lr_decay

      #print(param.nbatch, opt.lr)
      _cb(param)
      if param.nbatch%100==0:
        print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)
      if param.nbatch>=0 and param.nbatch%400==0:
        acc = lfw_test(param.nbatch)
        if acc>highest_acc[0]:
          highest_acc[0] = acc
        if acc>0.9 and acc-last_save_acc[0]>=0.01:
          print('saving', mbatch, acc, last_save_acc[0])
          _arg, _aux = model.get_params()
          mx.model.save_checkpoint(args.prefix, mbatch, model.symbol, _arg, _aux)
          last_save_acc[0] = acc
        print('[%d]highest Accu: %1.3f'%(param.nbatch, highest_acc[0]))




      sys.stdout.flush()
      sys.stderr.flush()

    epoch_cb = mx.callback.do_checkpoint(prefix, 1)
    #epoch_cb = None



    def _epoch_callback(epoch, sym, arg_params, aux_params):
      print('epoch-end', epoch)

    model.fit(train_dataiter,
        begin_epoch        = begin_epoch,
        num_epoch          = end_epoch,
        #eval_data          = val_dataiter,
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
    print('Called with argument:', args)
    train_net(args)

if __name__ == '__main__':
    main()
