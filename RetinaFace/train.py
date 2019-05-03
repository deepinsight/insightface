from __future__ import print_function
import sys
import argparse
import os
import pprint
import re
import mxnet as mx
import numpy as np
from mxnet.module import Module
import mxnet.optimizer as optimizer

from rcnn.logger import logger
from rcnn.config import config, default, generate_config
from rcnn.symbol import *
from rcnn.core import callback, metric
from rcnn.core.loader import CropLoader, CropLoader2
from rcnn.core.module import MutableModule
from rcnn.utils.load_data import load_gt_roidb, merge_roidb, filter_roidb
from rcnn.utils.load_model import load_param


def get_fixed_params(symbol, fixed_param):
    if not config.LAYER_FIX:
      return []
    fixed_param_names = []
    #for name in symbol.list_arguments():
    #  for f in fixed_param:
    #    if re.match(f, name):
    #      fixed_param_names.append(name)
    #pre = 'mobilenetv20_features_linearbottleneck'
    idx = 0
    for name in symbol.list_arguments():
      #print(idx, name)
      if idx<7 and name!='data':
        fixed_param_names.append(name)
      #elif name.startswith('stage1_'):
      #  fixed_param_names.append(name)
      if name.find('upsampling')>=0:
        fixed_param_names.append(name)

      idx+=1
    return fixed_param_names

def train_net(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch,
              lr=0.001, lr_step='5'):
    # setup config
    #init_config()
    #print(config)
    # setup multi-gpu

    input_batch_size = config.TRAIN.BATCH_IMAGES * len(ctx)

    # print config
    logger.info(pprint.pformat(config))

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in args.image_set.split('+')]
    roidbs = [load_gt_roidb(args.dataset, image_set, args.root_path, args.dataset_path,
                            flip=not args.no_flip)
              for image_set in image_sets]
    #roidb = merge_roidb(roidbs)
    #roidb = filter_roidb(roidb)
    roidb = roidbs[0]

    # load symbol
    #sym = eval('get_' + args.network + '_train')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    #feat_sym = sym.get_internals()['rpn_cls_score_output']
    #train_data = AnchorLoader(feat_sym, roidb, batch_size=input_batch_size, shuffle=not args.no_shuffle,
    #                          ctx=ctx, work_load_list=args.work_load_list,
    #                          feat_stride=config.RPN_FEAT_STRIDE, anchor_scales=config.ANCHOR_SCALES,
    #                          anchor_ratios=config.ANCHOR_RATIOS, aspect_grouping=config.TRAIN.ASPECT_GROUPING)

    # load and initialize params
    sym = None
    if len(pretrained)==0:
        arg_params = {}
        aux_params = {}
    else:
        logger.info('loading %s,%d'%(pretrained, epoch))
        sym, arg_params, aux_params = mx.model.load_checkpoint(pretrained, epoch)
        #arg_params, aux_params = load_param(pretrained, epoch, convert=True)
        #for k in ['rpn_conv_3x3', 'rpn_cls_score', 'rpn_bbox_pred', 'cls_score', 'bbox_pred']:
        #  _k = k+"_weight"
        #  if _k in arg_shape_dict:
        #    v = 0.001 if _k.startswith('bbox_') else 0.01
        #    arg_params[_k] = mx.random.normal(0, v, shape=arg_shape_dict[_k])
        #    print('init %s with normal %.5f'%(_k,v))
        #  _k = k+"_bias"
        #  if _k in arg_shape_dict:
        #    arg_params[_k] = mx.nd.zeros(shape=arg_shape_dict[_k])
        #    print('init %s with zero'%(_k))

    sym = eval('get_' + args.network + '_train')(sym)
    #print(sym.get_internals())
    feat_sym = []
    for stride in config.RPN_FEAT_STRIDE:
        feat_sym.append(sym.get_internals()['face_rpn_cls_score_stride%s_output' % stride])



    train_data = CropLoader(feat_sym, roidb, batch_size=input_batch_size, shuffle=not args.no_shuffle,
                                  ctx=ctx, work_load_list=args.work_load_list)


    # infer max shape
    max_data_shape = [('data', (1, 3, max([v[1] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    #max_data_shape = [('data', (1, 3, max([v[1] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
    max_data_shape.append(('gt_boxes', (1, roidb[0]['max_num_boxes'], 5)))
    logger.info('providing maximum shape %s %s' % (max_data_shape, max_label_shape))

    # infer shape
    data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
    arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    out_shape_dict = dict(zip(sym.list_outputs(), out_shape))
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))
    logger.info('output shape %s' % pprint.pformat(out_shape_dict))


    for k,v in arg_shape_dict.iteritems():
      if k.find('upsampling')>=0:
        print('initializing upsampling_weight', k)
        arg_params[k] = mx.nd.zeros(shape=v)
        init = mx.init.Initializer()
        init._init_bilinear(k, arg_params[k])
            #print(args[k])

    # check parameter shapes
    #for k in sym.list_arguments():
    #    if k in data_shape_dict:
    #        continue
    #    assert k in arg_params, k + ' not initialized'
    #    assert arg_params[k].shape == arg_shape_dict[k], \
    #        'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
    #for k in sym.list_auxiliary_states():
    #    assert k in aux_params, k + ' not initialized'
    #    assert aux_params[k].shape == aux_shape_dict[k], \
    #        'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

    fixed_param_prefix = config.FIXED_PARAMS
    # create solver
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
    fixed_param_names = get_fixed_params(sym, fixed_param_prefix)
    print('fixed', fixed_param_names, file=sys.stderr)
    mod = Module(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, work_load_list=args.work_load_list,
                        fixed_param_names=fixed_param_names)

    # metric
    eval_metrics = mx.metric.CompositeEvalMetric()
    mid=0
    for m in range(len(config.RPN_FEAT_STRIDE)):
      stride = config.RPN_FEAT_STRIDE[m]
      #mid = m*MSTEP
      _metric = metric.RPNAccMetric(pred_idx=mid, label_idx=mid+1, name='RPNAcc_s%s'%stride)
      eval_metrics.add(_metric)
      mid+=2
      #_metric = metric.RPNLogLossMetric(pred_idx=mid, label_idx=mid+1)
      #eval_metrics.add(_metric)

      _metric = metric.RPNL1LossMetric(loss_idx=mid, weight_idx=mid+1, name='RPNL1Loss_s%s'%stride)
      eval_metrics.add(_metric)
      mid+=2
      if config.FACE_LANDMARK:
        _metric = metric.RPNL1LossMetric(loss_idx=mid, weight_idx=mid+1, name='RPNLandMarkL1Loss_s%s'%stride)
        eval_metrics.add(_metric)
        mid+=2
      if config.HEAD_BOX:
        _metric = metric.RPNAccMetric(pred_idx=mid, label_idx=mid+1, name='RPNAcc_head_s%s'%stride)
        eval_metrics.add(_metric)
        mid+=2
        #_metric = metric.RPNLogLossMetric(pred_idx=mid, label_idx=mid+1)
        #eval_metrics.add(_metric)

        _metric = metric.RPNL1LossMetric(loss_idx=mid, weight_idx=mid+1, name='RPNL1Loss_head_s%s'%stride)
        eval_metrics.add(_metric)
        mid+=2

    # callback
    #means = np.tile(np.array(config.TRAIN.BBOX_MEANS), config.NUM_CLASSES)
    #stds = np.tile(np.array(config.TRAIN.BBOX_STDS), config.NUM_CLASSES)
    #epoch_end_callback = callback.do_checkpoint(prefix, means, stds)
    epoch_end_callback = None
    # decide learning rate
    #base_lr = lr
    #lr_factor = 0.1
    #lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))

    lr_epoch = [int(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr_iters = [int(epoch * len(roidb) / input_batch_size) for epoch in lr_epoch_diff]

    lr_steps = []
    if len(lr_iters)==5:
      factors = [0.5, 0.5, 0.4, 0.1, 0.1]
      for i in range(5):
        lr_steps.append( (lr_iters[i], factors[i]) )
    elif len(lr_iters)==8: #warmup
      for li in lr_iters[0:5]:
        lr_steps.append( (li, 1.5849) )
      for li in lr_iters[5:]:
        lr_steps.append( (li, 0.1) )
    else:
      for li in lr_iters:
        lr_steps.append( (li, 0.1) )
    #lr_steps = [ (20,0.1), (40, 0.1) ] #XXX

    end_epoch = 10000
    logger.info('lr %f lr_epoch_diff %s lr_steps %s' % (lr, lr_epoch_diff, lr_steps))
    # optimizer
    opt = optimizer.SGD(learning_rate=lr, momentum=0.9, wd=0.0005, rescale_grad=1.0/len(ctx), clip_gradient=None)
    initializer=mx.init.Xavier()
    #initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style

    train_data = mx.io.PrefetchingIter(train_data)

    _cb = mx.callback.Speedometer(train_data.batch_size, frequent=args.frequent, auto_reset=False)
    global_step = [0]

    def save_model(epoch):
      arg, aux = mod.get_params()
      all_layers = mod.symbol.get_internals()
      outs = []
      for stride in config.RPN_FEAT_STRIDE:
        num_anchors = config.RPN_ANCHOR_CFG[str(stride)]['NUM_ANCHORS']
        _name = 'face_rpn_cls_score_stride%d_output' % stride
        rpn_cls_score = all_layers[_name]


        # prepare rpn data
        rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                  shape=(0, 2, -1, 0),
                                                  name="face_rpn_cls_score_reshape_stride%d" % stride)

        rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape,
                                                   mode="channel",
                                                   name="face_rpn_cls_prob_stride%d" % stride)
        rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob,
                                                 shape=(0, 2 * num_anchors, -1, 0),
                                                 name='face_rpn_cls_prob_reshape_stride%d' % stride)
        _name = 'face_rpn_bbox_pred_stride%d_output' % stride
        rpn_bbox_pred = all_layers[_name]
        outs.append(rpn_cls_prob_reshape)
        outs.append(rpn_bbox_pred)
        if config.FACE_LANDMARK:
          _name = 'face_rpn_landmark_pred_stride%d_output' % stride
          rpn_landmark_pred = all_layers[_name]
          outs.append(rpn_landmark_pred)
      _sym = mx.sym.Group(outs)
      mx.model.save_checkpoint(prefix, epoch, _sym, arg, aux)

    def _batch_callback(param):
      #global global_step
      _cb(param)
      global_step[0]+=1
      mbatch = global_step[0]
      for step in lr_steps:
        if mbatch==step[0]:
          opt.lr *= step[1]
          print('lr change to', opt.lr,' in batch', mbatch, file=sys.stderr)
          break

      if mbatch==lr_steps[-1][0]:
        print('saving final checkpoint', mbatch, file=sys.stderr)
        save_model(0)
        #arg, aux = mod.get_params()
        #mx.model.save_checkpoint(prefix, 99, mod.symbol, arg, aux)
        sys.exit(0)

    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=_batch_callback, kvstore=args.kvstore,
            optimizer=opt,
            initializer = initializer,
            allow_missing=True,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)


def parse_args():
    parser = argparse.ArgumentParser(description='Train RetinaFace')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    parser.add_argument('--image_set', help='image_set name', default=default.image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    # training
    parser.add_argument('--frequent', help='frequency of logging', default=default.frequent, type=int)
    parser.add_argument('--kvstore', help='the kv-store type', default=default.kvstore, type=str)
    parser.add_argument('--work_load_list', help='work load for different devices', default=None, type=list)
    parser.add_argument('--no_flip', help='disable flip images', action='store_true')
    parser.add_argument('--no_shuffle', help='disable random shuffle', action='store_true')
    # e2e
    #parser.add_argument('--gpus', help='GPU device to train with', default='0,1,2,3', type=str)
    parser.add_argument('--pretrained', help='pretrained model prefix', default=default.pretrained, type=str)
    parser.add_argument('--pretrained_epoch', help='pretrained model epoch', default=default.pretrained_epoch, type=int)
    parser.add_argument('--prefix', help='new model prefix', default=default.prefix, type=str)
    parser.add_argument('--begin_epoch', help='begin epoch of training, use with resume', default=0, type=int)
    parser.add_argument('--end_epoch', help='end epoch of training', default=default.end_epoch, type=int)
    parser.add_argument('--lr', help='base learning rate', default=default.lr, type=float)
    parser.add_argument('--lr_step', help='learning rate steps (in epoch)', default=default.lr_step, type=str)
    parser.add_argument('--no_ohem', help='disable online hard mining', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info('Called with argument: %s' % args)
    #ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
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
    train_net(args, ctx, args.pretrained, args.pretrained_epoch, args.prefix, args.begin_epoch, args.end_epoch,
              lr=args.lr, lr_step=args.lr_step)

if __name__ == '__main__':
    main()
