import argparse
import logging
import pprint
import mxnet as mx
import numpy as np
import os.path as osp
import cPickle as pkl

from ..config import config, default, generate_config
from ..symbol import *
from ..core import callback, metric
from ..core.loader import MaskROIIter
from ..core.module import MutableModule
from ..processing.bbox_regression import add_bbox_regression_targets, add_mask_targets
from ..processing.assign_levels import add_assign_targets
from ..utils.load_data import load_proposal_roidb, merge_roidb #, filter_roidb
from ..utils.load_model import load_param

def train_maskrcnn(network, dataset, image_set, root_path, dataset_path,
               frequent, kvstore, work_load_list, no_flip, no_shuffle, resume,
               ctx, pretrained, epoch, prefix, begin_epoch, end_epoch,
               train_shared, lr, lr_step, proposal, maskrcnn_stage=None):
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # load symbol
    config.TRAIN.BATCH_IMAGES = 1
    config.TRAIN.BATCH_ROIS = 256
    sym = eval('get_' + network + '_maskrcnn')(num_classes=config.NUM_CLASSES)

    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    # print config
    pprint.pprint(config)

    USE_CACHE = True

    if USE_CACHE:
      roidb_file = root_path + '/cache/' + dataset + '_roidb_with_mask.pkl'
      mean_file = root_path + '/cache/' + dataset + '_roidb_mean.pkl'
      std_file = root_path + '/cache/' + dataset + '_roidb_std.pkl'
      if maskrcnn_stage is not None:
          roidb_file = root_path + '/cache/' + dataset + '_roidb_with_mask_' + maskrcnn_stage + '.pkl'
          mean_file = root_path + '/cache/' + dataset + '_roidb_mean_' + maskrcnn_stage + '.pkl'
          std_file = root_path + '/cache/' + dataset + '_roidb_std_' + maskrcnn_stage + '.pkl'

    if USE_CACHE and osp.exists(roidb_file) and osp.exists(mean_file) and osp.exists(std_file):
        print 'Load ' + roidb_file
        with open(roidb_file, 'r') as f:
            roidb = pkl.load(f)
        print 'Load ' + mean_file
        with open(mean_file, 'r') as f:
            means = pkl.load(f)
        print 'Load ' + std_file
        with open(std_file, 'r') as f:
            stds = pkl.load(f)
    else:
        # load dataset and prepare imdb for training
        image_sets = [iset for iset in image_set.split('+')]
        roidbs = [load_proposal_roidb(dataset, image_set, root_path, dataset_path,
                                      proposal=proposal, append_gt=True, flip=not no_flip)
                  for image_set in image_sets]
        roidb = merge_roidb(roidbs)

        def filter_roidb(roidb):
            """ remove roidb entries without usable rois """

            def is_valid(entry):
                """ valid images have at least 1 fg or bg roi """
                overlaps = entry['max_overlaps']
                fg_inds = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
                bg_inds = np.where((overlaps < config.TRAIN.BG_THRESH_HI) & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
                valid = len(fg_inds) > 0 and len(bg_inds) > 0
                return valid

            num = len(roidb)
            filtered_roidb = [entry for entry in roidb if is_valid(entry)]
            num_after = len(filtered_roidb)
            print 'filtered %d roidb entries: %d -> %d' % (num - num_after, num, num_after)

            return filtered_roidb

        roidb = filter_roidb(roidb)
        means, stds = add_bbox_regression_targets(roidb)
        add_assign_targets(roidb)
        add_mask_targets(roidb)
        if USE_CACHE:
          for file, obj in zip([roidb_file, mean_file, std_file], [roidb, means, stds]):
              with open(file, 'w') as f:
                  pkl.dump(obj, f, -1)

    # load training data
    train_data = MaskROIIter(roidb, batch_size=input_batch_size, shuffle=not no_shuffle,
                             ctx=ctx, work_load_list=work_load_list, aspect_grouping=config.TRAIN.ASPECT_GROUPING)

    # infer max shape
    max_data_shape = [('data', (input_batch_size, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    max_label_shape = []
    for s in config.RCNN_FEAT_STRIDE:
        max_data_shape.append(('rois_stride%s' % s, (input_batch_size, config.TRAIN.BATCH_ROIS, 5)))
        max_label_shape.append(('label_stride%s' % s, (input_batch_size, config.TRAIN.BATCH_ROIS)))
        max_label_shape.append(('bbox_target_stride%s' % s, (input_batch_size, config.TRAIN.BATCH_ROIS*config.NUM_CLASSES*4)))
        max_label_shape.append(('bbox_weight_stride%s' % s, (input_batch_size, config.TRAIN.BATCH_ROIS*config.NUM_CLASSES*4)))
        max_label_shape.append(('mask_target_stride%s' % s, (input_batch_size, config.TRAIN.BATCH_ROIS, config.NUM_CLASSES, 28, 28)))
        max_label_shape.append(('mask_weight_stride%s' % s, (input_batch_size, config.TRAIN.BATCH_ROIS, config.NUM_CLASSES, 1, 1)))
    # infer shape
    data_shape_dict = dict(train_data.provide_data + train_data.provide_label)

    arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    out_shape_dict = zip(sym.list_outputs(), out_shape)
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))
    print 'output shape'
    pprint.pprint(out_shape_dict)

    # load and initialize params
    if resume:
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    else:
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)
        init_bbox_pred = mx.init.Normal(sigma=0.001)
        init_internal = mx.init.Normal(sigma=0.01)
        init = mx.init.Xavier(factor_type="in", rnd_type='gaussian', magnitude=2)
        for k in sym.list_arguments():
            if k in data_shape_dict:
                continue
            if k not in arg_params:
                print 'init', k
                arg_params[k] = mx.nd.zeros(shape=arg_shape_dict[k])
                init_internal(k, arg_params[k])
                if k in ['rcnn_fc_bbox_weight', 'bbox_pred_weight']:
                    init_bbox_pred(k, arg_params[k])
                if k.endswith('bias'):
                    arg_params[k] = mx.nd.zeros(shape=arg_shape_dict[k])
                if 'ctx_red_weight' in k:
                    ctx_shape = np.array(arg_shape_dict[k])
                    ctx_shape[1] /= 2
                    arg_params[k][:] = np.concatenate((np.eye(ctx_shape[1]).reshape(ctx_shape), np.zeros(ctx_shape)), axis=1)

        for k in sym.list_auxiliary_states():
            if k not in aux_params:
                print 'init', k
                aux_params[k] = mx.nd.zeros(shape=aux_shape_dict[k])
                init(k, aux_params[k])

    # check parameter shapes
    for k in sym.list_arguments():
        if k in data_shape_dict:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
    for k in sym.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

    # prepare training
    # create solver
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
    if train_shared:
        fixed_param_prefix = config.FIXED_PARAMS_SHARED
    else:
        fixed_param_prefix = config.FIXED_PARAMS
    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, work_load_list=work_load_list,
                        max_data_shapes=max_data_shape, max_label_shapes=max_label_shape,
                        fixed_param_prefix=fixed_param_prefix)

    # decide training params
    # metric
    eval_metric = metric.RCNNAccMetric()
    cls_metric = metric.RCNNLogLossMetric()
    bbox_metric = metric.RCNNL1LossMetric()
    mask_acc_metric = metric.MaskAccMetric()
    mask_log_metric = metric.MaskLogLossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [eval_metric, cls_metric, bbox_metric, mask_acc_metric, mask_log_metric]:
        eval_metrics.add(child_metric)
    # callback
    batch_end_callback = mx.callback.Speedometer(train_data.batch_size, frequent=frequent)
    epoch_end_callback = callback.do_checkpoint(prefix, means, stds)
    # decide learning rate
    base_lr = lr
    lr_factor = 0.1
    lr_epoch = [int(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    print 'lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)
    # optimizer
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0001,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': (1.0 / batch_size),
                        'clip_gradient': 5}

    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)

