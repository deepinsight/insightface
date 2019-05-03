import argparse
import pprint
import mxnet as mx

from ..logger import logger
from ..config import config, default, generate_config
from ..symbol import *
from ..core import callback, metric
from ..core.loader import ROIIter
from ..core.module import MutableModule
from ..processing.bbox_regression import add_bbox_regression_targets
from ..utils.load_data import load_proposal_roidb, merge_roidb, filter_roidb
from ..utils.load_model import load_param


def train_rcnn(network, dataset, image_set, root_path, dataset_path,
               frequent, kvstore, work_load_list, no_flip, no_shuffle, resume,
               ctx, pretrained, epoch, prefix, begin_epoch, end_epoch,
               train_shared, lr, lr_step, proposal):
    # set up config
    config.TRAIN.BATCH_IMAGES = 2
    config.TRAIN.BATCH_ROIS = 128
    if proposal == 'ss':
        config.TRAIN.BG_THRESH_LO = 0.1  # reproduce Fast R-CNN

    # load symbol
    sym = eval('get_' + network + '_rcnn')(num_classes=config.NUM_CLASSES)

    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    # print config
    logger.info(pprint.pformat(config))

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in image_set.split('+')]
    roidbs = [load_proposal_roidb(dataset, image_set, root_path, dataset_path,
                                  proposal=proposal, append_gt=True, flip=not no_flip)
              for image_set in image_sets]
    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb)
    means, stds = add_bbox_regression_targets(roidb)

    # load training data
    train_data = ROIIter(roidb, batch_size=input_batch_size, shuffle=not no_shuffle,
                         ctx=ctx, work_load_list=work_load_list, aspect_grouping=config.TRAIN.ASPECT_GROUPING)

    # infer max shape
    max_data_shape = [('data', (input_batch_size, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    logger.info('providing maximum shape %s' % max_data_shape)

    # infer shape
    data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
    arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    out_shape_dict = dict(zip(sym.list_outputs(), out_shape))
    aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))
    logger.info('output shape %s' % pprint.pformat(out_shape_dict))

    # load and initialize params
    if resume:
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    else:
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)
        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
        arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['bbox_pred_weight'])
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'])

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
                        max_data_shapes=max_data_shape, fixed_param_prefix=fixed_param_prefix)

    # decide training params
    # metric
    eval_metric = metric.RCNNAccMetric()
    cls_metric = metric.RCNNLogLossMetric()
    bbox_metric = metric.RCNNL1LossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)
    # callback
    batch_end_callback = mx.callback.Speedometer(train_data.batch_size, frequent=frequent, auto_reset=False)
    epoch_end_callback = callback.do_checkpoint(prefix, means, stds)
    # decide learning rate
    base_lr = lr
    lr_factor = 0.1
    lr_epoch = [int(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    logger.info('lr %f lr_epoch_diff %s lr_iters %s' % (lr, lr_epoch_diff, lr_iters))
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)
    # optimizer
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': (1.0 / batch_size),
                        'clip_gradient': 5}

    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN Network')
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
    parser.add_argument('--resume', help='continue training', action='store_true')
    # rcnn
    parser.add_argument('--gpus', help='GPU device to train with', default='0', type=str)
    parser.add_argument('--pretrained', help='pretrained model prefix', default=default.pretrained, type=str)
    parser.add_argument('--pretrained_epoch', help='pretrained model epoch', default=default.pretrained_epoch, type=int)
    parser.add_argument('--prefix', help='new model prefix', default=default.rcnn_prefix, type=str)
    parser.add_argument('--begin_epoch', help='begin epoch of training', default=0, type=int)
    parser.add_argument('--end_epoch', help='end epoch of training', default=default.rcnn_epoch, type=int)
    parser.add_argument('--lr', help='base learning rate', default=default.rcnn_lr, type=float)
    parser.add_argument('--lr_step', help='learning rate steps (in epoch)', default=default.rcnn_lr_step, type=str)
    parser.add_argument('--train_shared', help='second round train shared params', action='store_true')
    parser.add_argument('--proposal', help='can be ss for selective search or rpn', default='rpn', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info('Called with argument: %s' % args)
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    train_rcnn(args.network, args.dataset, args.image_set, args.root_path, args.dataset_path,
               args.frequent, args.kvstore, args.work_load_list, args.no_flip, args.no_shuffle, args.resume,
               ctx, args.pretrained, args.pretrained_epoch, args.prefix, args.begin_epoch, args.end_epoch,
               train_shared=args.train_shared, lr=args.lr, lr_step=args.lr_step, proposal=args.proposal)

if __name__ == '__main__':
    main()
