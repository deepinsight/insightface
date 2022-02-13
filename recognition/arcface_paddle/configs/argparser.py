# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import argparse
import importlib


def print_args(args):
    logging.info('--------args----------')
    for k in list(vars(args).keys()):
        logging.info('%s: %s' % (k, vars(args)[k]))
    logging.info('------------------------\n')


def str2bool(v):
    return str(v).lower() in ("true", "t", "1")


def tostrlist(v):
    if isinstance(v, list):
        return v
    elif isinstance(v, str):
        return [e.strip() for e in v.split(',')]


def tointlist(v):
    if isinstance(v, list):
        return v
    elif isinstance(v, str):
        return [int(e.strip()) for e in v.split(',')]


def get_config(config_file):
    assert config_file.startswith(
        'configs/'), 'config file setting must start with configs/'
    temp_config_name = os.path.basename(config_file)
    temp_module_name = os.path.splitext(temp_config_name)[0]
    config = importlib.import_module("configs.config")
    cfg = config.config
    config = importlib.import_module("configs.%s" % temp_module_name)
    job_cfg = config.config
    cfg.update(job_cfg)
    if cfg.output is None:
        cfg.output = osp.join('work_dirs', temp_module_name)
    return cfg


class UserNamespace(object):
    pass


def parse_args():

    parser = argparse.ArgumentParser(description='Paddle Face Training')
    user_namespace = UserNamespace()
    parser.add_argument(
        '--config_file', type=str, required=True, help='config file path')
    parser.parse_known_args(namespace=user_namespace)
    cfg = get_config(user_namespace.config_file)

    # Model setting
    parser.add_argument(
        '--is_static',
        type=str2bool,
        default=cfg.is_static,
        help='whether to use static mode')
    parser.add_argument(
        '--backbone', type=str, default=cfg.backbone, help='backbone network')
    parser.add_argument(
        '--classifier',
        type=str,
        default=cfg.classifier,
        help='classification network')
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=cfg.embedding_size,
        help='embedding size')
    parser.add_argument(
        '--model_parallel',
        type=str2bool,
        default=cfg.model_parallel,
        help='whether to use model parallel')
    parser.add_argument(
        '--sample_ratio',
        type=float,
        default=cfg.sample_ratio,
        help='sample rate, use partial fc sample if sample rate less than 1.0')
    parser.add_argument(
        '--loss', type=str, default=cfg.loss, help='loss function')
    parser.add_argument(
        '--dropout',
        type=float,
        default=cfg.dropout,
        help='probability of dropout')

    # AMP setting
    parser.add_argument(
        '--fp16',
        type=str2bool,
        default=cfg.fp16,
        help='whether to use fp16 training')
    parser.add_argument(
        '--init_loss_scaling',
        type=float,
        default=cfg.init_loss_scaling,
        help='The initial loss scaling factor.')
    parser.add_argument(
        '--max_loss_scaling',
        type=float,
        default=cfg.max_loss_scaling,
        help='The maximum loss scaling factor.')
    parser.add_argument(
        '--incr_every_n_steps',
        type=int,
        default=cfg.incr_every_n_steps,
        help='Increases loss scaling every n consecutive steps with finite gradients.'
    )
    parser.add_argument(
        '--decr_every_n_nan_or_inf',
        type=int,
        default=cfg.decr_every_n_nan_or_inf,
        help='Decreases loss scaling every n accumulated steps with nan or inf gradients.'
    )
    parser.add_argument(
        '--incr_ratio',
        type=float,
        default=cfg.incr_ratio,
        help='The multiplier to use when increasing the loss scaling.')
    parser.add_argument(
        '--decr_ratio',
        type=float,
        default=cfg.decr_ratio,
        help='The less-than-one-multiplier to use when decreasing the loss scaling.'
    )
    parser.add_argument(
        '--use_dynamic_loss_scaling',
        type=str2bool,
        default=cfg.use_dynamic_loss_scaling,
        help='Whether to use dynamic loss scaling.')
    parser.add_argument(
        '--custom_white_list',
        type=tostrlist,
        default=cfg.custom_white_list,
        help='fp16 custom white list.')
    parser.add_argument(
        '--custom_black_list',
        type=tostrlist,
        default=cfg.custom_black_list,
        help='fp16 custom black list.')

    # Optimizer setting
    parser.add_argument(
        '--lr', type=float, default=cfg.lr, help='learning rate')
    parser.add_argument(
        '--lr_decay',
        type=float,
        default=cfg.lr_decay,
        help='learning rate decay factor')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=cfg.weight_decay,
        help='weight decay')
    parser.add_argument(
        '--momentum', type=float, default=cfg.momentum, help='sgd momentum')
    parser.add_argument(
        '--train_unit',
        type=str,
        default=cfg.train_unit,
        help='train unit, "step" or "epoch"')
    parser.add_argument(
        '--warmup_num',
        type=int,
        default=cfg.warmup_num,
        help='warmup num according train unit')
    parser.add_argument(
        '--train_num',
        type=int,
        default=cfg.train_num,
        help='train num according train unit')
    parser.add_argument(
        '--decay_boundaries',
        type=tointlist,
        default=cfg.decay_boundaries,
        help='piecewise decay boundaries')

    # Train dataset setting
    parser.add_argument(
        '--use_synthetic_dataset',
        type=str2bool,
        default=cfg.use_synthetic_dataset,
        help='whether to use synthetic dataset')
    parser.add_argument(
        '--dataset', type=str, default=cfg.dataset, help='train dataset name')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=cfg.data_dir,
        help='train dataset directory')
    parser.add_argument(
        '--label_file',
        type=str,
        default=cfg.label_file,
        help='train label file name, each line split by "\t"')
    parser.add_argument(
        '--is_bin',
        type=str2bool,
        default=cfg.is_bin,
        help='whether the train data is bin or original image file')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=cfg.num_classes,
        help='classes of train dataset')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=cfg.batch_size,
        help='batch size of each rank')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=cfg.num_workers,
        help='the number workers of DataLoader')

    # Validation dataset setting
    parser.add_argument(
        '--do_validation_while_train',
        type=str2bool,
        default=cfg.do_validation_while_train,
        help='do validation while train')
    parser.add_argument(
        '--validation_interval_step',
        type=int,
        default=cfg.validation_interval_step,
        help='validation interval step')
    parser.add_argument(
        '--val_targets',
        type=tostrlist,
        default=cfg.val_targets,
        help='val targets, list or str split by comma')

    # IO setting
    parser.add_argument(
        '--logdir', type=str, default=cfg.logdir, help='log dir')
    parser.add_argument(
        '--log_interval_step',
        type=int,
        default=cfg.log_interval_step,
        help='log interval step')
    parser.add_argument(
        '--output', type=str, default=cfg.output, help='output dir')
    parser.add_argument(
        '--resume', type=str2bool, default=cfg.resume, help='whether to using resume training')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=cfg.checkpoint_dir,
        help='set checkpoint direcotry when resume training')
    parser.add_argument(
        '--max_num_last_checkpoint',
        type=int,
        default=cfg.max_num_last_checkpoint,
        help='the maximum number of lastest checkpoint to keep')

    args = parser.parse_args(namespace=user_namespace)
    return args
