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

import time
import os
import sys
import numpy as np
import logging

import paddle
from visualdl import LogWriter

from utils.logging import AverageMeter, init_logging, CallBackLogging
from datasets import CommonDataset, SyntheticDataset
from utils import losses

from .utils.verification import CallBackVerification
from .utils.io import Checkpoint
from .utils.amp import LSCGradScaler

from . import classifiers
from . import backbones

RELATED_FLAGS_SETTING = {
    'FLAGS_cudnn_exhaustive_search': 1,
    'FLAGS_cudnn_batchnorm_spatial_persistent': 1,
    'FLAGS_max_inplace_grad_add': 8,
    'FLAGS_fraction_of_gpu_memory_to_use': 0.9999,
}
paddle.fluid.set_flags(RELATED_FLAGS_SETTING)


def train(args):
    writer = LogWriter(logdir=args.logdir)

    rank = int(os.getenv("PADDLE_TRAINER_ID", 0))
    world_size = int(os.getenv("PADDLE_TRAINERS_NUM", 1))

    gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
    place = paddle.CUDAPlace(gpu_id)

    if world_size > 1:
        import paddle.distributed.fleet as fleet
        from .utils.data_parallel import sync_gradients, sync_params

        strategy = fleet.DistributedStrategy()
        strategy.without_graph_optimization = True
        fleet.init(is_collective=True, strategy=strategy)

    if args.use_synthetic_dataset:
        trainset = SyntheticDataset(args.num_classes, fp16=args.fp16)
    else:
        trainset = CommonDataset(
            root_dir=args.data_dir,
            label_file=args.label_file,
            fp16=args.fp16,
            is_bin=args.is_bin)

    num_image = len(trainset)
    total_batch_size = args.batch_size * world_size
    steps_per_epoch = num_image // total_batch_size
    if args.train_unit == 'epoch':
        warmup_steps = steps_per_epoch * args.warmup_num
        total_steps = steps_per_epoch * args.train_num
        decay_steps = [x * steps_per_epoch for x in args.decay_boundaries]
        total_epoch = args.train_num
    else:
        warmup_steps = args.warmup_num
        total_steps = args.train_num
        decay_steps = [x for x in args.decay_boundaries]
        total_epoch = (total_steps + steps_per_epoch - 1) // steps_per_epoch

    if rank == 0:
        logging.info('world_size: {}'.format(world_size))
        logging.info('total_batch_size: {}'.format(total_batch_size))
        logging.info('warmup_steps: {}'.format(warmup_steps))
        logging.info('steps_per_epoch: {}'.format(steps_per_epoch))
        logging.info('total_steps: {}'.format(total_steps))
        logging.info('total_epoch: {}'.format(total_epoch))
        logging.info('decay_steps: {}'.format(decay_steps))

    base_lr = total_batch_size * args.lr / 512
    lr_scheduler = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=decay_steps,
        values=[
            base_lr * (args.lr_decay**i) for i in range(len(decay_steps) + 1)
        ])
    if warmup_steps > 0:
        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            lr_scheduler, warmup_steps, 0, base_lr)

    if args.fp16:
        paddle.set_default_dtype("float16")

    margin_loss_params = eval("losses.{}".format(args.loss))()
    backbone = eval("backbones.{}".format(args.backbone))(
        num_features=args.embedding_size, dropout=args.dropout)
    classifier = eval("classifiers.{}".format(args.classifier))(
        rank=rank,
        world_size=world_size,
        num_classes=args.num_classes,
        margin1=margin_loss_params.margin1,
        margin2=margin_loss_params.margin2,
        margin3=margin_loss_params.margin3,
        scale=margin_loss_params.scale,
        sample_ratio=args.sample_ratio,
        embedding_size=args.embedding_size,
        fp16=args.fp16)

    backbone.train()
    classifier.train()

    optimizer = paddle.optimizer.Momentum(
        parameters=[{
            'params': backbone.parameters(),
        }, {
            'params': classifier.parameters(),
        }],
        learning_rate=lr_scheduler,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    if args.fp16:
        optimizer._dtype = 'float32'

    if world_size > 1:
        # sync backbone params for data parallel
        sync_params(backbone.parameters())

    if args.do_validation_while_train:
        callback_verification = CallBackVerification(
            args.validation_interval_step,
            rank,
            args.batch_size,
            args.val_targets,
            args.data_dir,
            fp16=args.fp16, )

    callback_logging = CallBackLogging(args.log_interval_step, rank,
                                       world_size, total_steps,
                                       args.batch_size, writer)

    checkpoint = Checkpoint(
        rank=rank,
        world_size=world_size,
        embedding_size=args.embedding_size,
        num_classes=args.num_classes,
        model_save_dir=os.path.join(args.output, args.backbone),
        checkpoint_dir=args.checkpoint_dir,
        max_num_last_checkpoint=args.max_num_last_checkpoint)

    start_epoch = 0
    global_step = 0
    loss_avg = AverageMeter()
    if args.resume:
        extra_info = checkpoint.load(
            backbone, classifier, optimizer, for_train=True)
        start_epoch = extra_info['epoch'] + 1
        lr_state = extra_info['lr_state']
        # there last_epoch means last_step in for PiecewiseDecay
        # since we always use step style for lr_scheduler
        global_step = lr_state['last_epoch']
        lr_scheduler.set_state_dict(lr_state)

    train_loader = paddle.io.DataLoader(
        trainset,
        places=place,
        num_workers=args.num_workers,
        batch_sampler=paddle.io.DistributedBatchSampler(
            dataset=trainset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True))

    scaler = LSCGradScaler(
        enable=args.fp16,
        init_loss_scaling=args.init_loss_scaling,
        incr_ratio=args.incr_ratio,
        decr_ratio=args.decr_ratio,
        incr_every_n_steps=args.incr_every_n_steps,
        decr_every_n_nan_or_inf=args.decr_every_n_nan_or_inf,
        use_dynamic_loss_scaling=args.use_dynamic_loss_scaling)

    for epoch in range(start_epoch, total_epoch):
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        for step, (img, label) in enumerate(train_loader):
            train_reader_cost += time.time() - reader_start
            global_step += 1
            train_start = time.time()
            with paddle.amp.auto_cast(enable=args.fp16):
                features = backbone(img)
                loss_v = classifier(features, label)

            scaler.scale(loss_v).backward()
            if world_size > 1:
                # data parallel sync backbone gradients
                sync_gradients(backbone.parameters())

            scaler.step(optimizer)
            classifier.step(optimizer)
            optimizer.clear_grad()
            classifier.clear_grad()

            train_run_cost += time.time() - train_start
            total_samples += len(img)

            lr_value = optimizer.get_lr()
            loss_avg.update(loss_v.item(), 1)
            callback_logging(
                global_step,
                loss_avg,
                epoch,
                lr_value,
                avg_reader_cost=train_reader_cost / args.log_interval_step,
                avg_batch_cost=(train_reader_cost + train_run_cost) / args.log_interval_step,
                avg_samples=total_samples / args.log_interval_step,
                ips=total_samples / (train_reader_cost + train_run_cost))
           
            if args.do_validation_while_train:
                callback_verification(global_step, backbone)
            lr_scheduler.step()

            if global_step >= total_steps:
                break
            sys.stdout.flush()
            if rank is 0 and global_step > 0 and global_step % args.log_interval_step == 0:
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            reader_start = time.time()
        checkpoint.save(
            backbone, classifier, optimizer, epoch=epoch, for_train=True)
    writer.close()
