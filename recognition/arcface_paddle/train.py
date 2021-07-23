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

from dataloader import CommonDataset

from paddle.io import DataLoader
from config import config as cfg
from partial_fc import PartialFC
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter
import paddle.nn.functional as F
from paddle.nn import ClipGradByNorm
from visualdl import LogWriter
import paddle
import backbones
import argparse
import losses
import time
import os
import sys


def main(args):
    world_size = int(1.0)
    rank = int(0.0)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    else:
        time.sleep(2)

    writer = LogWriter(logdir=args.logdir)
    trainset = CommonDataset(root_dir=cfg.data_dir, label_file=cfg.file_list, is_bin=args.is_bin)
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0)

    backbone = eval("backbones.{}".format(args.network))()
    backbone.train()

    clip_by_norm = ClipGradByNorm(5.0)
    margin_softmax = eval("losses.{}".format(args.loss))()

    module_partial_fc = PartialFC(
        rank=0,
        world_size=1,
        resume=0,
        batch_size=args.batch_size,
        margin_softmax=margin_softmax,
        num_classes=cfg.num_classes,
        sample_rate=cfg.sample_rate,
        embedding_size=args.embedding_size,
        prefix=args.output)

    scheduler_backbone_decay = paddle.optimizer.lr.LambdaDecay(
        learning_rate=args.lr, lr_lambda=cfg.lr_func, verbose=True)
    scheduler_backbone = paddle.optimizer.lr.LinearWarmup(
        learning_rate=scheduler_backbone_decay,
        warmup_steps=cfg.warmup_epoch,
        start_lr=0,
        end_lr=args.lr / 512 * args.batch_size,
        verbose=True)
    opt_backbone = paddle.optimizer.Momentum(
        parameters=backbone.parameters(),
        learning_rate=scheduler_backbone,
        momentum=0.9,
        weight_decay=args.weight_decay,
        grad_clip=clip_by_norm)

    scheduler_pfc_decay = paddle.optimizer.lr.LambdaDecay(
        learning_rate=args.lr, lr_lambda=cfg.lr_func, verbose=True)
    scheduler_pfc = paddle.optimizer.lr.LinearWarmup(
        learning_rate=scheduler_pfc_decay,
        warmup_steps=cfg.warmup_epoch,
        start_lr=0,
        end_lr=args.lr / 512 * args.batch_size,
        verbose=True)
    opt_pfc = paddle.optimizer.Momentum(
        parameters=module_partial_fc.parameters(),
        learning_rate=scheduler_pfc,
        momentum=0.9,
        weight_decay=args.weight_decay,
        grad_clip=clip_by_norm)

    start_epoch = 0
    total_step = int(
        len(trainset) / args.batch_size / world_size * cfg.num_epoch)
    if rank == 0:
        print("Total Step is: %d" % total_step)

    callback_verification = CallBackVerification(2000, rank, cfg.val_targets,
                                                 cfg.data_dir)
    callback_logging = CallBackLogging(10, rank, total_step, args.batch_size,
                                       world_size, writer)
    callback_checkpoint = CallBackModelCheckpoint(rank, args.output,
                                                  args.network)

    loss = AverageMeter()
    global_step = 0
    for epoch in range(start_epoch, cfg.num_epoch):
        for step, (img, label) in enumerate(train_loader):
            label = label.flatten()
            global_step += 1
            sys.stdout.flush()
            features = F.normalize(backbone(img))
            x_grad, loss_v = module_partial_fc.forward_backward(
                label, features, opt_pfc)
            sys.stdout.flush()
            (features.multiply(x_grad)).backward()
            sys.stdout.flush()
            opt_backbone.step()
            opt_pfc.step()
            module_partial_fc.update()
            opt_backbone.clear_gradients()
            opt_pfc.clear_gradients()
            sys.stdout.flush()

            lr_backbone_value = opt_backbone._global_learning_rate().numpy()[0]
            lr_pfc_value = opt_backbone._global_learning_rate().numpy()[0]

            loss.update(loss_v, 1)
            callback_logging(global_step, loss, epoch, lr_backbone_value,
                             lr_pfc_value)
            sys.stdout.flush()
            callback_verification(global_step, backbone)
        callback_checkpoint(global_step, backbone, module_partial_fc)
        scheduler_backbone.step()
        scheduler_pfc.step()
    writer.close()


if __name__ == '__main__':
    def str2bool(v):
        return v.lower() in ("true", "t", "1")
    
    parser = argparse.ArgumentParser(description='Paddle ArcFace Training')
    parser.add_argument(
        '--network',
        type=str,
        default='MobileFaceNet_128',
        help='backbone network')
    parser.add_argument(
        '--loss', type=str, default='ArcFace', help='loss function')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument(
        '--batch_size', type=int, default=512, help='batch size')
    parser.add_argument(
        '--weight_decay', type=float, default=2e-4, help='weight decay')
    parser.add_argument(
        '--embedding_size', type=int, default=128, help='embedding size')
    parser.add_argument('--logdir', type=str, default='./log', help='log dir')
    parser.add_argument(
        '--output', type=str, default='emore_arcface', help='output dir')
    parser.add_argument('--resume', type=int, default=0, help='model resuming')
    parser.add_argument('--is_bin', type=str2bool, default=True, help='whether the train data is bin or original image file')
    args = parser.parse_args()
    main(args)
