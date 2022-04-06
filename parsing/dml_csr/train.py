#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author     :   Qingping Zheng
@Contact    :   qingpingzheng2014@gmail.com
@File       :   train.py
@Time       :   10/01/21 00:00 PM
@Desc       :   
@License    :   Licensed under the Apache License, Version 2.0 (the "License"); 
@Copyright  :   Copyright 2015 The Authors. All Rights Reserved.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path as osp
import timeit
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import utils.schp as schp

from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from inplace_abn import InPlaceABNSync
from inplace_abn import InPlaceABN

from dataset import datasets
from networks import dml_csr
from utils.logging import get_root_logger
from utils.utils import decode_parsing, inv_preprocess, SingleGPU
from utils.encoding import DataParallelModel, DataParallelCriterion 
from utils.miou import compute_mean_ioU
from utils.warmup_scheduler import SGDRScheduler
from loss.criterion import Criterion
from test import valid

torch.multiprocessing.set_start_method("spawn", force=True)

RESTORE_FROM = 'resnet101-imagenet.pth'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Training Network")
    # Data Preference
    parser.add_argument("--data-dir", type=str, default='./datasets',
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--train-dataset", type=str, default='train', choices=['train', 'valid', 'test'],
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--valid-dataset", type=str, default='test', choices=['valid', 'test_resize', 'test'],
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default='473,473',
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--num-classes", type=int, default=11,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--edge-classes", type=int, default=2,
                        help="Number of classes to predict (including background).") 
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=1234,
                        help="Random seed to have reproducible results.")
    # Model
    parser.add_argument("--restore-from", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot-dir", type=str, default='./snapshots/',
                        help="Where to save snapshots of the model.")
    # Training Strategy
    parser.add_argument("--save-num-images", type=int, default=2,
                        help="How many images to save.")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--eval_epochs", type=int, default=1,
                        help="Number of classes to predict (including background).")                                            
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="choose the number of recurrence.")
    parser.add_argument("--epochs", type=int, default=150,
                        help="choose the number of recurrence.")
    # Distributed Training
    parser.add_argument("--gpu", type=str, default='None',
                        help="choose gpu numbers")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="choose gpu numbers")
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    # self correlation training
    parser.add_argument("--schp-start", type=int, default=100, 
                        help='schp start epoch')
    parser.add_argument("--cycle-epochs", type=int, default=10, 
                        help='schp cyclical epoch')
    parser.add_argument("--schp-restore", type=str, default=None,
                        help="Where restore schp model parameters from.")
    parser.add_argument("--lambda-s", type=float, default=1,   
                        help='segmentation loss weight')
    parser.add_argument("--lambda-e", type=float, default=1, 
                        help='edge loss weight')
    parser.add_argument("--lambda-c", type=float, default=0.1, 
                        help='segmentation-edge consistency loss weight')
    return parser.parse_args()


args = get_arguments()
TIMESTAMP = "{0:%Y_%m_%dT%H_%M_%S/}".format(datetime.now())
global writer

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, total_iters):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, total_iters, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def adjust_learning_rate_pose(optimizer, epoch):
    decay = 0
    if epoch + 1 >= 230:
        decay = 0.05
    elif epoch + 1 >= 200:
        decay = 0.1
    elif epoch + 1 >= 120:
        decay = 0.25
    elif epoch + 1 >= 90:
        decay = 0.5
    else:
        decay = 1

    lr = args.learning_rate * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003


def main():
    """Create the model and start the training."""
    cycle_n = 0
    start_epoch = args.start_epoch
    writer = SummaryWriter(osp.join(args.snapshot_dir, TIMESTAMP)) 
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]
    best_f1 = 0

    torch.cuda.set_device(args.local_rank)

    try:
        world_size = int(os.environ['WORLD_SIZE'])
        distributed = world_size > 1
    except:
        distributed = False
        world_size = 1
    if distributed:
        dist.init_process_group(backend=args.dist_backend, init_method='env://')
    rank = 0 if not distributed else dist.get_rank()

    log_file = args.snapshot_dir + '/' + TIMESTAMP + 'output.log'
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    logger.info(f'Distributed training: {distributed}')

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
 
    if distributed:
        model = dml_csr.DML_CSR(args.num_classes)
        schp_model = dml_csr.DML_CSR(args.num_classes)
    else:
        model = dml_csr.DML_CSR(args.num_classes, InPlaceABN)
        schp_model = dml_csr.DML_CSR(args.num_classes, InPlaceABN)
    
    if args.restore_from is not None:
        print('Resume training from {}'.format(args.restore_from))
        model.load_state_dict(torch.load(args.restore_from), True)
        start_epoch = int(float(args.restore_from.split('.')[0].split('_')[-1])) + 1
    else:
        resnet_params = torch.load(RESTORE_FROM)
        new_params = model.state_dict().copy()
        for i in resnet_params:
            i_parts = i.split('.')
            if not i_parts[0] == 'fc':
                new_params['.'.join(i_parts[0:])] = resnet_params[i]
        model.load_state_dict(new_params)
    model.cuda()

    args.schp_restore = osp.join(args.snapshot_dir, TIMESTAMP, 'best.pth')
    if os.path.exists(args.schp_restore):
        print('Resume schp checkpoint from {}'.format(args.schp_restore))
        schp_model.load_state_dict(torch.load(args.schp_restore), True)
    else:
        schp_resnet_params = torch.load(RESTORE_FROM)
        schp_new_params = schp_model.state_dict().copy()
        for i in schp_resnet_params:
            i_parts = i.split('.')
            if not i_parts[0] == 'fc':
                schp_new_params['.'.join(i_parts[0:])] = schp_resnet_params[i]
        schp_model.load_state_dict(schp_new_params)
    schp_model.cuda()

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        schp_model = torch.nn.parallel.DistributedDataParallel(schp_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    else:
        model = SingleGPU(model)
        schp_model = SingleGPU(schp_model)

    criterion = Criterion(loss_weight=[1, 1, 1, 4, 1], 
                lambda_1=args.lambda_s, lambda_2=args.lambda_e, lambda_3=args.lambda_c, num_classes=args.num_classes)
    criterion.cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    
    train_dataset = FaceDataSet(args.data_dir, args.train_dataset, crop_size=input_size, transform=transform)
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size , shuffle=False, num_workers=2,
                                  pin_memory=True, drop_last=True, sampler=train_sampler)
    
    val_dataset = datasets[str(args.model_type)](args.data_dir, args.valid_dataset, crop_size=input_size, transform=transform)
    num_samples = len(val_dataset)
    valloader = data.DataLoader(val_dataset, batch_size=args.batch_size , shuffle=False, pin_memory=True, drop_last=False)

    # Optimizer Initialization
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                        weight_decay=args.weight_decay)
    lr_scheduler = SGDRScheduler(optimizer, total_epoch=args.epochs,
                                eta_min=args.learning_rate / 100, warmup_epoch=10,
                                start_cyclical=args.schp_start, cyclical_base_lr=args.learning_rate / 2,
                                cyclical_epoch=args.cycle_epochs)

    optimizer.zero_grad()

    total_iters = args.epochs * len(trainloader)
    start = timeit.default_timer()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if distributed:
            train_sampler.set_epoch(epoch)
        for i_iter, batch in enumerate(trainloader):
            i_iter += len(trainloader) * epoch

            if epoch < args.schp_start:
                lr = adjust_learning_rate(optimizer, i_iter, total_iters)
            else:
                lr = lr_scheduler.get_lr()[0]

            images, labels, edges, semantic_edges, _ = batch
            labels = labels.long().cuda(non_blocking=True)
            edges = edges.long().cuda(non_blocking=True)
            semantic_edges = semantic_edges.long().cuda(non_blocking=True)

            preds = model(images)
            
            if cycle_n >= 1:
                with torch.no_grad():
                    soft_preds, soft_edges, soft_semantic_edges = schp_model(images)
            else:
                soft_preds = None
                soft_edges = None
                soft_semantic_edges = None

            loss = criterion(preds, [labels, edges, semantic_edges, soft_preds, soft_edges, soft_semantic_edges], cycle_n)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            with torch.no_grad():
                loss = loss.detach() * labels.shape[0]
                count = labels.new_tensor([labels.shape[0]], dtype=torch.long)
                if dist.is_initialized():
                    dist.all_reduce(count, dist.ReduceOp.SUM)
                    dist.all_reduce(loss, dist.ReduceOp.SUM)
                loss /= count.item()

            if not dist.is_initialized() or dist.get_rank() == 0:
                if i_iter % 50 == 0:
                    writer.add_scalar('learning_rate', lr, i_iter)
                    writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)

                if i_iter % 500 == 0:
                    images_inv = inv_preprocess(images, args.save_num_images)
                    labels_colors = decode_parsing(labels, args.save_num_images, args.num_classes, is_pred=False)
                    edges_colors = decode_parsing(edges, args.save_num_images, 2, is_pred=False)
                    semantic_edges_colors = decode_parsing(semantic_edges, args.save_num_images, args.num_classes, is_pred=False)

                    if isinstance(preds, list):
                        preds = preds[0]
                    preds_colors = decode_parsing(preds[0], args.save_num_images, args.num_classes, is_pred=True)
                    pred_edges = decode_parsing(preds[1], args.save_num_images, 2, is_pred=True)
                    pred_semantic_edges_colors = decode_parsing(preds[2], args.save_num_images, args.num_classes, is_pred=True)

                    img = vutils.make_grid(images_inv, normalize=False, scale_each=True)
                    lab = vutils.make_grid(labels_colors, normalize=False, scale_each=True)
                    pred = vutils.make_grid(preds_colors, normalize=False, scale_each=True)
                    edge = vutils.make_grid(edges_colors, normalize=False, scale_each=True)
                    pred_edge = vutils.make_grid(pred_edges, normalize=False, scale_each=True)
                    pred_semantic_edges = vutils.make_grid(pred_semantic_edges_colors, normalize=False, scale_each=True)


                    writer.add_image('Images/', img, i_iter)
                    writer.add_image('Labels/', lab, i_iter)
                    writer.add_image('Preds/', pred, i_iter)
                    writer.add_image('Edge/', edge, i_iter)
                    writer.add_image('Pred_edge/', pred_edge, i_iter)
    
                cur_loss = loss.data.cpu().numpy()
                logger.info(f'iter = {i_iter} of {total_iters} completed, loss = {cur_loss}, lr = {lr}')

        if (epoch + 1) % (args.eval_epochs) == 0:
            parsing_preds, scales, centers = valid(model, valloader, input_size, num_samples)
            mIoU, f1 = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size, args.valid_dataset, True)

            if not dist.is_initialized() or dist.get_rank() == 0:  
                torch.save(model.module.state_dict(), osp.join(args.snapshot_dir, TIMESTAMP, 'checkpoint_{}.pth'.format(epoch + 1)))
                if 'Helen' in args.data_dir:
                    if f1['overall'] > best_f1:
                        torch.save(model.module.state_dict(), osp.join(args.snapshot_dir, TIMESTAMP, 'best.pth'))
                        best_f1 = f1['overall']
                else:
                    if f1['Mean_F1'] > best_f1:
                        torch.save(model.module.state_dict(), osp.join(args.snapshot_dir, TIMESTAMP, 'best.pth'))
                        best_f1 = f1['Mean_F1']

            writer.add_scalars('mIoU', mIoU, epoch)
            writer.add_scalars('f1', f1, epoch)
            logger.info(f'mIoU = {mIoU}, and f1 = {f1} of epoch = {epoch}, util now, best_f1 = {best_f1}')

            if (epoch + 1) >= args.schp_start and (epoch + 1 - args.schp_start) % args.cycle_epochs == 0:
                logger.info(f'Self-correction cycle number {cycle_n}')
                schp.moving_average(schp_model, model, 1.0 / (cycle_n + 1))
                cycle_n += 1
                schp.bn_re_estimate(trainloader, schp_model)
                parsing_preds, scales, centers = valid(schp_model, valloader, input_size, num_samples)
                mIoU, f1 = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size, args.valid_dataset, True)

                if not dist.is_initialized() or dist.get_rank() == 0: 
                    torch.save(schp_model.module.state_dict(), osp.join(args.snapshot_dir, TIMESTAMP, 'schp_{}_checkpoint.pth'.format(cycle_n)))

                    if 'Helen' in args.data_dir:
                        if f1['overall'] > best_f1:
                            torch.save(schp_model.module.state_dict(), osp.join(args.snapshot_dir, TIMESTAMP, 'best.pth'))
                            best_f1 = f1['overall']
                    else:
                        if f1['Mean_F1'] > best_f1:
                            torch.save(schp_model.module.state_dict(), osp.join(args.snapshot_dir, TIMESTAMP, 'best.pth'))
                            best_f1 = f1['Mean_F1']
                writer.add_scalars('mIoU', mIoU, epoch)
                writer.add_scalars('f1', f1, epoch)
                logger.info(f'mIoU = {mIoU}, and f1 = {f1} of epoch = {epoch}, util now, best_f1 = {best_f1}')

            torch.cuda.empty_cache()
            end = timeit.default_timer()
            print('epoch = {} of {} completed using {} s'.format(epoch, args.epochs,
                                                                (end - start) / (epoch - start_epoch + 1)))

    end = timeit.default_timer()
    print(end - start, 'seconds')
 

if __name__ == '__main__':
    main()
