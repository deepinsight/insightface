import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils import *
from network.lightcnn112 import LightCNN_29Layers_cosface
from losses import IDMMD, CosFace
from dataset_mix import Real_Dataset_112_paired, IdentitySampler, GenIdx

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', default='0,1', type=str)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--epochs', default=15, type=int)
parser.add_argument('--pre_epoch', default=0, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=2e-4)
parser.add_argument('--step_size', default=5, type=int)
parser.add_argument('--print_iter', default=5, type=int)
parser.add_argument('--save_name', default='', type=str)
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--weights_lightcnn', default='', type=str)
parser.add_argument('--dataset', default='CASIA', type=str)

parser.add_argument('--img_root_R', default='', type=str)
parser.add_argument('--train_list_R', default='', type=str)

parser.add_argument('--input_mode', default='red', choices=['grey'], type=str)
parser.add_argument('--model_mode', default='9',choices=['9','29'], type=str)


def main():
    global args
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    cudnn.benchmark = True
    cudnn.enabled = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset = 'lamp' if args.dataset == 'LAMP-HQ' else args.dataset.lower()

    # train loader of real data
    real_dataset_paired = Real_Dataset_112_paired(args)
    vis_pos, nir_pos = GenIdx(real_dataset_paired.vis_labels, real_dataset_paired.nir_labels)
    sampler = IdentitySampler(real_dataset_paired.vis_labels, real_dataset_paired.nir_labels, vis_pos, nir_pos, args.batch_size, 4)
    
    real_dataset_paired.visIndex = sampler.visIndex
    real_dataset_paired.nirIndex = sampler.nirIndex
    
    train_loader_real_paired = torch.utils.data.DataLoader(
        real_dataset_paired, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True)
    
    num_classes = real_dataset_paired.num_classes

    model = LightCNN_29Layers_cosface(num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()

    # load pre trained model
    if args.pre_epoch:
        print('load pretrained model of epoch %d' % args.pre_epoch)
        load_model(model, "./model/lightCNN_epoch_%d.pth.tar" % args.pre_epoch)
    else:
        print("=> loading pretrained lightcnn '{}'".format(args.weights_lightcnn))
        load_model_train_lightcnn(model, args.weights_lightcnn)

    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_idmmd = IDMMD().cuda()
    margin_softmax = CosFace(s=64.0, m=0.4).cuda()

    '''
    Stage I: model pretrained for last fc2 parameters
    '''
    params_pretrain = []
    for name, value in model.named_parameters():
        if name == "module.weight":
            params_pretrain += [{"params": value, "lr": 1 * args.lr}]

    print("Stage I: trainable params ", len(params_pretrain))
    assert len(params_pretrain) > 0

    # optimizer
    optimizer_pretrain = torch.optim.SGD(params_pretrain, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, 5):
        pre_train_pair(train_loader_real_paired, model, criterion, margin_softmax, optimizer_pretrain, epoch)
        # save_checkpoint(model, epoch, args.save_name+"_pretrain", dataset)

    '''
    Stage II: model finetune for full network
    '''
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    start_epoch = args.pre_epoch + 1
    for epoch in range(start_epoch, args.epochs + 1):
        adjust_learning_rate(args.lr, args.step_size, optimizer, epoch)
        train(train_loader_real_paired, model, criterion, criterion_idmmd, margin_softmax, optimizer, epoch)
        if epoch == args.epochs or epoch % 10 == 0:
            save_checkpoint(model, epoch, args.save_name, dataset)
    

def pre_train_pair(train_loader, model, criterion, margin_softmax, optimizer, epoch):
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    for i, (vis_img, nir_img, vis_label, nir_label) in enumerate(train_loader):

        input = torch.cat((vis_img, nir_img), 0).cuda(non_blocking=True)
        label = torch.cat((vis_label, nir_label), 0).cuda(non_blocking=True)
        batch_size = input.size(0)

        if batch_size < 2*args.batch_size:
            continue

        # forward
        output = model(input)[0]
        output = margin_softmax(output, label)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, label.data, topk=(1, 5))
        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)

        # print log
        if i % args.print_iter == 0:
            info = "====> Epoch: [{:0>3d}][{:3d}/{:3d}] | ".format(epoch, i, len(train_loader))
            info += "Loss: ce: {:4.3f} | ".format(loss.item())
            info += "Prec@1: {:4.2f} ({:4.2f}) Prec@5: {:4.2f} ({:4.2f})".format(top1.val, top1.avg, top5.val, top5.avg)
            print(info)


def train(train_loader, model, criterion, criterion_idmmd, margin_softmax, optimizer, epoch, beta = 100):
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    for i, (vis_img, nir_img, vis_label, nir_label) in enumerate(train_loader):

        input = torch.cat((vis_img, nir_img), 0).cuda(non_blocking=True)
        label = torch.cat((vis_label, nir_label), 0).cuda(non_blocking=True)
        batch_size = input.size(0)

        if batch_size < 2*args.batch_size:
            continue
        
        # forward
        output, fc = model(input)
        output = margin_softmax(output, label)
        loss_ce = criterion(output, label)

        num_vis = vis_img.size(0)
        num_nir = nir_img.size(0)
        fc_vis, fc_nir = torch.split(fc, [num_vis, num_nir], dim=0)

        loss_idmmd = criterion_idmmd(fc_vis, fc_nir, label[:vis_img.size(0)])

        loss = loss_ce + beta * loss_idmmd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, label.data, topk=(1, 5))
        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)

        # print log
        if i % args.print_iter == 0:
            info = "====> Epoch: [{:0>3d}][{:3d}/{:3d}] | ".format(epoch, i, len(train_loader))
            info += "Loss_ce: {:4.3f} | ".format(loss_ce.data)
            info += "loss_idmmd: {:4.3f} | ".format(loss_idmmd.data)
            info += "Loss_all: {:4.3f} | ".format(loss.item())
            info += "Prec@1: {:4.2f} ({:4.2f}) Prec@5: {:4.2f} ({:4.2f})".format(top1.val, top1.avg, top5.val, top5.avg)
            print(info)


if __name__ == "__main__":
    main()
