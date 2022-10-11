import argparse
import logging
import os
import time
import timm
import glob
import numpy as np
import os.path as osp

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from dataset import FaceDataset, DataLoaderX, MXFaceDataset, get_tris

from backbones import get_network
from utils.utils_amp import MaxClipGradScaler
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_config import get_config
from lr_scheduler import get_scheduler
from timm.optim.optim_factory import create_optimizer





def main(args):
    cfg = get_config(args.config)
    if not cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist.init_process_group('nccl')
    except KeyError:
        world_size = 1
        rank = 0
        dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:12584", rank=rank, world_size=world_size)


    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    if not os.path.exists(cfg.output) and rank is 0:
        os.makedirs(cfg.output)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)
    if rank==0:
        logging.info(args)
        logging.info(cfg)
        print(cfg.flipindex.shape, cfg.flipindex[400:410])
    train_set = MXFaceDataset(cfg=cfg, is_train=True, local_rank=local_rank)
    cfg.num_images = len(train_set)
    cfg.world_size = world_size
    total_batch_size = cfg.batch_size * cfg.world_size
    epoch_steps = cfg.num_images // total_batch_size
    cfg.warmup_steps = epoch_steps * cfg.warmup_epochs
    if cfg.max_warmup_steps>0:
        cfg.warmup_steps = min(cfg.max_warmup_steps, cfg.warmup_steps)
    cfg.total_steps = epoch_steps * cfg.num_epochs
    if cfg.lr_epochs is not None:
        cfg.lr_steps = [m*epoch_steps for m in cfg.lr_epochs]
    else:
        cfg.lr_steps = None
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=False, drop_last=True)

    
    net = get_network(cfg).to(local_rank)
    

    if cfg.resume:
        try:
            
            ckpts = list(glob.glob(osp.join(cfg.resume_path, "backbone*.pth")))
            backbone_pth = sorted(ckpts)[-1]
            backbone_ckpt = torch.load(backbone_pth, map_location=torch.device(local_rank))
            net.load_state_dict(backbone_ckpt['model'])
            if rank==0:
                logging.info("backbone resume successfully! %s"%backbone_pth)
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("resume fail!!")
            raise RuntimeError


    net = torch.nn.parallel.DistributedDataParallel(
        module=net, broadcast_buffers=False, device_ids=[local_rank])
    net.train()


 

    if cfg.opt=='sgd':
        opt = torch.optim.SGD(
            params=[
                {"params": net.parameters()}, 
                ],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.opt=='adam':
        opt = torch.optim.Adam(
            params=[
                {"params": net.parameters()}, 
                ],
            lr=cfg.lr)
    elif cfg.opt=='adamw':
        opt = torch.optim.AdamW(
            params=[
                {"params": net.parameters()}, 
                ],
            lr=cfg.lr, weight_decay=cfg.weight_decay)


    scheduler = get_scheduler(opt, cfg)
    if cfg.resume:
        if rank==0:
            logging.info(opt)



    if cfg.resume:
        for g in opt_pfc.param_groups:
            for key in ['lr', 'initial_lr']:
                g[key] = cfg.lr



    start_epoch = 0
    total_step = cfg.total_steps
    if rank==0: 
        logging.info(opt)
        logging.info("Total Step is: %d" % total_step)


    loss = {
            'Loss': AverageMeter(),

           }

    global_step = 0
    grad_amp = None
    if cfg.fp16>0:
        if cfg.fp16==1:
            grad_amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
        elif cfg.fp16==2:
            grad_amp = MaxClipGradScaler(64, 1024, growth_interval=200)
        elif cfg.fp16==3:
            grad_amp = MaxClipGradScaler(4, 8, growth_interval=200)
        else:
            assert 'fp16 mode not set'

    callback_checkpoint = CallBackModelCheckpoint(rank, cfg)

    callback_checkpoint(global_step, net, opt)

    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, None)

    l1loss = nn.L1Loss()


    tris = get_tris(cfg)
    tri_index = torch.tensor(tris, dtype=torch.long).to(local_rank)
    use_eyes = cfg.eyes is not None

    for epoch in range(start_epoch, cfg.num_epochs):
        train_sampler.set_epoch(epoch)
        for step, value in enumerate(train_loader):
            global_step += 1
            img = value['img_local'].to(local_rank)
            dloss = {}
            assert cfg.task==0
            label_verts = value['verts'].to(local_rank)
            label_points2d = value['points2d'].to(local_rank)
            #need_eyes = 'eye_verts' in value
            preds = net(img)

            
            if use_eyes:
                pred_verts, pred_points2d, pred_eye_verts, pred_eye_points = preds.split([1220*3, 1220*2, 481*2*3, 481*2*2], dim=1)
                pred_eye_verts = pred_eye_verts.view(cfg.batch_size, 481*2, 3)
                pred_eye_points = pred_eye_points.view(cfg.batch_size, 481*2, 2)
            else:
                pred_verts, pred_points2d = preds.split([1220*3, 1220*2], dim=1)
            pred_verts = pred_verts.view(cfg.batch_size, 1220, 3)
            pred_points2d = pred_points2d.view(cfg.batch_size, 1220, 2)
            if not cfg.use_rtloss:
                loss1 = F.l1_loss(pred_verts, label_verts)
            else:
                label_Rt = value['rt'].to(local_rank)
                _ones = torch.ones([pred_verts.shape[0], 1220, 1], device=pred_verts.device)
                pred_verts = torch.cat([pred_verts/10, _ones], dim=2)
                pred_verts = torch.bmm(pred_verts,label_Rt) * 10.0
                label_verts = torch.cat([label_verts/10, _ones], dim=2)
                label_verts = torch.bmm(label_verts,label_Rt) * 10.0
                loss1 = F.l1_loss(pred_verts, label_verts)

            loss2 = F.l1_loss(pred_points2d, label_points2d)
            loss3d = loss1 * cfg.lossw_verts3d
            loss2d = loss2 * cfg.lossw_verts2d
            dloss['Loss'] = loss3d + loss2d
            dloss['Loss3D'] = loss3d
            dloss['Loss2D'] = loss2d
            if use_eyes:
                label_eye_verts = value['eye_verts'].to(local_rank)
                label_eye_points = value['eye_points'].to(local_rank)
                loss3 = F.l1_loss(pred_eye_verts, label_eye_verts)
                loss4 = F.l1_loss(pred_eye_points, label_eye_points)
                loss3 = loss3 * cfg.lossw_eyes3d
                loss4 = loss4 * cfg.lossw_eyes2d
                dloss['Loss'] += loss3
                dloss['Loss'] += loss4
                dloss['LossEye3d'] = loss3
                dloss['LossEye2d'] = loss4

            if cfg.loss_bone3d:
                bone_losses = []
                for i in range(3):
                    pred_verts_x = pred_verts[:,tri_index[:,i%3],:]
                    pred_verts_y = pred_verts[:,tri_index[:,(i+1)%3],:]
                    label_verts_x = label_verts[:,tri_index[:,i%3],:]
                    label_verts_y = label_verts[:,tri_index[:,(i+1)%3],:]
                    dist_pred = torch.norm(pred_verts_x - pred_verts_y, p=2, dim=-1, keepdim=False)
                    dist_label = torch.norm(label_verts_x - label_verts_y, p=2, dim=-1, keepdim=False)
                    bone_losses.append(F.l1_loss(dist_pred, dist_label) * cfg.lossw_bone3d)
                _loss = sum(bone_losses)
                dloss['Loss'] += _loss
                dloss['LossBone3d'] = _loss
                        

            if cfg.loss_bone2d:
                bone_losses = []
                for i in range(3):
                    pred_points2d_x = pred_points2d[:,tri_index[:,i%3],:]
                    pred_points2d_y = pred_points2d[:,tri_index[:,(i+1)%3],:]
                    label_points2d_x = label_points2d[:,tri_index[:,i%3],:]
                    label_points2d_y = label_points2d[:,tri_index[:,(i+1)%3],:]
                    dist_pred = torch.norm(pred_points2d_x - pred_points2d_y, p=2, dim=-1, keepdim=False)
                    dist_label = torch.norm(label_points2d_x - label_points2d_y, p=2, dim=-1, keepdim=False)
                    bone_losses.append(F.l1_loss(dist_pred, dist_label) * cfg.lossw_bone2d)
                _loss = sum(bone_losses)
                dloss['Loss'] += _loss
                dloss['LossBone2d'] = _loss
                        
            iter_loss = dloss['Loss']

            if cfg.fp16>0:
                grad_amp.scale(iter_loss).backward()
                grad_amp.unscale_(opt)
                if cfg.fp16<2:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                grad_amp.step(opt)
                grad_amp.update()
            else:
                iter_loss.backward()
                opt.step()


            opt.zero_grad()

            if cfg.lr_func is None:
                scheduler.step()

            with torch.no_grad():
                loss['Loss'].update(iter_loss.item(), 1)
                for k in dloss:
                    if k=='Loss':
                        continue
                    v = dloss[k].item()
                    if k not in loss:
                        loss[k] = AverageMeter()
                    loss[k].update(v, 1)

                callback_logging(global_step, loss, epoch, cfg.fp16, grad_amp, opt)

        if cfg.lr_func is not None:
            scheduler.step()

    callback_checkpoint(9999, net, opt)
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='JMLR Training')
    parser.add_argument('config', type=str, help='config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    args_ = parser.parse_args()
    main(args_)

