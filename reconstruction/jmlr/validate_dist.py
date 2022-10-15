from dataset import FaceDataset, DataLoaderX, MXFaceDataset
import argparse
import logging
import os
import time
import timm
import glob
import numpy as np
import os.path as osp
from utils.utils_config import get_config
from scipy.spatial.transform import Rotation

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import torch.utils.data.distributed
from backbones import get_network
from inference_simple import JMLRInference, Rt_from_6dof
from dataset import Rt26dof


def l2_distance(a, b):
    dist = np.sqrt(np.sum(np.square(a-b), axis=1))
    distance_list = np.sqrt(((a - b) ** 2).sum(axis=2)).mean(axis=1)
    return distance_list

def main(args):
    cfg = get_config(args.config)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    #dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    dist.init_process_group('nccl')

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    task1 = cfg.task
    cfg.aug_modes = []
    cfg.task = 0
    batch_size = cfg.batch_size
    dataset = MXFaceDataset(cfg=cfg, is_train=False, local_rank=local_rank)
    if local_rank==0:
        print('total:', len(dataset))
        print('total batch:', len(dataset)//(batch_size*world_size))
    cfg.task = task1
    net = JMLRInference(cfg, local_rank)
    net = net.to(local_rank)
    net.eval()
    #net = torch.nn.parallel.DistributedDataParallel(
    #    module=net, broadcast_buffers=False, device_ids=[local_rank])
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            pin_memory=True,
            num_workers=3,
            drop_last=False,
            )
    num_epochs = 1
    all_pred_verts = torch.zeros((len(dataset),1220,3), requires_grad=False).to(local_rank)
    all_pred_R = torch.zeros((len(dataset),3,3), requires_grad=False).to(local_rank)
    all_pred_t = torch.zeros((len(dataset),1,3), requires_grad=False).to(local_rank)
    all_pred_verts2d = torch.zeros((len(dataset),1220,2), requires_grad=False).to(local_rank)

    all_label_verts = torch.zeros((len(dataset),1220,3), requires_grad=False).to(local_rank)
    all_label_R = torch.zeros((len(dataset),3,3), requires_grad=False).to(local_rank)
    all_label_t = torch.zeros((len(dataset),1,3), requires_grad=False).to(local_rank)
    all_label_verts2d = torch.zeros((len(dataset),1220,2), requires_grad=False).to(local_rank)
    all_weight = 0.0
    FLIPS = [False, True] if cfg.enable_flip else [False]
    #FLIPS = [False]
    if local_rank==0:
        print('FLIPS:', FLIPS)
    for epoch in range(num_epochs):
        weight = 1.0

        if epoch>0:
            dataset.set_test_aug()
            weight = 0.6
        all_weight += weight
        #all_distance = torch.zeros((len(dataset),), requires_grad=False).to(local_rank)
        diff_R = []
        diff_t = []
        sampler.set_epoch(epoch)
        for idx, sample in enumerate(loader):
            img_local = sample['img_local']
            label_verts = sample['verts']
            tform = sample['tform']
            label_6dof = sample['6dof']
            data_idx = sample['idx']
            label_verts2d = sample['verts2d']
            img_local = img_local.to(local_rank)
            pred_verts, pred_verts2d, pred_points2d = [], [], []
            for is_flip in FLIPS:
                with torch.no_grad():
                    #pred_verts, R_pred, t_pred = infer.forward(img_local, img_raw, tform)
                    #pred1, pred2 = net(img_local.to(local_rank), img_raw.to(local_rank))
                    pred1, pred2, meta = net(img_local, is_flip=is_flip)
                _pred_verts = net.convert_verts(pred1, meta)
                pred_verts.append(_pred_verts)
                _pred_verts2d, _pred_points2d = net.convert_2d(pred2, tform, meta)
                pred_verts2d.append(_pred_verts2d)
                pred_points2d.append(_pred_points2d)
            pred_verts = sum(pred_verts) / len(pred_verts)
            pred_verts2d = sum(pred_verts2d) / len(pred_verts2d)
            pred_points2d = sum(pred_points2d) / len(pred_points2d)
            R_pred, t_pred = net.solve(pred_verts, pred_verts2d)
            label_6dof = label_6dof.cpu().numpy()
            label_6dof = label_6dof * cfg.label_6dof_std.reshape(1, 6) + cfg.label_6dof_mean.reshape(1,6)

            R_label, t_label = Rt_from_6dof(label_6dof)
            diff_R.append(np.mean(np.abs(R_pred - R_label)))
            diff_t.append(np.mean(np.abs(t_pred - t_label)))
            #distance = torch.tensor(distance, dtype=torch.float32, requires_grad=False).to(local_rank)
            data_idx = data_idx.view(-1)
            #all_distance[data_idx] = distance
            label_verts = label_verts.view(-1,1220,3) / 10.0
            if epoch==0:
                all_label_verts[data_idx,:,:] = label_verts.to(local_rank)
                all_label_R[data_idx,:,:] = torch.tensor(R_label).to(local_rank)
                all_label_t[data_idx,:,:] = torch.tensor(t_label).to(local_rank)
                all_label_verts2d[data_idx,:,:] = label_verts2d.to(local_rank)
            all_pred_verts[data_idx,:,:] += torch.tensor(pred_verts).to(local_rank) * weight
            #all_pred_R[data_idx,:,:] += torch.tensor(R_pred).to(local_rank) * weight
            #all_pred_t[data_idx,:,:] += torch.tensor(t_pred).to(local_rank) * weight
            all_pred_verts2d[data_idx,:,:] += torch.tensor(pred_verts2d).to(local_rank) * weight
            if idx%20==0 and local_rank==0:
                print('processing-epoch-idx:', epoch, idx)
                #print('distance:', distance.shape, distance.cpu().numpy().mean())
                print('diff_R:', np.mean(diff_R))
                print('diff_t:', np.mean(diff_t))

    dist.all_reduce(all_label_verts, op=dist.ReduceOp.SUM)
    dist.all_reduce(all_label_verts2d, op=dist.ReduceOp.SUM)
    dist.all_reduce(all_label_R, op=dist.ReduceOp.SUM)
    dist.all_reduce(all_label_t, op=dist.ReduceOp.SUM)

    dist.all_reduce(all_pred_verts, op=dist.ReduceOp.SUM)
    dist.all_reduce(all_pred_verts2d, op=dist.ReduceOp.SUM)
    #dist.all_reduce(all_pred_R, op=dist.ReduceOp.SUM)
    #dist.all_reduce(all_pred_t, op=dist.ReduceOp.SUM)
    #dist.all_reduce(all_distance, op=dist.ReduceOp.SUM)
    if local_rank==0:
        label_verts = all_label_verts.cpu().numpy()
        label_verts2d = all_label_verts2d.cpu().numpy()
        R_label = all_label_R.cpu().numpy()
        t_label = all_label_t.cpu().numpy()

        pred_verts = all_pred_verts.cpu().numpy() / all_weight
        #R_pred = all_pred_R.cpu().numpy() / all_weight
        #t_pred = all_pred_t.cpu().numpy() / all_weight
        pred_verts2d = all_pred_verts2d.cpu().numpy() / all_weight
        R_pred, t_pred = net.solve(pred_verts, pred_verts2d)
        #R_pred, t_pred = net.solve(pred_verts, label_verts2d)
        #R_pred, t_pred = net.solve(label_verts, pred_verts2d)


        X1 = label_verts @ R_label + t_label
        X2 = pred_verts @ R_pred + t_pred
        X3 = label_verts @ R_pred + t_pred
        X4 = pred_verts @ R_label + t_label
        distance = l2_distance(X1, X2) + l2_distance(X1, X3) + 10.0*l2_distance(X1,X4)
        distance *= 1000.0

        print('top20 distance:', np.mean(distance[:20]))


        score = np.mean(distance)
        print('epoch distance:', epoch, score)
        with open(os.path.join(cfg.output, 'val.txt'), 'w') as f:
            f.write("%f\n"%score)

if __name__ == "__main__":
    #torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='JMLR validation')
    parser.add_argument('config', type=str, help='config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    args_ = parser.parse_args()
    main(args_)

