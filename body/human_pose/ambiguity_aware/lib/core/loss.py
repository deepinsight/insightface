import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import math

parents = [1, 2, 13, 13, 3, 4, 7, 8, 12, 12, 9, 10, 14, 13, 13, 12, 15]
bone_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]

def kl_criterion(mu1, sigma1, mu2, sigma2, mean_only=False):
    if mean_only: 
        kld = (mu1 - mu2)**2 / (2*sigma2**2)
    else:
        kld = torch.log(sigma2/sigma1) + (sigma1**2 + (mu1 - mu2)**2)/(2*sigma2**2) - 1/2
    return kld

def loss_bone(x, y):
    bones_x = (x - x[:, parents])[:, bone_indices]
    bones_y = (y - y[:, parents])[:, bone_indices]
    loss_bone = (bones_x - bones_y).pow(2).sum(dim=-1).sqrt().mean()
    return loss_bone

def loss_3d(pred3d, gt3d):
    # pred3d, gt_3d = pred3d[..., :3], gt3d[..., :3]
    # bones_pred = (pred3d - pred3d[:, parents])[:, bone_indices]
    # bones_gt = (gt3d - gt3d[:, parents])[:, bone_indices]
    # loss_loc = (torch.abs(pred3d - gt3d)).sum(dim=-1).sqrt().mean()
    # loss_bone = (torch.abs(bones_pred - bones_gt)).sum(dim=-1).sqrt().mean()
    # return loss_loc + loss_bone
    # return torch.sum(torch.abs(pred3d[..., :3] - gt3d[..., :3]), dim=-1).mean()
    return torch.sum((pred3d[..., :3] - gt3d[..., :3]) ** 2, dim=-1).sqrt().mean()


def loss_2d(pred2d, gt2d):
    # return torch.sum(torch.abs(pred2d[..., :2] - gt2d[..., :2]), dim=-1).mean()
    return torch.sum((pred2d[..., :2] - gt2d[..., :2]) ** 2, dim=-1).sqrt().mean()

def loss_gadv(fake_logits):
    loss_func = torch.nn.BCEWithLogitsLoss().cuda()
    fake_one = torch.ones_like(fake_logits).cuda()
    loss_g = loss_func(fake_logits, fake_one)
    # loss_g = loss_func(fake_one, fake_logits)
    return loss_g

def loss_dadv(real_logits, fake_logits):
    loss_func = torch.nn.BCEWithLogitsLoss().cuda()
    real_one = torch.ones_like(real_logits).cuda()
    fake_one = torch.zeros_like(fake_logits).cuda()
    loss_d_real = loss_func(real_logits, real_one)
    loss_d_fake = loss_func(fake_logits, fake_one)
    # loss_d_real = loss_func(real_one, real_logits)
    # loss_d_fake = loss_func(fake_one, fake_logits)
    return loss_d_real + loss_d_fake

class Losses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, parent, summary_writer):
        super(Losses, self).__init__()
        self.bone_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16])
        self.parent=parent
        self.summary_writer=summary_writer
        self.train_cnt=0

    def forward(self, pred, lbl):
        gt_bones = (lbl - lbl[:, self.parent])[:, self.bone_idx]
        pred_bones = (pred - pred[:, self.parent])[:, self.bone_idx]
        self.loss_3d = torch.sqrt(torch.sum((lbl - pred) ** 2, -1)).mean()
        self.loss_bone = torch.sqrt(torch.sum((gt_bones - pred_bones) ** 2, -1)).mean()
        self.train_cnt += 1

        self.summary_writer.add_scalar('h36m_train3d_loss', self.loss_3d.item(), self.train_cnt)
        self.summary_writer.add_scalar('h36m_bone_loss', self.loss_bone.item(), self.train_cnt)
        return self.loss_3d + self.loss_bone
