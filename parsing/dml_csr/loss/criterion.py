#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author     :   Qingping Zheng
@Contact    :   qingpingzheng2014@gmail.com
@File       :   criterion.py
@Time       :   10/01/21 00:00 PM
@Desc       :   
@License    :   Licensed under the Apache License, Version 2.0 (the "License"); 
@Copyright  :   Copyright 2015 The Authors. All Rights Reserved.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import numpy as np

from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from .lovasz_softmax import LovaszSoftmax
from .kl_loss import KLDivergenceLoss
from .consistency_loss import ConsistencyLoss


class Criterion(nn.Module):
    """DML_CSR loss for face parsing.
    
    Put more focus on facial components like eyes, eyebrow, nose and mouth
    """
    def __init__(self, loss_weight=[1.0, 1.0, 1.0, 1.0, 1.0], ignore_index=255, lambda_1=1, lambda_2=1, lambda_3=1, num_classes=11):
        super(Criterion, self).__init__()
        self.ignore_index = ignore_index   
        self.loss_weight = loss_weight
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index) 
        self.criterion_weight = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
        # self.sed = WeightedCrossEntropyWithLogits()
        self.lovasz = LovaszSoftmax(ignore_index=ignore_index)
        self.kldiv = KLDivergenceLoss(ignore_index=ignore_index)
        self.reg = ConsistencyLoss(ignore_index=ignore_index)
        self.lamda_1 = lambda_1
        self.lamda_2 = lambda_2
        self.lamda_3 = lambda_3
        self.num_classes = num_classes
          
    def forward(self, preds, target, cycle_n=None):
        h, w = target[0].size(1), target[0].size(2)
        
        # binary edge
        input_binary_labels = target[1].data.cpu().numpy().astype(np.int64)
        binary_pos_num = np.sum(input_binary_labels==1).astype(np.float)
        binary_neg_num = np.sum(input_binary_labels==0).astype(np.float)

        binary_weight_pos = binary_neg_num/(binary_pos_num + binary_neg_num)
        binary_weight_neg = binary_pos_num/(binary_pos_num + binary_neg_num)
        binary_weights = (binary_weight_neg, binary_weight_pos)  
        binary_weights = torch.from_numpy(np.array(binary_weights)).float().cuda()

        # print('target', target[0].size(), target[1].size())
        binary_edge_p_num = target[1].cpu().numpy().reshape(target[1].size(0),-1).sum(axis=1)
        # print('edge_p_num_1', edge_p_num.shape)
        binary_edge_p_num = np.tile(binary_edge_p_num, [h, w, 1]).transpose(2,1,0)
        # print('edge_p_num_2', edge_p_num.shape)
        binary_edge_p_num = torch.from_numpy(binary_edge_p_num).cuda().float()

        # semantic edge
        input_semantic_labels = target[2].data.cpu().numpy().astype(np.int64)
        semantic_weights = []
        semantic_pos_num = np.sum(input_semantic_labels>0).astype(np.float)
        semantic_neg_num = np.sum(input_semantic_labels==0).astype(np.float)

        for lbl in range(self.num_classes):
            lbl_num = np.sum(input_semantic_labels==lbl).astype(np.float)
            weight_lbl = lbl_num/(semantic_pos_num + semantic_neg_num)
            semantic_weights.append(weight_lbl)
        semantic_weights = torch.from_numpy(np.array(semantic_weights)).float().cuda()

        # print('target', target[0].size(), target[1].size())
        semantic_edge_p_num = np.count_nonzero(target[2].cpu().numpy().reshape(target[2].size(0),-1), axis=1)
        # print('edge_p_num_1', edge_p_num.shape)
        semantic_edge_p_num = np.tile(semantic_edge_p_num, [h, w, 1]).transpose(2,1,0)
        # print('edge_p_num_2', edge_p_num.shape)
        semantic_edge_p_num = torch.from_numpy(semantic_edge_p_num).cuda().float()
        
        loss_binary_edge = 0; loss_semantic_edge = 0; loss_parse = 0; loss_att_parse = 0; loss_att_binary_edge = 0; loss_att_semantic_edge = 0; loss_consistency = 0
        # print(preds[1].size(), target[1].size(), weights, len(preds))

        # loss for parsing
        scale_parse = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True) # parsing
        loss_parse += 0.5 * self.lamda_1 * self.lovasz(scale_parse, target[0])
        
        if target[3] is None:
            loss_parse += 0.5 * self.lamda_1 * self.criterion(scale_parse, target[0])
        else:
            soft_scale_parse = F.interpolate(input=target[2], size=(h, w), mode='bilinear', align_corners=True)
            soft_scale_parse = moving_average(soft_scale_parse, to_one_hot(target[0], num_cls=self.num_classes),
                                                1.0 / (cycle_n + 1.0))
            loss_parse += 0.5 * self.lamda_1 * self.kldiv(scale_parse, soft_scale_parse, target[0])

        # loss for binary edge
        scale_binary_edge = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)  # edge 
        
        if target[4] is None:
            loss_binary_edge = self.lamda_2 * F.cross_entropy(scale_binary_edge, target[1], binary_weights)
        else:
            soft_scale_binary_edge = F.interpolate(input=target[4], size=(h, w), mode='bilinear', align_corners=True)
            soft_scale_binary_edge = moving_average(soft_scale_binary_edge, to_one_hot(target[1], num_cls=2),
                                                1.0 / (cycle_n + 1.0))
            loss_binary_edge += self.lamda_2 * self.kldiv(scale_binary_edge, soft_scale_binary_edge, target[0])

        # loss for semantic edge
        scale_semantic_edge = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)  # edge 
        
        if target[5] is None:
            loss_semantic_edge = self.lamda_3 * F.cross_entropy(scale_semantic_edge, target[2], semantic_weights)
            # loss_edge = self.lamda_2 * self.sed(scale_edge, target[1])
        else:
            soft_scale_semantic_edge = F.interpolate(input=target[5], size=(h, w), mode='bilinear', align_corners=True)
            soft_scale_semantic_edge = moving_average(soft_scale_semantic_edge, to_one_hot(target[2], num_cls=self.num_classes),
                                                1.0 / (cycle_n + 1.0))
            loss_semantic_edge += self.lamda_3 * self.kldiv(scale_semantic_edge, soft_scale_semantic_edge, target[0])
    
        # binary edge attention loss
        loss_att_binary_edge_ = self.criterion_weight(scale_parse, target[0]) * target[1].float()
        loss_att_binary_edge_ = loss_att_binary_edge_ / binary_edge_p_num  # only compute the edge pixels
        loss_att_binary_edge_ = torch.sum(loss_att_binary_edge_) / target[1].size(0)  # mean for batchsize      
                
        loss_parse += loss_parse
        loss_att_binary_edge += loss_att_binary_edge
        loss_att_binary_edge += loss_att_binary_edge_

        # semantic edge attention loss
        loss_att_semantic_edge_ = self.criterion_weight(scale_parse, target[0]) * target[2].float()
        loss_att_semantic_edge_ = loss_att_semantic_edge_ / semantic_edge_p_num  # only compute the edge pixels
        loss_att_semantic_edge_ = torch.sum(loss_att_semantic_edge_) / target[2].size(0)  # mean for batchsize      
                
        loss_parse += loss_parse
        loss_semantic_edge += loss_semantic_edge
        loss_att_semantic_edge += loss_att_semantic_edge_
        # loss_consistency += loss_consistency
        
        # print('loss_parse: {}\t loss_edge: {}\t loss_att_edge: {}\t loss_semantic_edge: {}'.format(loss_parse,loss_edge,loss_att_edge, loss_consistency))
        return self.loss_weight[0]*loss_parse + self.loss_weight[1]*loss_binary_edge + self.loss_weight[2]*loss_semantic_edge \
            + self.loss_weight[3]*loss_att_binary_edge + self.loss_weight[4]*loss_att_semantic_edge


def moving_average(target1, target2, alpha=1.0):
    target = 0
    target += (1.0 - alpha) * target1
    target += target2 * alpha
    return target


def to_one_hot(tensor, num_cls, dim=1, ignore_index=255):
    b, h, w = tensor.shape
    tensor[tensor == ignore_index] = 0
    onehot_tensor = torch.zeros(b, num_cls, h, w).cuda()
    onehot_tensor.scatter_(dim, tensor.unsqueeze(dim), 1)
    return onehot_tensor
