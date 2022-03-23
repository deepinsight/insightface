#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author     :   Qingping Zheng
@Contact    :   qingpingzheng2014@gmail.com
@File       :   consistency_loss.py
@Time       :   10/01/21 00:00 PM
@Desc       :   
@License    :   Licensed under the Apache License, Version 2.0 (the "License"); 
@Copyright  :   Copyright 2015 The Authors. All Rights Reserved.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch import nn


def generate_edge_tensor(label, edge_width=3):
    label = label.type(torch.cuda.FloatTensor)
    if len(label.shape) == 2:
        label = label.unsqueeze(0)
    n, h, w = label.shape
    edge = torch.zeros(label.shape, dtype=torch.float).cuda()
    # right
    edge_right = edge[:, 1:h, :]
    edge_right[(label[:, 1:h, :] != label[:, :h - 1, :]) & (label[:, 1:h, :] != 255)
               & (label[:, :h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :, :w - 1]
    edge_up[(label[:, :, :w - 1] != label[:, :, 1:w])
            & (label[:, :, :w - 1] != 255)
            & (label[:, :, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:, :h - 1, :w - 1]
    edge_upright[(label[:, :h - 1, :w - 1] != label[:, 1:h, 1:w])
                 & (label[:, :h - 1, :w - 1] != 255)
                 & (label[:, 1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:, :h - 1, 1:w]
    edge_bottomright[(label[:, :h - 1, 1:w] != label[:, 1:h, :w - 1])
                     & (label[:, :h - 1, 1:w] != 255)
                     & (label[:, 1:h, :w - 1] != 255)] = 1

    kernel = torch.ones((1, 1, edge_width, edge_width), dtype=torch.float).cuda()
    with torch.no_grad():
        edge = edge.unsqueeze(1)
        edge = F.conv2d(edge, kernel, stride=1, padding=1)
    edge[edge!=0] = 1
    edge = edge.squeeze()
    return edge


class ConsistencyLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(ConsistencyLoss, self).__init__()
        self.ignore_index=ignore_index

    def forward(self, parsing, edge, label):
        parsing_pre = torch.argmax(parsing, dim=1)
        parsing_pre[label==self.ignore_index]=self.ignore_index
        generated_edge = generate_edge_tensor(parsing_pre)
        edge_pre = torch.argmax(edge, dim=1)
        v_generate_edge = generated_edge[label!=255]
        v_edge_pre = edge_pre[label!=255]
        one = torch.ones_like(v_edge_pre)
        v_edge_pre = torch.where(v_edge_pre > 0, one, v_edge_pre)
        v_edge_pre = v_edge_pre.type(torch.cuda.FloatTensor)
        positive_union = (v_generate_edge==1)&(v_edge_pre==1) # only the positive values count
        return F.smooth_l1_loss(v_generate_edge[positive_union].squeeze(0), v_edge_pre[positive_union].squeeze(0))
