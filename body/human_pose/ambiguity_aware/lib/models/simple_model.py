#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5, spectral_norm=False, use_bn=True):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.use_bn = use_bn
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)
        if spectral_norm: 
            self.w1 = nn.utils.spectral_norm(self.w1)
            self.w2 = nn.utils.spectral_norm(self.w2)

    def forward(self, x):
        y = self.w1(x)
        if self.use_bn:
            y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        if self.use_bn:
            y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class LinearModelBefore(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_stage=4,
                 p_dropout=0.5):
        super(LinearModelBefore, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size =  17 * 2
        # 3d joints
        self.output_size = 17 * 1

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        # self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        # y = self.w2(y)
        # y should be features, and x is reshaped inputs 
        # namely x, joints_sc for LinearModelAfter
        return y, x

class LinearModelAfter(nn.Module): 
    def __init__(self): 
        super(LinearModelAfter, self).__init__()
        self.main = nn.Linear(1024, 17 * 1)

    def forward(self, x, y): 
        return self.main(x)
