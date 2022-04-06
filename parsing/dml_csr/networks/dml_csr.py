#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author     :   Qingping Zheng
@Contact    :   qingpingzheng2014@gmail.com
@File       :   dml_csr.py
@Time       :   10/01/21 00:00 PM
@Desc       :   
@License    :   Licensed under the Apache License, Version 2.0 (the "License"); 
@Copyright  :   Copyright 2015 The Authors. All Rights Reserved.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch.nn as nn

from torch.nn import functional as F
from inplace_abn import InPlaceABNSync
from .modules.ddgcn import DDualGCNHead
from .modules.parsing import Parsing
from .modules.edges import Edges
from .modules.util import Bottleneck


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class DML_CSR(nn.Module):
    def __init__(self, 
                 num_classes, 
                 abn=InPlaceABNSync,
                 trained=True):
        super().__init__()
        self.inplanes = 128
        self.is_trained = trained

        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = abn(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = abn(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = abn(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = [3, 4, 23, 3]
        self.abn = abn
        strides = [1, 2, 1, 1]
        dilations = [1, 1, 1, 2]

        self.layer1 = self._make_layer(Bottleneck, 64, self.layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(Bottleneck, 128, self.layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(Bottleneck, 256, self.layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(Bottleneck, 512, self.layers[3], stride=strides[3], dilation=dilations[3], multi_grid=(1,1,1))
        # Context Aware
        self.context = DDualGCNHead(2048, 512, abn)
        self.layer6 = Parsing(512, 256, num_classes, abn)
        # edge
        if self.is_trained:
            self.edge_layer = Edges(abn, out_fea=num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.abn(planes * block.expansion, affine=True))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, abn=self.abn, dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, abn=self.abn, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x1 = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x1)
        x2 = self.layer1(x) # 119 x 119
        x3 = self.layer2(x2) # 60 x 60
        x4 = self.layer3(x3) # 60 x 60
        x5 = self.layer4(x4) # 60 x 60
        x = self.context(x5)
        seg, x = self.layer6(x, x2)

        if self.is_trained:
            binary_edge, semantic_edge, edge_fea = self.edge_layer(x2,x3,x4)
            return seg, binary_edge, semantic_edge
        
        return seg
        
