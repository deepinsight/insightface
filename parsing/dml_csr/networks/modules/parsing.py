#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author     :   Qingping Zheng
@Contact    :   qingpingzheng2014@gmail.com
@File       :   parsing.py
@Time       :   10/01/21 00:00 PM
@Desc       :   
@License    :   Licensed under the Apache License, Version 2.0 (the "License"); 
@Copyright  :   Copyright 2022 The Authors. All Rights Reserved.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.nn as nn

from inplace_abn import InPlaceABNSync


class Parsing(nn.Module):
    def __init__(self, in_plane1, in_plane2, num_classes, abn=InPlaceABNSync):
        super(Parsing, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_plane1, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            abn(256)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_plane2, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            abn(48)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            abn(256),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            abn(256)
            )
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()

        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x 
      
