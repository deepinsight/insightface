#!/usr/bin/env python3
# coding=utf-8

import math
import torch 
import torch.nn as nn 
from torch.nn.utils import spectral_norm

def get_common_block(in_channels, dropout=0.25, use_bn=True, bn_track=True, use_spectral_norm=False):
    fc1 = nn.Linear(in_channels, in_channels)
    fc2 = nn.Linear(in_channels, in_channels)
    relu1 = nn.ReLU(inplace=True); relu2 = nn.ReLU(inplace=True)
    dp1 = nn.Dropout(dropout); dp2 = nn.Dropout(dropout)
    if use_spectral_norm: 
        fc1 = spectral_norm(fc1)
        fc2 = spectral_norm(fc2)
    if use_bn: 
        bn1 = nn.BatchNorm1d(in_channels, track_running_stats=bn_track)
        bn2 = nn.BatchNorm1d(in_channels, track_running_stats=bn_track)
        layers = [fc1, bn1, relu1, dp1, fc2, bn2, relu2, dp2]
    else: 
        layers = [fc1, relu1, dp1, fc2, relu2, dp2]
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, in_channels, res_channels, dropout=0.25, use_bn=True, bn_track=True, use_spectral_norm=False):
        super(ResBlock, self).__init__()
        self.main = get_common_block(in_channels, dropout=dropout, use_bn=use_bn, bn_track=bn_track, use_spectral_norm=use_spectral_norm)
        self.sc = nn.Linear(in_channels + res_channels, in_channels)
        if use_spectral_norm: 
            self.sc = spectral_norm(self.sc)

    def forward(self, x, joint_sc):
        a = torch.cat([x, joint_sc], dim=-1)
        x = self.main(x)
        a = self.sc(a)
        return x + a
    
class Branch(nn.Module): 
    def __init__(self, in_channels, res_channels, out_channels, dropout=0.25, bn_track=True):
        super(Branch, self).__init__()
        # Branch will never use spectral_normalization
        self.main = get_common_block(in_channels, dropout=dropout, bn_track=bn_track, use_spectral_norm=False)
        self.sc = nn.Linear(in_channels + res_channels, in_channels)
        self.out = nn.Linear(in_channels, out_channels)

    def forward(self, x, joint_sc): 
        a = torch.cat([x, joint_sc], dim=-1)
        x = self.main(x)
        a = self.sc(a)
        return self.out(x + a)
