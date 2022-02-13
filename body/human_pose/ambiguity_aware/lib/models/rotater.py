#!/usr/bin/env python3
# coding=utf-8

import torch 
import torch.nn as nn
from lib.models.building_blocks import ResBlock 

class Rotater(nn.Module):
    # for input size, default value is: 17 * 2 + h + w + mean_hw = 37
    # input_size is: 17 * 3 = 51, output size is: 3(euler angles in order z, x, y)
    def __init__(self, input_size=51, is_euler=True, num_channels=1024, num_res_blocks=1, dropout=0.25, bn_track=True):
        super(Rotater, self).__init__()
        self.output_size = 3 if is_euler else 9
        self.blocks = nn.ModuleList()
        self.pre = nn.Linear(input_size, num_channels)
        for _ in range(num_res_blocks):
            self.blocks.append(ResBlock(num_channels, input_size, dropout=dropout, use_bn=True, bn_track=bn_track))
        self.out = nn.Linear(num_channels, self.output_size)

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1)
        out = self.pre(x)
        for block in self.blocks: 
            out = block(out, x)
        out = self.out(out)
        if self.output_size == 3: 
            # pre euler
            out = torch.clamp(out, min=-3.14159/2, max=3.14159/2)
            pass
        else: 
            # pre rotation matrix
            out = torch.clamp(out, min=-1.0, max=1.0)
        return out 

def get_rotater(cfg):
    num_res_blocks = cfg.NETWORK.ROTATER_RES_BLOCKS
    num_channels = cfg.NETWORK.NUM_CHANNELS 
    bn_track = cfg.NETWORK.BN_TRACK 
    dropout = cfg.NETWORK.DROPOUT
    is_euler = cfg.NETWORK.ROTATER_PRE_EULER
    rotater = Rotater(is_euler=is_euler, num_channels=num_channels, num_res_blocks=num_res_blocks, dropout=dropout, bn_track=bn_track)
    return rotater
