#!/usr/bin/env python3
# coding=utf-8

import torch 
import torch.nn as nn
from lib.models.building_blocks import ResBlock 

class Scaler(nn.Module):
    # for input size, default value is: 17 * 2 + h + w + mean_hw = 37
    def __init__(self, input_size=37, num_channels=1024, num_res_blocks=1, dropout=0.25, bn_track=True):
        super(Scaler, self).__init__()
        self.blocks = nn.ModuleList()
        self.pre = nn.Linear(input_size, num_channels)
        for _ in range(num_res_blocks):
            self.blocks.append(ResBlock(num_channels, input_size, dropout=dropout, use_bn=True, bn_track=bn_track))
        self.out = nn.Linear(num_channels, 1)

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1)
        out = self.pre(x)
        for block in self.blocks: 
            out = block(out, x)
        out = self.out(out)
        # add sigmoid 
        # out = 2 * torch.sigmoid(out)
        return out 

def get_scaler(cfg):
    input_size = cfg.NETWORK.SCALER_INPUT_SIZE 
    num_res_blocks = cfg.NETWORK.SCALER_RES_BLOCKS
    num_channels = cfg.NETWORK.NUM_CHANNELS 
    bn_track = cfg.NETWORK.BN_TRACK 
    dropout = cfg.NETWORK.DROPOUT
    scaler = Scaler(input_size=input_size, num_channels=num_channels, num_res_blocks=num_res_blocks, dropout=dropout, bn_track=bn_track)
    return scaler
