import torch
import torch.nn as nn
from lib.models.building_blocks import Branch, ResBlock 
from lib.utils.misc import init_weights

class LifterBefore(nn.Module):
    def __init__(self, num_joints, num_feats=2, num_channels=1024, num_res_blocks=2, dropout=0.25, bn_track=True):
        super(LifterBefore, self).__init__()
        input_size = num_joints * num_feats
        self.num_joints = num_joints
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.pre = nn.Sequential(
            nn.Linear(input_size, num_channels), 
            nn.BatchNorm1d(num_channels, track_running_stats=bn_track), 
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.blocks.append(ResBlock(num_channels, input_size, dropout=dropout, bn_track=bn_track, use_spectral_norm=False))

    def forward(self, joints_2d, root_3d=None):
        joints_2d = joints_2d.contiguous()
        bs = joints_2d.size(0)
        if root_3d is None:
            joints_in = joints_2d.view(bs, -1)
        else:
            joints_in = torch.cat([joints_2d.view(bs, -1), root_3d.view(bs, -1)], -1)
            
        x = self.pre(joints_in)
        for block in self.blocks:
            x = block(x, joints_in)
        out = x.view(x.size(0), self.num_channels)
        # joints_in will be used in lifter_after
        return out, joints_in

class Lifter(nn.Module): 
    def __init__(self, num_joints, num_xyz=3, num_feats=5, num_channels=1024, num_res_blocks=2, dropout=0.25, bn_track=True):
        super(Lifter, self).__init__()
        input_size = num_joints * num_feats
        self.num_xyz = num_xyz
        self.num_joints = num_joints
        self.num_res_blocks = num_res_blocks
        self.pre = nn.Sequential(
            nn.Linear(input_size, num_channels), 
            nn.BatchNorm1d(num_channels, track_running_stats=bn_track), 
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout)
        )
        self.blocks, self.branches = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_res_blocks):
            self.blocks.append(ResBlock(num_channels, input_size, dropout=dropout, bn_track=bn_track, use_spectral_norm=False))

        for _ in range(num_xyz): 
            self.branches.append(Branch(num_channels, input_size, num_joints, dropout=dropout, bn_track=bn_track))

    def forward(self, joints_2d, root_3d=None):
        joints_2d = joints_2d.contiguous()
        bs = joints_2d.size(0)
        if root_3d is None:
            joints_in = joints_2d.view(bs, -1)
        else:
            joints_in = torch.cat([joints_2d.view(bs, -1), root_3d.view(bs, -1)], -1)
            
        x = self.pre(joints_in)
        for block in self.blocks:
            x = block(x, joints_in)
        xyz = []
        for branch in self.branches: 
            xyz.append(branch(x, joints_in))
        out = torch.cat(xyz, -1)
        out = out.view(out.size(0), self.num_joints, self.num_xyz)
        return out 

def get_lifter(num_joints, num_feats=2, num_xyz=1, num_channels=1024, num_res_blocks=2, dropout=0.25, bn_track=True):
    lifter = Lifter(num_joints, num_xyz=num_xyz, num_feats=num_feats, num_channels=num_channels, \
        num_res_blocks=num_res_blocks, dropout=dropout, bn_track=bn_track)
    init_weights(lifter)
    return lifter

def get_lifter_before(num_joints, num_feats=2, num_channels=1024, num_res_blocks=2, dropout=0.25, bn_track=True):
    lifter_before = LifterBefore(num_joints, num_feats, num_channels, num_res_blocks, dropout=dropout, bn_track=bn_track)
    return lifter_before

def get_lifter_after(num_joints, num_feats=2, num_xyz=1, num_channels=1024, is_generic_baseline=False): 
    output_size = num_joints
    # output_size = num_joints if not is_generic_baseline else num_joints * 3
    input_size = num_joints * num_feats
    return Branch(num_channels, input_size, output_size)
