import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from lib.utils.misc import init_weights, load_pickle
from lib.utils.utils import Transform3D, transform_3d, transform_3d_v2
from lib.models.building_blocks import ResBlock
from lib.models.scaler import get_scaler
from lib.models.rotater import get_rotater
from lib.models.lifter import get_lifter_before, get_lifter_after
# use linear as the building block of simple discriminator
from lib.models.simple_model import LinearModelBefore, LinearModelAfter, Linear
from lib.utils.utils import rotate, rotate2

class SimpleDiscriminator(nn.Module):
    def __init__(self,
            linear_size=1024,
            num_stage=3,
            p_dropout=0.5, 
            spectral_norm=False, 
            use_bn=True):
        super(SimpleDiscriminator, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.use_spectral_norm = spectral_norm
        self.use_bn = use_bn

        # 2d joints
        self.input_size =  17 * 2
        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        if self.use_spectral_norm: 
            self.w1 = nn.utils.spectral_norm(self.w1)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout, spectral_norm=spectral_norm, use_bn=use_bn))
        self.linear_stages = nn.ModuleList(self.linear_stages)
        self.out = nn.Linear(self.linear_size, 1)
        if self.use_spectral_norm: 
            self.out = nn.utils.spectral_norm(self.out)

    def forward(self, joints_2d):
        joints_2d = joints_2d.contiguous()
        joints_2d = joints_2d.view(joints_2d.size(0), -1)
        x = self.w1(joints_2d)
        if self.use_bn:
            x = self.batch_norm1(x)
        for l in range(self.num_stage):
            x = self.linear_stages[l](x)
        out = self.out(x)

        return out 


class Discriminator(nn.Module): 
    def __init__(self, cfg, is_temp=False):
        super(Discriminator, self).__init__()
        num_feats, num_xyz = 2, 3
        num_joints = cfg.DATA.NUM_JOINTS
        num_channels = cfg.NETWORK.NUM_CHANNELS 
        num_res_blocks = cfg.NETWORK.DIS_RES_BLOCKS if not is_temp else cfg.NETWORK.DIS_TEMP_RES_BLOCKS
        dropout = cfg.NETWORK.DROPOUT 
        use_bn = cfg.NETWORK.DIS_USE_BN
        # whether to use spectral normalization for the discriminator
        use_spectral_norm = cfg.NETWORK.DIS_USE_SPECTRAL_NORM
        bn_track = cfg.NETWORK.BN_TRACK
        input_size = num_feats * num_joints if not is_temp else num_feats*2*num_joints
        self.num_xyz = num_xyz
        self.num_res_blocks = num_res_blocks
        self.pre = nn.ModuleList()
        if use_spectral_norm: 
            self.pre.append(nn.utils.spectral_norm(nn.Linear(input_size, num_channels)))
        else: 
            self.pre.append(nn.Linear(input_size, num_channels))
        if use_bn: 
            self.pre.append(nn.BatchNorm1d(num_channels, track_running_stats=bn_track))
        self.pre.append(nn.ReLU(inplace=True)); self.pre.append(nn.Dropout(dropout))
        self.pre = nn.Sequential(*self.pre)

        self.blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.blocks.append(ResBlock(num_channels, input_size, dropout=dropout, use_bn=use_bn, bn_track=bn_track, use_spectral_norm=use_spectral_norm))
        if use_spectral_norm:
            self.out = nn.utils.spectral_norm(nn.Linear(num_channels, 1))
        else: 
            self.out = nn.Linear(num_channels, 1)

    def forward(self, joints_2d):
        joints_2d = joints_2d.contiguous()
        joints_2d = joints_2d.view(joints_2d.size(0), -1)
        x = self.pre(joints_2d)
        for block in self.blocks:
            x = block(x, joints_2d)
        out = self.out(x)
        return out 


class PoseModel(nn.Module):
    def __init__(self, cfg):
        super(PoseModel, self).__init__()
        num_joints = cfg.DATA.NUM_JOINTS
        dropout = cfg.NETWORK.DROPOUT 
        bn_track = cfg.NETWORK.BN_TRACK
        num_channels = cfg.NETWORK.NUM_CHANNELS
        lifter_res_blocks = cfg.NETWORK.LIFTER_RES_BLOCKS 
        depth_estimator_res_blocks = cfg.NETWORK.DEPTH_ESTIMATOR_RES_BLOCKS
        self.is_15joints = num_joints == 15; self.num_joints = num_joints
        self.use_scaler = cfg.TRAIN.USE_SCALER
        self.use_rotater = cfg.TRAIN.USE_ROTATER
        self.c = cfg.TRAIN.CAMERA_SKELETON_DISTANCE
        self.use_new_rot = cfg.TRAIN.USE_NEW_ROT
        self.bound_azim = cfg.TRAIN.BOUND_AZIM 
        self.bound_elev = cfg.TRAIN.BOUND_ELEV
        self.num_frames = cfg.DATA.NUM_FRAMES
        self.frame_interval = cfg.DATA.FRAME_INTERVAL
        self.multi_new_temp = cfg.TRAIN.MULTI_NEW_TEMP
        self.use_simple_model = cfg.NETWORK.USE_SIMPLE_MODEL 

        if self.use_simple_model: 
            self.lifter_before = LinearModelBefore()
            self.lifter_after = LinearModelAfter()
        else: 
            self.lifter_before = get_lifter_before(num_joints, num_feats=2, num_channels=num_channels, num_res_blocks=lifter_res_blocks, bn_track=bn_track, dropout=dropout)
            self.lifter_after = get_lifter_after(num_joints, num_feats=2, num_xyz=1, num_channels=num_channels, is_generic_baseline=cfg.TRAIN.GENERIC_BASELINE)
        self.is_euler = cfg.NETWORK.ROTATER_PRE_EULER
        self.learn_symmetry = cfg.TRAIN.LEARN_SYMMETRY
        self.scaler_input_size = cfg.NETWORK.SCALER_INPUT_SIZE
        self.is_mpi = (cfg.DATA.DATASET_NAME == "mpi")
        self.is_surreal = (cfg.DATA.DATASET_NAME == 'surreal')
        self.is_generic_baseline = cfg.TRAIN.GENERIC_BASELINE
        self.scale_on_3d = cfg.TRAIN.SCALE_ON_3D
        if self.use_scaler: 
            self.scaler = get_scaler(cfg)
        if self.use_rotater: 
            self.rotater = get_rotater(cfg)
            self.rotate = rotate if self.is_euler else rotate2

    def _project(self, joints_3d):
        if self.is_generic_baseline:
            return joints_3d[..., :2]

        if self.is_surreal: 
            joints_2d = torch.clamp(joints_3d[..., :2] / joints_3d[..., 2:], min=-0.25, max=0.25)
        elif not self.is_mpi:
            joints_2d = torch.clamp(joints_3d[..., :2] / joints_3d[..., 2:], min=-0.2, max=0.2)
        else:
            joints_2d = torch.clamp(joints_3d[..., :2] / joints_3d[..., 2:], min=-0.35, max=0.35)
        return joints_2d

    def _compute_3d(self, joints_2d, depths):
        # joints_2d: (N * J * 2), depths: (N * J * 1)
        if depths.dim() < joints_2d.dim(): 
            depths = depths.unsqueeze(-1)
        ones = torch.ones_like(depths)
        depths = torch.max(ones, depths + self.c)
        if self.is_generic_baseline:
            joints_3d = torch.cat((joints_2d, depths), dim=-1)
        else:
            joints_3d = torch.cat((depths * joints_2d, depths), dim=-1)
        return joints_3d

    @staticmethod
    def _get_extra_info(joints_2d):
        # joints_2d expected to be (N * J * 2)
        ws = joints_2d[..., 0].max(dim=1, keepdims=True)[0] - joints_2d[..., 0].min(dim=1, keepdims=True)[0]
        hs = joints_2d[..., 1].max(dim=1, keepdims=True)[0] - joints_2d[..., 1].min(dim=1, keepdims=True)[0]
        # output shape should be (N, )
        return hs, ws

    def forward(self, joints_in, rot=None, is_train=True, is_diff=False):
        # first we get additional infomation from the 2d inputs
        if self.use_scaler: 
            bs = joints_in.size(0)
            hs, ws = self._get_extra_info(joints_in)
            hws = (hs + ws) / 2
            if self.scaler_input_size == 1:
                scaler_inputs = hws.view(bs, -1)
            elif self.scaler_input_size == 34: 
                scaler_inputs = joints_in.view(bs, -1)
            elif self.scaler_input_size == 35: 
                scaler_inputs = torch.cat([joints_in.view(bs, -1), hws], dim=1)
            elif self.scaler_input_size == 37:
                scaler_inputs = torch.cat([joints_in.view(bs, -1), hs, ws, hws], dim=1)
            else: 
                raise NotImplementedError("Can not recognize {} for scaler's input size".format(self.scaler_input_size))
            # output scales's shape shall be (N, 1)
            if self.learn_symmetry:
                scale_mids = self.scaler(scaler_inputs)
                scales = 2 * scale_mids - hws
            else: 
                scales = self.scaler(scaler_inputs)
                scale_mids = (scales + hws) / 2
            # multiply the 2d inputs 
            if not self.scale_on_3d:
                joints_in = joints_in * scales.view(-1, 1, 1)
            scaled_2d = None
        else: 
            scale_mids = None
            scales = None 
            scaled_2d = None

        if not is_train: 
            center_joints_in = joints_in
            out_features, out_joints_sc = [], []
            features, joints_sc = self.lifter_before(joints_in)

            depths = self.lifter_after(features, joints_sc)
            if depths.size(1) == self.num_joints:
                estimated_joints_3d = self._compute_3d(center_joints_in, depths)
            else: 
                estimated_joints_3d = depths.view(depths.size(0), self.num_joints, 3)
            if self.scale_on_3d:
                estimated_joints_3d = estimated_joints_3d * scales.view(-1, 1, 1)

            if self.use_rotater: 
                rot_out = self.rotater(estimated_joints_3d)
                estimated_joints_3d = self.rotate(estimated_joints_3d, rot_out)
            return estimated_joints_3d, scales, scale_mids

        rot_y, rot_x, rot_z = rot[:, 0:1], rot[:, 1:2], rot[:, 2:3]

        out_features, out_joints_sc = [], []
        first_half_feature, second_half_feature = None, None
        # the center frame and its joints
        # if use multi new tmep, joints_in's shape will always be (N * 1 * J * 3)
        features, joints_sc = self.lifter_before(joints_in)

        depths = self.lifter_after(features, joints_sc)
        if depths.size(1) == self.num_joints:
            estimated_joints_3d = self._compute_3d(joints_in, depths)
        else: 
            estimated_joints_3d = depths.view(depths.size(0), -1, 3)

        if self.scale_on_3d: 
            estimated_joints_3d = estimated_joints_3d * scales.view(-1, 1, 1)

        # this will be used in inverse transformation later
        root_3d = estimated_joints_3d[:, 13:14] if self.num_joints == 17 else estimated_joints_3d[:, 12:13]

        # transfrom
        transformed_joints_3d = transform_3d_v2(estimated_joints_3d, rot_y, rot_x, self.c, False, rot_z=rot_z, use_new_rot=self.use_new_rot, is_mpi=self.is_mpi)
        projected_joints_2d = self._project(transformed_joints_3d)
        projected_joints_2d = projected_joints_2d + 1e-5

        # reconstruct
        recon_features, recon_joints_sc = self.lifter_before(projected_joints_2d)
        recon_depths = self.lifter_after(recon_features, recon_joints_sc)
        if recon_depths.size(1) == self.num_joints:
            recon_joints_3d = self._compute_3d(projected_joints_2d.detach(), recon_depths)
        else: 
            recon_joints_3d = recon_depths.view(recon_depths.size(0), -1, 3)

        # transfrom inverse
        recovered_joints_3d = transform_3d_v2(recon_joints_3d, -rot_y, -rot_x, self.c, True, rot_z=-rot_z if rot_z is not None else None)
        recovered_joints_3d += root_3d
        recovered_joints_2d = self._project(recovered_joints_3d)
        return (first_half_feature, second_half_feature), estimated_joints_3d, \
                transformed_joints_3d, projected_joints_2d, recon_joints_3d, recovered_joints_3d, recovered_joints_2d, scale_mids, scales

def get_pose_model(cfg, model_path=None):
    pose_model = PoseModel(cfg)
    if model_path is not None: 
        state_dict = torch.load(model_path)['state_dict']
        pose_model.load_state_dict(state_dict)
    return pose_model

def get_discriminator(cfg, is_temp=False, model_path=None):
    if cfg.NETWORK.USE_SIMPLE_MODEL:
        discriminator = SimpleDiscriminator(spectral_norm=cfg.NETWORK.DIS_USE_SPECTRAL_NORM, use_bn=cfg.NETWORK.DIS_USE_BN)
    else:
        discriminator = Discriminator(cfg, is_temp=is_temp)
    if model_path is None and not cfg.NETWORK.USE_SIMPLE_MODEL:
        init_weights(discriminator)
    elif model_path is not None:
        state_dict = torch.load(model_path)['state_dict']
        discriminator.load_state_dict(state_dict)
    return discriminator
