import sys
from torch.utils.data import Dataset, DataLoader
import os
import os.path as osp
import glob
import numpy as np
import random
import cv2
import pickle as pkl
import json
import h5py
import torch
import matplotlib.pyplot as plt
from lib.utils.misc import process_dataset_for_video, save_pickle, load_pickle


class Human36MDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.is_train = is_train
        self.data_path = config.DATA.TRAIN_PATH if is_train else config.DATA.VALID_PATH
        self.frame_interval = config.DATA.FRAME_INTERVAL
        self.num_frames = config.DATA.NUM_FRAMES
        assert self.num_frames % 2, f"Please use odd number of frames, current: {self.num_frames}"
        self.scale_path = osp.join("../data", "h36m_{}_scales{}".format("train" if is_train else "valid", ".pkl" if config.USE_GT else "_pre.pkl"))
        self.use_gt_scale = config.TRAIN.USE_GT_SCALE
        self.use_same_norm_2d, self.use_same_norm_3d = config.DATA.USE_SAME_NORM_2D, config.DATA.USE_SAME_NORM_3D
        self.seed_set = False
        self.head_root_distance = 1 / config.TRAIN.CAMERA_SKELETON_DISTANCE
        self.v3d_2d_to_ours = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 0, 7, 9, 10]
        # whether to use dataset adapted from k[MaÌ]inetics
        self.use_gt = config.USE_GT
        self.exp_tmc = config.DATA.EXP_TMC
        self.exp_tmc_start = config.DATA.EXP_TMC_START
        self.exp_tmc_deterministic = config.DATA.EXP_TMC_DETERMINISTIC 
        self.exp_tmc_interval = config.DATA.EXP_TMC_INTERVAL 
        self.min_diff_dist = config.DATA.MIN_DIFF_DIST
        self.bound_azim = config.TRAIN.BOUND_AZIM # y axis rotation  
        self.bound_elev = config.TRAIN.BOUND_ELEV
        self.online_rot = config.DATA.ONLINE_ROT
        self.is_generic_baseline = config.TRAIN.GENERIC_BASELINE
        self.is_15joints = config.DATA.NUM_JOINTS == 15
        self.map_to_15joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16]   # exclude thorax and spine 
        self._load_data_set()

    def _load_data_set(self):
        if self.is_train:
            print('start loading hum3.6m {} data.'.format("train" if self.is_train else "test"))
        key = "joint_2d_gt" if self.use_gt else "joint_2d_pre"
        fp = h5py.File(self.data_path, "r")

        self.kp2ds = np.array(fp[key])[:, self.v3d_2d_to_ours, :2]
        self.kp2ds[:, :, 0] = (self.kp2ds[:, :, 0] - 514.0435) / 500.0
        self.kp2ds[:, :, 1] = (self.kp2ds[:, :, 1] - 506.7003) / 500.0
        # locate root at the origin 
        self.kp2ds = self.kp2ds - self.kp2ds[:, 13:14]
        self.kp2ds[:, 13] = 1e-5
        # imagenames will be used to sample frames 
        self.imagenames = [name.decode() for name in fp['imagename'][:]]
        if 'seqname' not in fp.keys():
            # first we close the already opened (read-only) h5
            fp.close()
            print("Process corresponding dataset...")
            process_dataset_for_video(self.data_path)
            fp = h5py.File(self.data_path, "r")
        self.sequence_lens = np.array(fp['seqlen'])
        self.sequence_names = [name.decode() for name in fp['seqname'][:]]
        self.seqname2seqindex = {name: i for i, name in enumerate(np.unique(self.sequence_names))}
        self.seqindex2seqname = {i: name for name, i in self.seqname2seqindex.items()}
        with open("../data/seqindex2seqname.pkl", "wb") as f: 
            pkl.dump(self.seqindex2seqname, f)
        self.indices_in_seq = np.array(fp['index_in_seq'])

        # normlize again so that the mean distance of head and root is 1/c
        if not self.is_generic_baseline:
            if not self.use_same_norm_2d:
                factor_gt = self.head_root_distance / (np.tile(np.linalg.norm(self.kp2ds[:, -1] - self.kp2ds[:, 13], axis=1).reshape(-1, 1, 1), (1, 17, 2)) + 1e-8)
            else:
                factor_gt = self.head_root_distance / np.linalg.norm(self.kp2ds[:, -1] - self.kp2ds[:, 13], axis=1).mean()
            self.kp2ds = self.kp2ds * factor_gt 

        self.kp3ds = np.array(fp['joint_3d_gt'])[:, self.v3d_2d_to_ours, :3]
        factor_3d = np.linalg.norm(self.kp3ds[:, -1] - self.kp3ds[:, 13], axis=1).mean()
        factor_filename = "../data/h36m_{}_factor_3d.pkl".format("train" if self.is_train else "test")
        if not self.use_same_norm_3d and not osp.exists(factor_filename):
            factor_3d = (np.tile(np.linalg.norm(self.kp3ds[:, -1] - self.kp3ds[:, 13], axis=1).reshape(-1, 1, 1), (1, 17, 3)) + 1e-8)
            save_pickle(factor_3d, factor_filename)

        self.scales = load_pickle(self.scale_path)['scale'] if osp.exists(self.scale_path) else None
        if self.use_gt_scale:
            assert self.scales is not None, "Want to use ground-truth, you must calculate tht beforehand, check {}".format(self.scale_path)
            self.kp2ds = self.kp2ds * self.scales['scale'].reshape(-1, 1, 1)

        fp.close()
        print('finished load human36m {} data, total {} samples'.format("train" if self.is_train else "test", \
            self.kp2ds.shape[0]))

        # generate the rotation factors 
        num_examples = self.kp2ds.shape[0]
        np.random.seed(2019)
        self.bound_y = self.bound_azim;  self.bound_x = self.bound_elev;  self.bound_z = self.bound_elev / 2
        rotation_y = (2 * np.random.random_sample((num_examples, 1)) - 1) * self.bound_y
        rotation_x = (2 * np.random.random_sample((num_examples, 1)) - 1) * self.bound_x 
        rotation_z = (2 * np.random.random_sample((num_examples, 1)) - 1) * self.bound_z
        rotation_1 = np.concatenate((rotation_y, rotation_x, rotation_z), axis=1)
        rotation_2 = rotation_1.copy()
        rotation_2[:, 0] = rotation_2[:, 0] + np.pi
        self.rotation = np.concatenate((rotation_1, rotation_2), axis=0)
        np.random.shuffle(self.rotation)
        self.rotation = torch.from_numpy(self.rotation).float()

        self.kp2ds = torch.from_numpy(self.kp2ds).float()
        self.kp3ds = torch.from_numpy(self.kp3ds).float()
        if self.scales is not None:
            self.scales = torch.from_numpy(self.scales).float()

    def get_seqnames(self):
        return self.sequence_names

    def __len__(self):
        return self.kp2ds.shape[0]

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)

        kps_3d = self.kp3ds[index]
        seq_len = self.sequence_lens[index]
        index_in_seq = self.indices_in_seq[index]
        interval = int((self.num_frames - 1) / 2) * self.frame_interval
        start, end = index_in_seq - interval, index_in_seq + interval
        kps_3d = self.kp3ds[index]

        kps_2d = self.kp2ds[index]
        if index_in_seq + 5 < seq_len: 
            diff1_index = index + 5 
        else: 
            diff1_index = index - 5

        if self.frame_interval > 1:
            if index_in_seq + self.frame_interval < seq_len: 
                diff1_index = index + self.frame_interval
            else: 
                diff1_index = index - self.frame_interval


        if self.exp_tmc:
            if index_in_seq + (self.exp_tmc_start + self.exp_tmc_interval) < seq_len: 
                diff1_index = index + (np.random.randint(self.exp_tmc_start, self.exp_tmc_start + self.exp_tmc_interval) if not self.exp_tmc_deterministic else self.exp_tmc_interval)
            else: 
                diff1_index = index - (np.random.randint(self.exp_tmc_start, self.exp_tmc_start + self.exp_tmc_interval) if not self.exp_tmc_deterministic else self.exp_tmc_interval)
        diff1 = self.kp2ds[diff1_index]

        diff_dist = np.random.randint(-index_in_seq, seq_len - index_in_seq)
        while abs(diff_dist) < self.min_diff_dist: 
            diff_dist = np.random.randint(-index_in_seq, seq_len - index_in_seq)
        diff_index = index + diff_dist 
        diff2 = self.kp2ds[diff_index]

        if not self.online_rot:
            rot = self.rotation[index]
        else: 
            rot = np.random.rand(3, ) * np.array([self.bound_y, self.bound_x, self.bound_z])
            rot = torch.from_numpy(rot).float()
        # the flag will always be 1 when no extra data is used 
        # for valdiation, simply ignore scale
        if self.scales is not None and self.is_train: 
            scale = self.scales[index]
        else:
            scale = 0
        seqname = self.sequence_names[index]
        seqindex = self.seqname2seqindex[seqname]

        if self.is_15joints:
            kps_2d = kps_2d[self.map_to_15joints]
            kps_3d = kps_3d[self.map_to_15joints]
            diff1 = diff1[self.map_to_15joints]
            diff2 = diff2[self.map_to_15joints]

        return kps_2d, kps_3d, rot, diff1, diff2, scale

