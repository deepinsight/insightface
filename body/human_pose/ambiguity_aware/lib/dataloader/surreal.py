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
from lib.utils.misc import process_dataset_for_video

class SurrealDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.is_train = is_train
        self.frame_interval = config.DATA.FRAME_INTERVAL
        # randomization will lead to inferior performance
        # since diff will only be used when training 
        self.data_path = config.DATA.TRAIN_PATH if self.is_train else config.DATA.VALID_PATH
        self.use_same_norm_2d = config.DATA.USE_SAME_NORM_2D 
        self.use_same_norm_3d = config.DATA.USE_SAME_NORM_2D
        self.seed_set = False
        self.head_root_distance = 1 / config.TRAIN.CAMERA_SKELETON_DISTANCE
        # whether to use dataset adapted from k[MaÌ]inetics
        self.use_gt = config.USE_GT
        self.min_diff_dist = config.DATA.MIN_DIFF_DIST
        self.bound_azim = config.TRAIN.BOUND_AZIM # y axis rotation  
        self.bound_elev = config.TRAIN.BOUND_ELEV
        self._load_data_set()

    def get_seqnames(self): 
        return self.sequence_names

    def _load_data_set(self):
        # self.v3d_2d_to_ours = np.arange(17)
        if self.is_train:
            print('start loading surreal {} data.'.format("train" if self.is_train else "test"))
        key = "original_joint_2d_gt" if self.use_gt else "joint_2d_pre"
        assert self.use_gt
        fp = h5py.File(self.data_path, "r")
        self.kp2ds = np.array(fp[key])
        self.kp2ds[:, :, 0] = (self.kp2ds[:, :, 0] - 160.0) / 160.0
        self.kp2ds[:, :, 1] = (self.kp2ds[:, :, 1] - 160.0) / 160.0
        # locate root at the origin 
        # self.kp2ds[:, 12] = (self.kp2ds[:, 8] + self.kp2ds[:, 9]) / 2
        self.kp2ds = self.kp2ds - self.kp2ds[:, 13:14]
        self.kp2ds[:, 13] = 1e-5
        # imagenames will be used to sample frames 
        self.imagenames = [name.decode() for name in fp['imagename'][:]]
        if 'seqname' not in fp.keys(): 
            fp.close()
            print("Process corresponding dataset...")
            process_dataset_for_video(self.data_path, is_surreal=True)
            fp = h5py.File(self.data_path, "r")
        self.sequence_lens = np.array(fp['seqlen'])
        self.sequence_names = [name.decode() for name in fp['seqname'][:]]
        self.indices_in_seq = np.array(fp['index_in_seq'])

        # normlize again so that the mean distance of head and root is 1/c
        if not self.use_same_norm_2d:
            factor_gt = self.head_root_distance / (np.tile(np.linalg.norm(self.kp2ds[:, -1] - self.kp2ds[:, 13], axis=1).reshape(-1, 1, 1), (1, 17, 2)) + 1e-8)
        else:
            factor_gt = self.head_root_distance / np.linalg.norm(self.kp2ds[:, -1] - self.kp2ds[:, 13], axis=1).mean()
        self.kp2ds = self.kp2ds * factor_gt 

        self.kp3ds = np.array(fp['joint_3d_gt'])
        # self.kp3ds[:, 12] = (self.kp3ds[:, 8] + self.kp3ds[:, 9]) / 2
        factor_3d = np.linalg.norm(self.kp3ds[:, -1] - self.kp3ds[:, 13], axis=1).mean()
        factor_filename = "../data/surreal_{}_factor_3d.pkl".format("train" if self.is_train else "test")
        if not self.use_same_norm_3d and not osp.exists(factor_filename):
            factor_3d = (np.tile(np.linalg.norm(self.kp3ds[:, -1] - self.kp3ds[:, 13], axis=1).reshape(-1, 1, 1), (1, 17, 3)) + 1e-8)
            with open(factor_filename, "wb") as f:
                pkl.dump(factor_3d, f)

        fp.close()
        print('finished load surreal {} data, total {} samples'.format("train" if self.is_train else "test", \
            self.kp2ds.shape[0]))

        # get random diff1 
        self.diff_indices = []
        for index in range(self.kp2ds.shape[0]): 
            index_in_seq = self.indices_in_seq[index]
            seq_len = self.sequence_lens[index]
            if seq_len == 1: 
                diff1_index = index
            elif index_in_seq + self.frame_interval < seq_len: 
                diff_index = index + self.frame_interval
            else: 
                diff_index = index - self.frame_interval 
            self.diff_indices.append(diff_index)

        # generate the rotation factors 
        num_examples = self.kp2ds.shape[0]
        np.random.seed(2019)
        rotation_y = (2 * np.random.random_sample((num_examples, 1)) - 1) * self.bound_azim 
        rotation_x = (2 * np.random.random_sample((num_examples, 1)) - 1) * self.bound_elev 
        rotation_z = (2 * np.random.random_sample((num_examples, 1)) - 1) * self.bound_elev / 2
        rotation_1 = np.concatenate((rotation_y, rotation_x, rotation_z), axis=1)
        rotation_2 = rotation_1.copy()
        rotation_2[:, 0] = rotation_2[:, 0] + np.pi
        self.rotation = np.concatenate((rotation_1, rotation_2), axis=0)
        np.random.shuffle(self.rotation)
        self.rotation = torch.from_numpy(self.rotation).float()

        self.kp2ds = torch.from_numpy(self.kp2ds).float()
        self.kp3ds = torch.from_numpy(self.kp3ds).float()

    def __len__(self):
        return self.kp2ds.shape[0]

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)

        seq_len = self.sequence_lens[index]
        index_in_seq = self.indices_in_seq[index]
        kps_3d = self.kp3ds[index]
        rot = self.rotation[index]
        # index in its sequence
        kps_2d = self.kp2ds[index]
        kps_3d = self.kp3ds[index]
        diff1 = self.kp2ds[self.diff_indices[index]]
        if seq_len == 1: 
            diff_dist = 0
        else:
            diff_dist = np.random.randint(-index_in_seq, seq_len-index_in_seq)
            while abs(diff_dist) < self.min_diff_dist: 
                diff_dist = np.random.randint(-index_in_seq, seq_len-index_in_seq)
        diff2_index = index + diff_dist 
        diff2 = self.kp2ds[diff2_index]
        # current form: F * J * 2
        # we need to swap the last two axis, so that the item will be in the form J * 2 * F where
        # J is the number of keypoints and F is the number of frames
        # kps_2d = kps_2d.permute(1, 2, 0).contiguous()
        # diff = self.diff[all_indices].permute(1, 2, 0).contiguous()
        kps_2d = self.kp2ds[index]

        rot = self.rotation[index]
        # the flag will always be 1 when no extra data is used 
        # flag = self.flags[index]
        # for valdiation, simply ignore scale
        scale = 0

        return kps_2d, kps_3d, rot, diff1, diff2, scale

