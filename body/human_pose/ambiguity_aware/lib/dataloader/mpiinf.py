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

class MPIINFDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.frame_interval = config.DATA.FRAME_INTERVAL
        # for mpi dataset, we convert its order to match with h36m
        self.mpi2h36m = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7, 1, 14, 15, 16, 0]
        # randomization will lead to inferior performance
        self.scale_path = "../data/mpi_train_scales.pkl" if config.USE_GT else "../data/mpi_train_scales_pre.pkl"
        self.use_same_norm_2d = config.DATA.USE_SAME_NORM_2D
        self.use_same_norm_3d = config.DATA.USE_SAME_NORM_3D
        self.is_train = is_train
        self.data_path = config.DATA.TRAIN_PATH if self.is_train else config.DATA.VALID_PATH
        self.head_root_distance = 1 / config.TRAIN.CAMERA_SKELETON_DISTANCE
        # whether to use dataset adapted from k[MaÌ]inetics
        self.use_gt = config.USE_GT
        self.use_ideal_scale = config.DATA.USE_IDEAL_SCALE
        self.min_diff_dist = config.DATA.MIN_DIFF_DIST
        self.use_scaler = config.TRAIN.USE_SCALER
        self.bound_azim = float(config.TRAIN.BOUND_AZIM) # y axis rotation  
        self.bound_elev = float(config.TRAIN.BOUND_ELEV)
        self._load_data_set()

    def _load_data_set(self):
        if self.is_train:
            print('start loading mpiinf {} data.'.format("train" if self.is_train else "test"))
        key = "joint_2d_gt" if self.use_gt else "joint_2d_pre"
        fp = h5py.File(self.data_path, "r")
        self.kp2ds = np.array(fp[key])[:, self.mpi2h36m, :2]
        self.kp2ds[:, :, 0] = (self.kp2ds[..., 0] - 1024.0) / 1024.0
        self.kp2ds[:, :, 1] = (self.kp2ds[..., 1] - 1024.0) / 1024.0
        # self.kp2ds = np.maximum(np.minimum(self.kp2ds, 1.0), -1.0)
        # locate root at the origin 
        self.kp2ds = self.kp2ds - self.kp2ds[:, 13:14]
        self.kp2ds[:, 13] = 1e-5
        # imagenames will be used to sample frames 
        self.imagenames = [name.decode() for name in fp['imagename'][:]]
        if 'seqname' not in fp.keys():
            # first we close the already opened (read-only) h5 
            fp.close()
            print("Process corresponding dataset...")
            process_dataset_for_video(self.data_path, is_mpi=True)
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

        self.kp3ds = np.array(fp['joint_3d_gt'])[:, self.mpi2h36m, :3] / 1000.0
        # factor_3d = np.linalg.norm(self.kp3ds[:, -1] - self.kp3ds[:, 13], axis=1).mean())
        factor_filename = "../data/mpi_{}_factor_3d.pkl".format("train" if self.is_train else "test")
        if not self.use_same_norm_3d:
            factor_3d = (np.tile(np.linalg.norm(self.kp3ds[:, -1] - self.kp3ds[:, 13], axis=1).reshape(-1, 1, 1), (1, 17, 3)) + 1e-8)
            print(factor_3d.shape)
            with open(factor_filename, "wb") as f:
                pkl.dump(factor_3d, f)
        
        if osp.exists(self.scale_path):
            f = open(self.scale_path, "rb")
            self.scales = pkl.load(f)['scale']
            f.close()
        else:
            if self.use_scaler:
                pass
                # raise Warning("You haven't generated the computed scales, if you don't need to observe the scale error during training, \njust ignore this warning because it won't affect training.")
            self.scales = None

        if self.use_ideal_scale: 
            # scales computed from projection of 3d 
            f = open("../data/mpi_{}_scales.pkl".format("train" if self.is_train else "valid"), "rb")
            scales = pkl.load(f)
            f.close()
            self.kp2ds = self.kp2ds * scales

        fp.close()
        print('finished load mpiinf {} data, total {} samples'.format("train" if self.is_train else "test", \
            self.kp2ds.shape[0]))

        # generate the rotation factors 
        num_examples = self.kp2ds.shape[0]
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
        if self.scales is not None:
            self.scales = torch.from_numpy(self.scales).float()

    def get_seqnames(self):
        return self.sequence_names

    def __len__(self):
        return self.kp2ds.shape[0]

    def __getitem__(self, index):
        seq_len = self.sequence_lens[index]
        index_in_seq = self.indices_in_seq[index]
        kps_3d = self.kp3ds[index]
        rot = self.rotation[index]
        if not self.is_train: 
            kps_2d = self.kp2ds[index]
            # don't use
            diff1 = diff2 = self.kp2ds[index]
        else:
            kps_2d = self.kp2ds[index]
            if self.frame_interval + index < seq_len:
                diff1_index = index + self.frame_interval
            else: 
                diff1_index = index - self.frame_interval
            diff1 = self.kp2ds[diff1_index]

            diff_dist = np.random.randint(-index_in_seq, seq_len - index_in_seq)
            while abs(diff_dist) < self.min_diff_dist: 
                diff_dist = np.random.randint(-index_in_seq, seq_len - index_in_seq)
            diff2_index = index + diff_dist 
            diff2 = self.kp2ds[diff2_index]

        rot = self.rotation[index]
        # for valdiation, simply ignore scale
        if self.scales is None or not self.is_train: 
            scale = 0
        else:
            scale = self.scales[index]
        return kps_2d, kps_3d, rot, diff1, diff2, scale

