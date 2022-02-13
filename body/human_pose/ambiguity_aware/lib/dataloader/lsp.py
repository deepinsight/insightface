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
from scipy.io import loadmat

class LSPDataset(Dataset): 
    def __init__(self): 
        super(LSPDataset, self).__init__()
        filename = "../data/joints.mat"
        self.joints = loadmat(filename)['joints']
        self.joints = np.transpose(self.joints, (2, 1, 0))
        # 1. right ankle 2. right knee  3. right hip 
        # 4. left hip    5. left knee   6. left ankle 
        # 7. right wrist 8. right elbow 9. right shoulder 
        # 10.left shoulder 11. left elbow 12. left wrist 
        # 13. neck # 14. headtop 
        self.original_joints = self.joints.copy()
        self.joints[..., 0] = (self.joints[..., 0] - 55.0) / 55.0
        self.joints[..., 1] = (self.joints[..., 1] - 90.0) / 90.0
        thorax = (self.joints[:, 8:9] + self.joints[:, 9:10]) / 2
        pelvis = (self.joints[:, 2:3] + self.joints[:, 3:4]) / 2
        spine = (thorax + pelvis) / 2
        self.joints = np.concatenate((self.joints, thorax, pelvis, spine), axis=1)
        lsp2h36m_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 12, 13]
        self.joints = self.joints[:, lsp2h36m_indices].astype(np.float32)[..., :2]
        self.joints = self.joints - self.joints[:, 13:14]
        self.joints[:, 13:14] = 1e-5

        factor = np.linalg.norm(self.joints[:, 13:14] - self.joints[:, -1:], axis=2).mean()
        self.joints = self.joints / factor / 10.0

        thorax = (self.original_joints[:, 8:9] + self.original_joints[:, 9:10]) / 2
        pelvis = (self.original_joints[:, 2:3] + self.original_joints[:, 3:4]) / 2
        spine = (thorax + pelvis) / 2
        self.original_joints = np.concatenate((self.original_joints, thorax, pelvis, spine), axis=1)
        self.original_joints = self.original_joints[:, lsp2h36m_indices].astype(np.float32)[..., :2]
        for index in range(len(self.joints)): 
            if index + 1 not in [1003, 1120, 1262, 1273, 1312, 1379, 1639, 1723, 1991, 209, 387, 879]: 
                continue
            with open("demo_input/" + str(index + 1) + ".pkl", "wb") as f: 
                pkl.dump({'joints_2d': self.joints[index], "original_joints_2d": self.original_joints[index]}, f)

    def __getitem__(self, index): 
        return self.joints[index], self.original_joints[index]

    def __len__(self): 
        return len(self.joints)

