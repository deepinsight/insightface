#!/usr/bin/env python3
# coding=utf-8

import h5py 
import numpy as np
import pickle as pkl
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train')
parser.add_argument('--prefix', default="mpi")
args = parser.parse_args()

prefix = args.prefix 
mode = args.mode

mpi2h36m = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7, 1, 14, 15, 16, 0] if prefix == "mpi" else list(range(17))
readpath = f"../data/{prefix}_{mode}_pred3.h5"
savepath = f"../data/mpi_{mode}_scales.pkl"

f = h5py.File(readpath, "r")
joints_2d_gt = np.array(f['joint_2d_gt'])[:, mpi2h36m]
# joints_3d_pre = np.array(f['joint_3d_pre'])
joints_3d_gt = np.array(f['joint_3d_gt'])[:, mpi2h36m] / 1000.0
f.close()

if prefix == "mpi":
    factors = 0.7577316 if mode == "valid" else 0.7286965902 
else: 
    factors = 0.680019 if mode == "valid" else 0.6451607

joints_2d_gt[:, :, 0] = (joints_2d_gt[:, :, 0] - 1024.0) / 1024.0
joints_2d_gt[:, :, 1] = (joints_2d_gt[:, :, 1] - 1024.0) / 1024.0
root2d = joints_2d_gt[:, 13:14].copy()
joints_2d_gt = joints_2d_gt - root2d
joints_2d_gt[:, 13:14] = 1e-5

# factor_2d = 1 / 10 / np.linalg.norm(joints_2d_gt[:, -1] - joints_2d_gt[:, 13], axis=1).reshape(-1, 1, 1)
factor_2d = 1 / 10 / np.linalg.norm(joints_2d_gt[:, -1] - joints_2d_gt[:, 13], axis=1).reshape(-1, 1, 1)
# scale the 2d joints
# joints_2d_gt = joints_2d_gt * factor_2d * factors[:, 0:1, 0:1]
joints_2d_gt = joints_2d_gt * factor_2d

# then we project the 3d joints 
# minus the root and shift to (0, 0, 10)
joints_3d_gt = joints_3d_gt - joints_3d_gt[:, 13:14].copy()
joints_3d_gt = joints_3d_gt / factors
shift = np.array([0, 0, 10]).reshape(1, 1, 3)
root3d_gt = joints_3d_gt[:, 13:14].copy()
joints_3d_gt = joints_3d_gt - root3d_gt + shift 

# project the 3d joints 
# N * J * 2
project_gt_2d = joints_3d_gt[..., :2] / joints_3d_gt[..., 2:]
x1_min, x1_max = joints_2d_gt[..., 0:1].min(axis=1, keepdims=True), joints_2d_gt[..., 0:1].max(axis=1, keepdims=True)
y1_min, y1_max = joints_2d_gt[..., 1:].min(axis=1, keepdims=True), joints_2d_gt[..., 1:].max(axis=1, keepdims=True)
x2_min, x2_max = project_gt_2d[..., 0:1].min(axis=1, keepdims=True), project_gt_2d[..., 0:1].max(axis=1, keepdims=True)
y2_min, y2_max = project_gt_2d[..., 1:].min(axis=1, keepdims=True), project_gt_2d[..., 1:].max(axis=1, keepdims=True)
scales = ((x2_max - x2_min) / (x1_max - x1_min) + (y2_max - y2_min) / (y1_max - y1_min)) / 2
heights, widths = y1_max - y1_min, x1_max - x1_min
scale_mids = (scales + (heights + widths) / 2) / 2
print("Mean/Std of scale mid: {:.3f}/{:.3f}".format(scale_mids.mean(), scale_mids.std()))

with open(savepath, "wb") as f: 
    pkl.dump({"scale": scales.reshape(-1), "scale_mid": scale_mids.reshape(-1)}, f)

err_gt = np.linalg.norm(project_gt_2d - joints_2d_gt, axis=-1).mean()
print("Projection GT error is: {:.4f}".format(err_gt))

# first descale, minus the root, and shift 
# joints_3d_pre = joints_3d_pre / factors
# root3d_pre = joints_3d_pre[:, 13:14].copy()
# joints_3d_pre = joints_3d_pre - root3d_pre + shift 
# project_pre_2d = joints_3d_pre[..., :2] / joints_3d_pre[..., 2:]
# err_pre = np.linalg.norm(project_pre_2d - joints_2d_gt, axis=-1).mean()
# print("Projection PRE error is: {:.4f}".format(err_pre))

