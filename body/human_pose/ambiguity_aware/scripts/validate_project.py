#!/usr/bin/env python3
# coding=utf-8

import h5py 
import numpy as np
import pickle as pkl

v3d_to_ours = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 0, 7, 9, 10]

# filepath = "../data/h36m_valid_pred_3d.h5"
filepath = "../../unsupervised_mesh/data/h36m_valid_pred_3d_mesh.h5"
f = h5py.File(filepath, "r")
joints_2d_gt = np.array(f['joint_2d_gt'])
joints_3d_pre = np.array(f['joint_3d_pre'])
joints_3d_gt = np.array(f['joint_3d_gt'])
f.close()

factor_path = "../data/h36m_test_factor_3d.pkl"
f = open(factor_path, "rb")
factors = pkl.load(f)
f.close()
factors = 0.680019

# joints_2d_gt[:, :, 0] = (joints_2d_gt[:, :, 0] - 514.0435) / 500.0
# joints_2d_gt[:, :, 1] = (joints_2d_gt[:, :, 1] - 506.7003) / 500.0
joints_2d_gt = (joints_2d_gt - 112.0) / 112.0
root2d = joints_2d_gt[:, 13:14].copy()
joints_2d_gt = joints_2d_gt - root2d
joints_2d_gt[:, 13:14] = 1e-5

factor_2d = 1 / 10 / np.linalg.norm(joints_2d_gt[:, -1] - joints_2d_gt[:, 13], axis=1).mean()
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

# with open("../data/h36m_valid_scales.pkl", "wb") as f: 
#     pkl.dump({"scale": scales.reshape(-1), "scale_mid": scale_mids.reshape(-1)}, f)

err_gt = np.linalg.norm(project_gt_2d - joints_2d_gt, axis=-1).mean()
print("Projection GT error is: {:.4f}".format(err_gt))

# first descale, minus the root, and shift 
joints_3d_pre = joints_3d_pre / factors
root3d_pre = joints_3d_pre[:, 13:14].copy()
joints_3d_pre = joints_3d_pre - root3d_pre + shift 
project_pre_2d = joints_3d_pre[..., :2] / joints_3d_pre[..., 2:]
err_pre = np.linalg.norm(project_pre_2d - joints_2d_gt, axis=-1).mean()
print("Projection PRE error is: {:.4f}".format(err_pre))

