#!/usr/bin/env python3
# coding=utf-8

import _init_paths
import os 
import os.path as osp
import cv2
import numpy as np
import torch
import argparse 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.switch_backend('agg')
plt.ioff()

from tqdm import tqdm
from torch.utils.data import DataLoader
from lib.models.model import get_pose_model, get_discriminator
from lib.core.config import  config, update_config, update_dir
from lib.dataloader.lsp import LSPDataset 

image_root = "../data/lsp_images"
pairs = [(0, 1), (1, 2), (2, 13), (3, 13), (3, 4), (4, 5), (6, 7), (7, 8), (8, 12), (9, 10),(9, 12), (10, 11),(12, 14), (12, 15), (13, 14), (15, 16)]
pairs_left = [(3, 13), (3, 4), (4, 5), (9, 10), (9, 12), (10, 11)]
pairs_right = [(0, 1), (1, 2), (2, 13), (6, 7), (7, 8), (8, 12)]
colors = {
    'pink': np.array([197, 27, 125]),  # L lower leg
    'light_pink': np.array([233, 163, 201]),  # L upper leg
    'light_green': np.array([161, 215, 106]),  # L lower arm
    'green': np.array([77, 146, 33]),  # L upper arm
    'red': np.array([215, 48, 39]),  # head
    'light_red': np.array([252, 146, 114]),  # head
    'light_orange': np.array([252, 141, 89]),  # chest
    'purple': np.array([118, 42, 131]),  # R lower leg
    'light_purple': np.array([175, 141, 195]),  # R upper
    'light_blue': np.array([145, 191, 219]),  # R lower arm
    'blue': np.array([69, 117, 180]),  # R upper arm
    'gray': np.array([130, 130, 130]),  #
    'white': np.array([255, 255, 255]),  #
}

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='../cfg/h36m_gt_adv.yaml')
parser.add_argument('--pretrain', default='../models/adv.pth.tar')
# parser.add_argument('--cfg', default='../cfg/best_samenorm.yaml')
args = parser.parse_args()

update_config(args.cfg)

pose_model = get_pose_model(config)
print(pose_model)
# state_dict = torch.load("../output/model4118.pth.tar")['pose_model_state_dict']
assert osp.exists(args.pretrain), "Can not find pretrained model at {}".format(args.pretrain)
state_dict = torch.load(args.pretrain)['pose_model_state_dict']
pose_model.load_state_dict(state_dict)
pose_model.eval()
torch.set_grad_enabled(False)

lsp_dataset = LSPDataset()
lsp_loader = DataLoader(lsp_dataset, batch_size=32, num_workers=4, drop_last=False)

all_joints_2d = []
all_joints_3d_pre = []
for joints_2d, original_joints_2d in tqdm(lsp_loader): 
    joints_3d, _, _ = pose_model(joints_2d, is_train=False)
    all_joints_2d.append(original_joints_2d.numpy())
    all_joints_3d_pre.append(joints_3d.numpy())

all_joints_2d = np.concatenate(all_joints_2d, axis=0)
all_joints_3d_pre = np.concatenate(all_joints_3d_pre, axis=0)
print(all_joints_3d_pre.shape)


for idx, joints_3d_pre in tqdm(enumerate(all_joints_3d_pre)):
    joints_2d = all_joints_2d[idx]
    joints_3d_pre = joints_3d_pre - joints_3d_pre[13:14]
    image_path = osp.join(image_root, "im%04d.jpg" % (idx + 1))
    print(image_path)
    image = cv2.imread(image_path)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=10., azim=45.)

    for pair in pairs: 
        i, j = pair
        if pair in pairs_left: 
            color = "blue"
            cv_color = (255, 0, 0)
        elif pair in pairs_right: 
            color = "green"
            cv_color = (0, 255, 0)
        else: 
            color = "darkorange"
            cv_color = (89, 141, 252)
        x1, y1 = joints_2d[i].astype(np.int32)
        x2, y2 = joints_2d[j].astype(np.int32)
        
        cv2.line(image, (x1, y1), (x2, y2), cv_color, 2)
        x1, y1, z1 = joints_3d_pre[i]
        x2, y2, z2 = joints_3d_pre[j]
        ax.plot([z1, z2], [x1, x2], [-y1, -y2], c=color, linewidth=3)

    image = image[::-1, :, ::-1].copy().astype(np.float32) / 255.
    r = 0.95
    xroot = yroot = zroot = 0.
    # radius = max(4, (np.mean(image.shape[:2]) * 0.01).astype(int))
    radius = 0.75
    xx = np.linspace(-r * radius + xroot, r * radius + xroot, image.shape[1])
    yy = np.linspace(-r * radius + yroot, r * radius + yroot, image.shape[0])
    xx, yy = np.meshgrid(xx, yy)
    zz = np.ones_like(xx) * (-3.2* radius + zroot)
    ax.plot_surface(zz, xx, yy, rstride=1, cstride=1, facecolors=image, shade=False)
    ax.set_xlabel('Z', fontsize=13)
    ax.set_ylabel("X", fontsize=13)
    ax.set_zlabel("Y", fontsize=13)
    ax.set_ylim3d([-radius+xroot, radius+xroot])
    ax.set_zlim3d([-radius+yroot, radius+yroot])
    ax.set_xlim3d([-2.5 * radius+zroot, radius+zroot])
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.set_zticklabels([])

    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)

    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)

    plt.savefig("lsp_vis/{}.png".format(idx+1))
    plt.close()
