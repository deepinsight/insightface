#!/usr/bin/env python3
# coding=utf-8

import os
import time
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
import warnings 
warnings.filterwarnings('ignore')

pairs = [(0, 1), (1, 2), (2, 13), (3, 13), (3, 4), (4, 5), (6, 7), (7, 8), (8, 12), (9, 10), (9, 12), 
        (10, 11), (12, 14), (12, 15), (13, 14), (15, 16)]
pairs_left = [(3, 13), (3, 4), (4, 5), (9, 10), (9, 12), (10, 11)]
pairs_right = [(0, 1), (1, 2), (2, 13), (6, 7), (7, 8), (8, 12)]
radius = 0.75
white = (1.0, 1.0, 1.0, 1.0)

def vis_joints(joints, factor=0.6451607):
    # joints: keys indicate the title(2d/3d), values contain the joints in numpy format
    for k, v in joints.items(): 
        if '3d' in k and 'gt' not in k:
            # back to original scale
            joints[k] = (joints[k] - joints[k][:, 13:14].copy()) * factor
    titles_2d = [item for item in joints if '2d' in item]
    joints_2d = [joints[title] for title in titles_2d]
    titles_3d = [item for item in joints if '3d' in item]
    joints_3d = [joints[title] for title in titles_3d]
    all_titles = titles_2d + titles_3d
    all_joints = joints_2d + joints_3d
    rows, cols = min(2, all_joints[0].shape[0]), len(all_joints)

    figsize = (4 * cols, 4 * rows)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    row = 0

    for idx in range(rows * cols):
        title, joints = all_titles[idx % cols], all_joints[idx % cols]
        if idx % cols == 0 and idx > 0:
            row += 1
        joints = joints[row]
        if '2d' in title:
            ax = fig.add_subplot(rows, cols, idx+1)
        else:
            ax = fig.add_subplot(rows, cols, idx+1, projection='3d')
        ax.set_title(title)
        for pair in pairs:
            if pair in pairs_left: 
                color = 'blue'
            elif pair in pairs_right: 
                color = 'green'
            else:
                color = 'orange'
            i, j = pair
            if '2d' in title: 
                ax.plot([joints[i, 0], joints[j, 0]], [-joints[i, 1], -joints[j, 1]], c=color)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_axis_off()
            else:
                ax.plot([joints[i, 2], joints[j, 2]], [joints[i, 0], joints[j, 0]], [-joints[i, 1], -joints[j, 1]], c=color)
                xroot, yroot, zroot = joints[13, 0], -joints[13, 1], joints[13, 2]
                ax.set_ylim3d([-radius+xroot, radius+xroot])
                ax.set_zlim3d([-radius+yroot, radius+yroot])
                ax.set_xlim3d([-radius+zroot, radius+zroot])
                ax.w_xaxis.set_pane_color(white)
                ax.w_yaxis.set_pane_color(white)

                ax.w_xaxis.line.set_color(white)
                ax.w_yaxis.line.set_color(white)
                ax.w_zaxis.line.set_color(white)
                ax.get_xaxis().set_ticklabels([])
                ax.get_yaxis().set_ticklabels([])
                ax.set_zticklabels([])
    
    name = str(time.time()).replace('.', '0') + ".png"
    plt.savefig(name)
    img = imread(name)
    os.remove(name)
    return img

def plot_scalemid_dist(scale_mids_pre, scale_mids_gt): 
    assert isinstance(scale_mids_pre, list) and isinstance(scale_mids_gt, list)
    scale_mids_pre, scale_mids_gt = np.array(scale_mids_pre), np.array(scale_mids_gt)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    mean, std = scale_mids_pre.mean(), scale_mids_pre.std()
    sns.distplot(scale_mids_pre)
    plt.xlabel('scale_mids', fontsize=12)
    plt.ylabel('prob', fontsize=12)
    plt.title("Pre: mean/std: {:.3f}/{:.3f}".format(mean, std))
    plt.subplot(122)
    mean, std = scale_mids_gt.mean(), scale_mids_gt.std()
    sns.distplot(scale_mids_gt)
    plt.xlabel('scale_mids', fontsize=12)
    plt.ylabel('prob', fontsize=12)
    plt.title("GT: mean/std: {:.3f}/{:.3f}".format(mean, std))
    name = str(time.time()).replace('.', '0') + ".png"
    plt.savefig(name)
    img = imread(name)
    os.remove(name)
    return img

def plot_scalemid_seq_dist(pre, gt, seqnames, num_seq=3):
    pre, gt = np.array(pre).reshape(-1), np.array(gt).reshape(-1)
    seqnames = np.array(seqnames)
    unique_seqnames = np.unique(seqnames).tolist()
    random.shuffle(unique_seqnames)
    fig, axes = plt.subplots(1, num_seq, figsize=(6*num_seq, 6))
    for idx, (uname, ax) in enumerate(zip(unique_seqnames, axes)):
        if idx == num_seq: 
            break
        mask = np.nonzero(seqnames == uname)[0]
        pre_ = pre[mask]; gt_ = gt[mask]
        sns.distplot(pre_, ax=ax, label='pre')
        sns.distplot(gt_, ax=ax, label='gt')
        ax.set_xlabel("scale_mid", fontsize=11)
        ax.set_ylabel("prob", fontsize=11)
        ax.set_title(f"sequence {idx}")
        ax.legend()
    name = str(time.time()).replace('.', '0') + ".png"
    plt.savefig(name)
    img = imread(name)
    os.remove(name)
    return img
    


