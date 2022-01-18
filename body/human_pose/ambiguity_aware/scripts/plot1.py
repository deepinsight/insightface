#!/usr/bin/env python3
# coding=utf-8

import os
import cv2
import random
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
plt.switch_backend('agg')
plt.ioff()

import h5py
from tqdm import trange
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seq_num', type=int, default=1, help='Specify the number of sequences to render')
parser.add_argument('--save_dir', type=str, default="../vis/", help='Specify the directory the save the visualization')
parser.add_argument('--in_filename', type=str, default= "../data/h36m_valid_pred_3d.h5", help="Speicfy the dataset to load from")
args = parser.parse_args()
seq_num = args.seq_num 
save_dir = args.save_dir
in_filename = args.in_filename
os.makedirs(save_dir, exist_ok=True)

v3d_to_ours = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 0, 7, 9, 10]
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
jcolors = [
    'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink',
    'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue',
    'purple', 'purple', 'red', 'green', 'green', 'white', 'white'
]
ecolors = {
    0: 'light_pink',
    1: 'light_pink',
    2: 'light_pink',
    3: 'pink',
    4: 'pink',
    5: 'pink',
    6: 'light_blue',
    7: 'light_blue',
    8: 'light_blue',
    9: 'blue',
    10: 'blue',
    11: 'blue',
    12: 'purple',
    13: 'light_green',
    14: 'light_green',
    15: 'purple'
}

root = "/yzbdata/MeshTrack/Data/HMR/Human/Subject/"
image_root = osp.join(root, "datapre_all")

in_filename = "../data/h36m_valid_pred_3d4118.h5"
in_filename_ssadv = "../data/h36m_valid_pred_3dssadv.h5"

print("Read from", in_filename)
f = h5py.File(in_filename, "r")
imagenames = [name.decode() for name in f['imagename'][:]]
# 2d joints in the order of v3d convention
# poses2d = np.array(f['joint_2d_gt'])[:, v3d_to_ours]
poses2d = np.array(f['joint_2d_gt'])
poses3d = np.array(f['joint_3d_pre'])
poses3d_gt = np.array(f['joint_3d_gt'])
poses3d_gt = poses3d_gt - poses3d_gt[:, 13:14]
f.close()

f = h5py.File(in_filename_ssadv, "r")
poses3d_ssadv = np.array(f['joint_3d_pre'])
f.close()

t = trange(0, len(imagenames))
processed_video_names = []

def plot_skeleton_2d(all_frames, joints_2d): 
    out_frames = []
    radius = max(4, (np.mean(all_frames[0].shape[:2]) * 0.01).astype(int))
    for idx in range(len(all_frames)): 
        for pair in pairs: 
            i, j = pair 
            pt1, pt2 = joints_2d[idx, i], joints_2d[idx, j] 
            x11, y11 = pt1 
            x22, y22 = pt2 
            if pair in pairs_left: 
                color = (205, 0, 0)
            elif pair in pairs_right: 
                color = (0, 205, 0)
            else: 
                color = (0, 165, 255)
            cv2.line(all_frames[idx], (int(x11), int(y11)), (int(x22), int(y22)), color, radius-2)
        
def get_xxyys(names): 
    xxyys = []
    # should be subject, action, camera
    splits = names[0].split('/')
    video_name = '/'.join(splits[:-1])
    part_label_path = osp.join(root, splits[0], 'MySegmentsMat', 'PartLabels',
                splits[1] + ("cam" + splits[2]).replace('cam0', '.54138969').replace('cam2','.58860488').replace('cam1', '.55011271').replace('cam3', '.60457274') + ".mat")
    f = h5py.File(part_label_path, "r")
    for idx, name in enumerate(names): 
        partmask = f[f['Feat'][idx*30, 0]][()].T 
        yp, xp = np.where(partmask != 0)
        xmin, xmax = np.min(xp), np.max(xp) + 1 
        ymin, ymax = np.min(yp), np.max(yp) + 1 
        xxyys.append((xmin, xmax, ymin, ymax))
    f.close()
    return xxyys

def crop_image(all_frames, xxyys, scale_factor=0.25): 
    out_frames = []
    for frame, xxyy in zip(all_frames, xxyys): 
        h, w = frame.shape[:2]
        xmin, xmax, ymin, ymax = xxyy 
        xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
        l = max(xmax - xmin, ymax - ymin)
        xmin, xmax = max(0, xc - l/2), min(w, xc + l / 2)
        ymin, ymax = max(0, yc - l/2), min(h, yc + l / 2)
        xmin, xmax = int(xmin), int(xmax)
        ymin, ymax = int(ymin), int(ymax)
        frame = frame[ymin:ymax, xmin:xmax, :].copy()
        frame = cv2.resize(frame, (int(scale_factor * w), int(scale_factor * h)))
        frame = frame[::-1, :, ::-1] / 255
        out_frames.append(frame)
    return out_frames

for imageid in t:
    name = imagenames[imageid]
    splits = name.split('/')
    video_name = '/'.join(splits[:3])
    if len(processed_video_names) == seq_num: 
        print("Finished! Rendered {} sequences, saved to {}".format(seq_num, save_dir))
        break
    if video_name in processed_video_names:
        continue 
    else:
        processed_video_names.append(video_name)
    print(video_name)
    recs = [(idx, name) for idx, name in enumerate(imagenames) if video_name in name]
    # downsample 
    recs = recs[::30]
    # cand_list = [x*5 for x in [440, 565, 770]]
    # cand_list = [200, 250, 300, 350, 400, 450, 500, 520, 550, 590, 620, 660, 700, 740, 770, 800, 830, 845]
    # recs = list(filter(lambda x: x[0] in cand_list,  recs))
    # recs = list(filter(lambda x: x[0] in [65*5, 100*5, 905*5, 1160*5], recs))
    recs = sorted(recs, key=lambda x: int(x[1].split('/')[-1]))
    names_in_video = [rec[1] for rec in recs]
    indices_in_video = [rec[0] for rec in recs]
    path_format = osp.join(image_root, splits[0], splits[1].replace(' ', '_'), "cam" + splits[2], "{:06d}.jpg")
    poses3d_in_video = poses3d[indices_in_video]
    poses2d_in_video = poses2d[indices_in_video]
    poses3d_ssadv_in_video = poses3d_ssadv[indices_in_video]
    poses3d_gt_in_video = poses3d_gt[indices_in_video]
    all_frames = [cv2.imread(path_format.format(int(name.split('/')[-1])+1)) for name in names_in_video]
    print("Ploting 2d skeleton...")
    plot_skeleton_2d(all_frames, poses2d_in_video)
    # scale_factor = 0.25
    # all_frames = [cv2.resize(frame, (int(scale_factor * frame.shape[1]), int(scale_factor * frame.shape[0])))[::-1, :, ::-1] / 255 for frame in all_frames]
    print("Getting bounding boxes...")
    xxyys = get_xxyys(names_in_video)
    print("Cropping images...")
    all_frames = crop_image(all_frames, xxyys, scale_factor=0.2)
    print("Generating gifs...")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=10., azim=45.)
    lines_3d, lines_3d_gt = [], []
    lines_3d_ssadv = []
    radius = 0.75 
    initialized = False
    num_render = len(names_in_video)
    print(num_render, " frames to plot")

    def update_video(frame_idx):
        global initialized, lines_3d, lines_3d_gt, lines_3d_ssadv
        print("{}/{} ".format(frame_idx, num_render), end='\r')
        pose2d = poses2d_in_video[frame_idx]
        pose3d = poses3d_in_video[frame_idx]
        pose3d_ssadv = poses3d_ssadv_in_video[frame_idx]
        pose3d_gt = poses3d_gt_in_video[frame_idx]
        if not initialized:
            for idx, pair in enumerate(pairs):
                i, j = pair
                if pair in pairs_left: 
                    color = "blue"
                elif pair in pairs_right: 
                    color = "green"
                else: 
                    color = "darkorange"
                # pt1, pt2 = pose3d[i], pose3d[j]
                # x11, y11, z11 = pt1
                # x22, y22, z22 = pt2
                # lines_3d.append(ax.plot([z11, z22], [x11, x22], [-y11, -y22], c='red', linewidth=3, label="pre"))
                pt1, pt2 = pose3d_gt[i], pose3d_gt[j]
                x11, y11, z11 = pt1 
                x22, y22, z22 = pt2 
                lines_3d_gt.append(ax.plot([z11, z22], [x11, x22], [-y11, -y22], c=color, linewidth=3, label="gt"))
                # pt1, pt2 = pose3d_ssadv[i], pose3d_ssadv[j]
                # x11, y11, z11 = pt1 
                # x22, y22, z22 = pt2
                # lines_3d_ssadv.append(ax.plot([z11, z22], [x11, x22], [-y11, -y22], c="red", linewidth=3, label="ssadv"))
            initialized = True
        else:
            for idx, pair in enumerate(pairs):
                i, j = pair
                # pt1, pt2 = pose3d[i], pose3d[j]
                # x11, y11, z11 = pt1
                # x22, y22, z22 = pt2
                # lines_3d[idx][0].set_xdata([z11, z22])
                # lines_3d[idx][0].set_ydata([x11, x22])
                # lines_3d[idx][0].set_3d_properties([-y11, -y22])
                pt1, pt2 = pose3d_gt[i], pose3d_gt[j]
                x11, y11, z11 = pt1
                x22, y22, z22 = pt2
                lines_3d_gt[idx][0].set_xdata([z11, z22])
                lines_3d_gt[idx][0].set_ydata([x11, x22])
                lines_3d_gt[idx][0].set_3d_properties([-y11, -y22])
                # pt1, pt2 = pose3d_ssadv[i], pose3d_ssadv[j]
                # x11, y11, z11 = pt1
                # x22, y22, z22 = pt2
                # lines_3d_ssadv[idx][0].set_xdata([z11, z22])
                # lines_3d_ssadv[idx][0].set_ydata([x11, x22])
                # lines_3d_ssadv[idx][0].set_3d_properties([-y11, -y22])

        xroot, yroot, zroot = pose3d_gt[13, 0], -pose3d_gt[13, 1], pose3d_gt[13, 2]
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

        r = 0.95
        # radius = max(4, (np.mean(all_frames[0].shape[:2]) * 0.01).astype(int))
        xx = np.linspace(-r * radius + xroot, r * radius + xroot, all_frames[frame_idx].shape[1])
        yy = np.linspace(-r * radius + yroot, r * radius + yroot, all_frames[frame_idx].shape[0])
        xx, yy = np.meshgrid(xx, yy)
        zz = np.ones_like(xx) * (-3.2* radius + zroot)
        ax.set_xlabel('Z', fontsize=13)
        ax.set_ylabel("X", fontsize=13)
        ax.set_zlabel("Y", fontsize=13)
        ax.plot_surface(zz, xx, yy, rstride=1, cstride=1, facecolors=all_frames[frame_idx], shade=False)
        plt.savefig(osp.join(save_dir, f"{video_name.replace('/', '_')}_{frame_idx}.png"))

    for idx in range(len(names_in_video)): 
        update_video(idx)
    ani = animation.FuncAnimation(fig, update_video, range(len(names_in_video)), interval=20)
    save_name = name.replace('/', '_')
    ani.save(osp.join(save_dir, f"{save_name}.gif"), writer='imagemagick', fps=20)
    t.set_postfix(index=int(imageid))
