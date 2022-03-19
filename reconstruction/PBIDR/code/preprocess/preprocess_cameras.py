import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import argparse
from glob import glob
import os
import sys
import pickle
sys.path.append('../code')
from scipy.spatial.transform import Rotation
import utils.general as utils


def get_Ps_from_Faces(R, T):
    Ps = []
    cam_locs = []

    intrinsics = np.concatenate([[4500.0], [0.0], [512.0], [0.0], [4500.0], [512.0], [0.0], [0.0], [1.0]], axis=0)
    intrinsics = np.reshape(intrinsics, [3, 3])

    projection = np.concatenate([[1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [0.0]], axis=0)
    projection = np.reshape(projection, [3, 4])

    I14 = np.concatenate([[0.0], [0.0], [0.0], [1.0]], axis=0)
    I14 = np.reshape(I14, [1, 4])

    for i in range(0, len(R)):
        R0 = R[i]
        T0 = T[i].reshape(3, 1)

        p = np.concatenate([np.concatenate([R[i].T, T[i].reshape(3, 1)], axis=1), I14], axis=0)
        P = intrinsics @ projection @ p
        P = P.astype(np.float64)

        camera_loc = -np.dot(R0, T0)
        cam_locs.append(camera_loc)
        Ps.append(P)

    return np.array(Ps)


def get_all_mask_points_white_bg(masks_dir):
    mask_paths = sorted(utils.glob_imgs(masks_dir))
    mask_points_all=[]
    mask_ims = []
    for path in mask_paths:
        img = mpimg.imread(path)
        cur_mask = img.max(axis=2) < 0.9
        mask_points = np.where(img.max(axis=2) < 0.9)
        xs = mask_points[1]
        ys = mask_points[0]
        mask_points_all.append(np.stack((xs,ys,np.ones_like(xs))).astype(float))
        mask_ims.append(cur_mask)
    return mask_points_all,np.array(mask_ims)


def get_normalization(source_dir):
    print('Preprocessing', source_dir)

    masks_dir= '{0}/mask'.format(source_dir)
    mask_points_all, masks_all = get_all_mask_points_white_bg(masks_dir)
    number_of_cameras = len(masks_all)
    R = np.load('{0}/R.npy'.format(source_dir))
    T = np.load('{0}/T.npy'.format(source_dir))
    Ps = get_Ps_from_Faces(R, T)
    normalization = np.eye(4).astype(np.float32)

    cameras_new={}
    for i in range(number_of_cameras):
        cameras_new['scale_mat_%d' % i] = normalization
        cameras_new['world_mat_%d' % i] = np.concatenate((Ps[i],np.array([[0,0,0,1.0]])),axis=0).astype(np.float32)

    np.savez('{0}/{1}.npz'.format(source_dir, "cameras"), **cameras_new)
    print(normalization)
    print('--------------------------------------------------------')

    if False: #for debugging
        for i in range(number_of_cameras):
            plt.figure()

            plt.imshow(mpimg.imread('%s/%03d.png' % ('{0}/mask'.format(source_dir), i+1)))
            xy = (Ps[i,:2, :] @ (np.concatenate((np.array(all_Xs), np.ones((len(all_Xs), 1))), axis=1).T)) / (
                        Ps[i,2, :] @ (np.concatenate((np.array(all_Xs), np.ones((len(all_Xs), 1))), axis=1).T))

            plt.plot(xy[0, :], xy[1, :], '*')
            plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_id', '-i', type=int, default=0, help='data source folder for preprocess')
    parser.add_argument('--dataset', '-d', type=str, default='Face', help='dataset dir')
    opt = parser.parse_args()

    SCAN_DIR = '../data/' + opt.dataset + '/scan' + str(opt.scan_id)
    get_normalization(SCAN_DIR)

    print('Done!')