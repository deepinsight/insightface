import pickle
import numpy as np
import os
import os.path as osp
import glob
import argparse
import cv2
import time
import datetime
import pickle
import sklearn
import mxnet as mx
from utils.utils_config import get_config
from dataset import MXFaceDataset, Rt26dof

if __name__ == "__main__":
    cfg = get_config('configs/s1.py')
    cfg.task = 0
    save_path = os.path.join(cfg.cache_dir, 'train.meta')
    assert not osp.exists(save_path)
    dataset = MXFaceDataset(cfg, is_train=True, norm_6dof=False, degrees_6dof=True, local_rank=0)
    #dataset.transform = None
    print('total:', len(dataset))
    total = len(dataset)
    meta = np.zeros( (total, 3), dtype=np.float32 )
    for idx in range(total):
        #image, label_verts, label_6dof = dataset[idx]
        #img_raw, img_local, label_verts, label_Rt, tform = dataset[idx]
        img, label_verts, label_6dof, label_points2d, _, _ = dataset[idx]
        pose = label_6dof.numpy()[:3]
        print(idx, pose)
        meta[idx] = pose

    np.save(save_path, meta)

