#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author     :   Qingping Zheng
@Contact    :   qingpingzheng2014@gmail.com
@File       :   datasets.py
@Time       :   10/01/21 00:00 PM
@Desc       :   
@License    :   Licensed under the Apache License, Version 2.0 (the "License"); 
@Copyright  :   Copyright 2015 The Authors. All Rights Reserved.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import os.path as osp
import pickle
import random

import torch
import torchvision.transforms.functional as TF

from glob import glob
from typing import Tuple
from utils import transforms


class FaceDataSet(torch.utils.data.Dataset):
    """Face data set for model training and validating

    Examples:

        ./CelebAMask
            |---test
            |---train
                |---images
                    |---0.jpg
                    |---1.jpg
                |---labels
                    |---0.png
                    |---1.png
                |---edges
                    |---0.png
                    |---1.png
            |---valid
            |---label_names.txt
            |---test_list.txt
            |---train_list.txt
                |---images/0.jpg labels/0.png
                |---images/1.jpg labels/1.png
            |---valid_list.txt

    Args:
      root: A string, training/validating dataset path, e.g. "./CelebAMask"
      dataset: A string, one of `"train"`, `"test"`, `"valid"`.
      crop_size: A list of two intergers.
      scale_factor: A float number.
      rotation_factor: An integer number.
      ignore_label: An integer number, default is 255.
      transformer: A function of torchvision.transforms.Compose([])
    """
    def __init__(self, 
                 root: str, 
                 dataset: str, 
                 crop_size: list=[473, 473], 
                 scale_factor: float=0.25,
                 rotation_factor: int=30, 
                 ignore_label: int =255, 
                 transform=None) -> None:

        self.root = root
        self.dataset = dataset
        self.crop_size = np.asarray(crop_size)
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.ignore_label = ignore_label
        self.transform = transform

        self.flip_prob = 0.5
        self.flip_pairs = [[4, 5], [6, 7]]
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]

        self.file_list_name = osp.join(root, dataset + '_list.txt')
        self.im_list = [line.split()[0][7:-4] for line in open(self.file_list_name).readlines()]
        self.number_samples = len(self.im_list)


    def __len__(self) -> int:
        return self.number_samples 

    def _box2cs(self, box: list) -> tuple:
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x: float, y: float, w: float, h: float) -> tuple:
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale

    def __getitem__(self, index: int) -> tuple:
        # Load training image
        im_name = self.im_list[index]
        im_path = osp.join(self.root, self.dataset, 'images', im_name + '.jpg')
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)
        # Get center and scale
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset not in ['test', 'valid']:
            edge_path = osp.join(self.root, self.dataset, 'edges', im_name + '.png')
            edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            parsing_anno_path = osp.join(self.root, self.dataset, 'labels', im_name + '.png')
            parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)

            if self.dataset in 'train':

                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                    if random.random() <= 0.6 else 0

        trans = transforms.get_affine_transform(center, s, r, self.crop_size)
        image = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        
        if self.dataset not in ['test', 'valid']:
            edge = cv2.warpAffine(
                edge,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0))

        if self.transform:
            image = self.transform(image)

        meta = {
            'name': im_name,
            'center': center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r,
            'origin': image
        }

        if self.dataset not in 'train':
            return image, meta
        else:

            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))

            label_parsing = torch.from_numpy(label_parsing)

            return image, label_parsing, edge, meta

