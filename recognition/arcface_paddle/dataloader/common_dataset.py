# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle.io import Dataset
from paddle.vision import transforms
import os
import cv2
from PIL import Image
import random
import paddle
import numpy as np

from dataloader.kv_helper import read_img_from_bin


class CommonDataset(Dataset):
    def __init__(self, root_dir, label_file, is_bin=True):
        super(CommonDataset, self).__init__()
        self.root_dir = root_dir
        self.label_file = label_file
        self.full_lines = self.get_file_list(label_file)
        self.delimiter = "\t"
        self.is_bin = is_bin
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.num_samples = len(self.full_lines)

    def get_file_list(self, label_file):
        with open(label_file, "r") as fin:
            full_lines = fin.readlines()

        print("finish reading file, image num: {}".format(len(full_lines)))
        return full_lines

    def __getitem__(self, idx):
        try:
            line = self.full_lines[idx]

            img_path, label = line.split(self.delimiter)
            label = int(label)
            label = paddle.to_tensor(label, dtype='int64')
            img_path = os.path.join(self.root_dir, img_path)
            if self.is_bin:
                img = read_img_from_bin(img_path)
            else:
                img = cv2.imread(img_path)
            img = img[:, :, ::-1]
            img = self.transform(img)
            return img, label

        except Exception as e:
            print("data read faild: {}, exception info: {}".format(line, e))
            return self.__getitem__(random.randint(0, len(self)))

    def __len__(self):
        return self.num_samples
