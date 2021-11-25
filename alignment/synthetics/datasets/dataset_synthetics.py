import os
import os.path as osp
import queue as Queue
import pickle
import threading
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .augs import RectangleBorderAugmentation

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch



class FaceDataset(Dataset):
    def __init__(self, root_dir, is_train):
        super(FaceDataset, self).__init__()

        #self.local_rank = local_rank
        self.is_train = is_train
        self.input_size = 256
        self.num_kps = 68
        transform_list = []
        if is_train:
            transform_list += \
                [
                    A.ColorJitter(brightness=0.8, contrast=0.5, p=0.5),
                    A.ToGray(p=0.1),
                    A.ISONoise(p=0.1),
                    A.MedianBlur(blur_limit=(1,7), p=0.1),
                    A.GaussianBlur(blur_limit=(1,7), p=0.1),
                    A.MotionBlur(blur_limit=(5,12), p=0.1),
                    A.ImageCompression(quality_lower=50, quality_upper=90, p=0.05),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=40, interpolation=cv2.INTER_LINEAR, 
                        border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.8),
                    A.HorizontalFlip(p=0.5),
                    RectangleBorderAugmentation(limit=0.33, fill_value=0, p=0.2),
                ]
        transform_list += \
            [
                A.geometric.resize.Resize(self.input_size, self.input_size, interpolation=cv2.INTER_LINEAR, always_apply=True),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ]
        self.transform = A.ReplayCompose(
            transform_list,
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
        )
        self.root_dir = root_dir
        with open(osp.join(root_dir, 'annot.pkl'), 'rb') as f:
            annot = pickle.load(f)
            self.X, self.Y = annot
        train_size = int(len(self.X)*0.99)
        
        if is_train:
            self.X = self.X[:train_size]
            self.Y = self.Y[:train_size]
        else:
            self.X = self.X[train_size:]
            self.Y = self.Y[train_size:]
        #if local_rank==0:
        #    logging.info('data_transform_list:%s'%transform_list)
        flip_parts = ([1, 17], [2, 16], [3, 15], [4, 14], [5, 13], [6, 12], [7, 11], [8, 10],
            [18, 27], [19, 26], [20, 25], [21, 24], [22, 23],
            [32, 36], [33, 35],
            [37, 46], [38, 45], [39, 44], [40, 43], [41, 48], [42, 47],
            [49, 55], [50, 54], [51, 53], [62, 64], [61, 65], [68, 66], [59, 57], [60, 56])
        self.flip_order = np.arange(self.num_kps)
        for pair in flip_parts:
            self.flip_order[pair[1]-1] = pair[0]-1
            self.flip_order[pair[0]-1] = pair[1]-1
        logging.info('len:%d'%len(self.X))
        print('!!!len:%d'%len(self.X))

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        image_path = os.path.join(self.root_dir, x)
        img = cv2.imread(image_path)[:,:,::-1]
        label = y
        if self.transform is not None:
            t = self.transform(image=img, keypoints=label)
            flipped = False
            for trans in t["replay"]["transforms"]:
                if trans["__class_fullname__"].endswith('HorizontalFlip'):
                    if trans["applied"]:
                        flipped = True
            img = t['image']
            label = t['keypoints']
            label = np.array(label, dtype=np.float32)
            #print(img.shape)
            if flipped:
                #label[:, 0] = self.input_size - 1 - label[:, 0]  #already applied in horizantal flip aug
                label = label[self.flip_order,:]
            label /= (self.input_size/2)
            label -= 1.0
            label = label.flatten()
            label = torch.tensor(label, dtype=torch.float32)
        return img, label

    def __len__(self):
        return len(self.X)

