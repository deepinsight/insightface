import os
import os.path as osp
import queue as Queue
import mxnet as mx
import pickle
import threading
import logging
import numpy as np
import insightface
from insightface.utils import face_align
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



class GazeDataset(Dataset):
    def __init__(self, root_dir, is_train):
        super(GazeDataset, self).__init__()

        #self.local_rank = local_rank
        self.is_train = is_train
        self.input_size = 160
        #self.num_kps = 68
        transform_list = []
        if is_train:
            transform_list += \
                [
                    A.ColorJitter(brightness=0.3, contrast=0.3, p=0.5),
                    A.ToGray(p=0.1),
                    A.ISONoise(p=0.1),
                    A.MedianBlur(blur_limit=(1,7), p=0.1),
                    A.GaussianBlur(blur_limit=(1,7), p=0.1),
                    A.MotionBlur(blur_limit=(5,13), p=0.1),
                    A.ImageCompression(quality_lower=10, quality_upper=90, p=0.05),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, interpolation=cv2.INTER_LINEAR, 
                        border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.6),
                    #A.HorizontalFlip(p=0.5),
                    RectangleBorderAugmentation(limit=0.2, fill_value=0, p=0.1),
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
        if is_train:
            path_imgrec = os.path.join(root_dir, 'train.rec')
            path_imgidx = os.path.join(root_dir, 'train.idx')
        else:
            path_imgrec = os.path.join(root_dir, 'val.rec')
            path_imgidx = os.path.join(root_dir, 'val.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        self.imgidx = np.array(list(self.imgrec.keys))
        logging.info('len:%d'%len(self.imgidx))
        print('!!!len:%d'%len(self.imgidx))
        #self.num_face = 1103
        self.num_eye = 481

    def __len__(self):
        return len(self.imgidx)

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        img = mx.image.imdecode(img).asnumpy() #rgb numpy
        y = np.array(header.label, dtype=np.float32).reshape( (-1, 3) )
        #print('!!!', y.shape)
        assert y.shape[0]==self.num_eye*2
        eye_l = y[:self.num_eye,:]
        eye_r = y[self.num_eye:,:]
        mean_z_l = np.mean(eye_l[:32,2])
        mean_z_r = np.mean(eye_r[:32,2])
        std_z_l = np.max(np.abs(eye_l[:32,2]))
        std_z_r = np.max(np.abs(eye_r[:32,2]))
        eye_l[:,2] -= mean_z_l
        eye_r[:,2] -= mean_z_r
        eye_l[:,2] /= std_z_l
        eye_r[:,2] /= std_z_r
        #print('!!!', np.max(eye_l[:,2]), np.min(eye_l[:,2]))
        y = np.concatenate( (eye_l, eye_r), axis=0)
        #y[:,2] /= 100.0
        
        #if self.is_train:
        #    black_edge = np.random.randint(img.shape[1]//3, size=4)
        #    if np.random.random()<0.5:
        #        img[:black_edge[0],:,:] = 0
        #        img[black_edge[1]*-1:,:,:] = 0
        #        img[:,:black_edge[2],:] = 0
        #        img[:,black_edge[3]*-1:,:] = 0
        #label = torch.tensor(y, dtype=torch.float32)
        #label = y
        kps_xy = []
        kps_z = []
        for i in range(y.shape[0]):
            kps_xy.append( (y[i][0], y[i][1]) )
            kps_z.append(y[i][2])
        if self.transform is not None:
            #sample = self.transform(image=sample)['image']
            #t = self.transform(image=img, keypoints=label, class_labels=self.class_labels, class_sides=self.class_sides)
            t = self.transform(image=img, keypoints=kps_xy)
            flipped = False
            #print(t.keys())
            #print('!!!flipped:', flipped)
            img = t['image']
            label_xy = t['keypoints']
            label_xy = np.array(label_xy, dtype=np.float32)
            label_xy /= (self.input_size/2)
            label_xy -= 1.0
            label_z = np.array(kps_z, dtype=np.float32).reshape((-1,1))
            #label_z /= (self.input_size/2)
            label = np.concatenate( (label_xy, label_z), axis=1)
            #label = label.flatten()
            label = torch.tensor(label, dtype=torch.float32)
        #print('label:', label.shape)
        return img, label

