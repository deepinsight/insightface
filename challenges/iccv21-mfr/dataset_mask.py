import numbers
import os
import queue as Queue
import threading

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from insightface.app import MaskAugmentation


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


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank, aug_modes="brightness=0.1+mask=0.1"):
        super(MXFaceDataset, self).__init__()
        default_aug_probs = {
                'brightness' : 0.2,
                'blur': 0.1,
                'mask': 0.1,
                }

        aug_mode_list = aug_modes.lower().split('+')
        aug_mode_map = {}
        for aug_mode_str in aug_mode_list:
            _aug = aug_mode_str.split('=')
            aug_key = _aug[0]
            if len(_aug)>1:
                aug_prob = float(_aug[1])
            else:
                aug_prob = default_aug_probs[aug_key]
            aug_mode_map[aug_key] = aug_prob

        transform_list = []
        self.mask_aug = False
        self.mask_prob = 0.0
        key = 'mask'
        if key in aug_mode_map:
            self.mask_aug = True
            self.mask_prob = aug_mode_map[key]
            transform_list.append(
                MaskAugmentation(mask_names=['mask_white', 'mask_blue', 'mask_black', 'mask_green'], mask_probs=[0.4, 0.4, 0.1, 0.1], h_low=0.33, h_high=0.4, p=self.mask_prob)
                )
        if local_rank==0:
            print('data_transform_list:', transform_list)
            print('mask:', self.mask_aug, self.mask_prob)
        key = 'brightness'
        if key in aug_mode_map:
            prob = aug_mode_map[key]
            transform_list.append(
                A.RandomBrightnessContrast(brightness_limit=0.125, contrast_limit=0.05, p=prob)
                )
        key = 'blur'
        if key in aug_mode_map:
            prob = aug_mode_map[key]
            transform_list.append(
                A.ImageCompression(quality_lower=30, quality_upper=80, p=prob)
                )
            transform_list.append(
                A.MedianBlur(blur_limit=(1,7), p=prob)
                )
            transform_list.append(
                A.MotionBlur(blur_limit=(5,12), p=prob)
                )
        transform_list += \
            [
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ]
        #here, the input for A transform is rgb cv2 img
        self.transform = A.Compose(
            transform_list 
        )
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        #print(header)
        #print(len(self.imgrec.keys))
        if header.flag > 0:
            if len(header.label)==2:
                self.imgidx = np.array(range(1, int(header.label[0])))
            else:
                self.imgidx = np.array(list(self.imgrec.keys))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
        #print('imgidx len:', len(self.imgidx))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        hlabel = header.label
        #print('hlabel:', hlabel.__class__)
        sample = mx.image.imdecode(img).asnumpy()
        if not isinstance(hlabel, numbers.Number):
            idlabel = hlabel[0]
        else:
            idlabel = hlabel
        label = torch.tensor(idlabel, dtype=torch.long)
        if self.transform is not None:
            sample = self.transform(image=sample, hlabel=hlabel)['image']
        return sample, label

    def __len__(self):
        return len(self.imgidx)

if __name__ == "__main__":
    import argparse, cv2, copy
    parser = argparse.ArgumentParser(description='dataset test')
    parser.add_argument('--dataset', type=str,  help='dataset path')
    parser.add_argument('--samples', type=int, default=256, help='')
    parser.add_argument('--cols', type=int, default=16, help='')
    args = parser.parse_args()
    assert args.samples%args.cols==0
    assert args.cols%2==0
    samples = args.samples
    cols = args.cols
    rows = args.samples // args.cols
    dataset = MXFaceDataset(root_dir=args.dataset, local_rank=0, aug_modes='mask=1.0')
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    dataset_0 = copy.deepcopy(dataset)
    #dataset_0.transform = None
    dataset_1 = copy.deepcopy(dataset)
    #dataset_1.transform = A.Compose(
    #    [
    #        A.RandomBrightnessContrast(brightness_limit=0.125, contrast_limit=0.05, p=1.0),
    #        A.ImageCompression(quality_lower=30, quality_upper=80, p=1.0),
    #        A.MedianBlur(blur_limit=(1,7), p=1.0),
    #        A.MotionBlur(blur_limit=(5,12), p=1.0),
    #        A.Affine(scale=(0.92, 1.08),  translate_percent=(-0.06, 0.06), rotate=(-6, 6), shear=None, interpolation=cv2.INTER_LINEAR, p=1.0),
    #    ]
    #)
    fig = np.zeros( (112*rows, 112*cols, 3), dtype=np.uint8 )
    for idx in range(samples):
        if idx%2==0:
            image, _ = dataset_0[idx//2]
        else:
            image, _ = dataset_1[idx//2]
        row_idx = idx // cols
        col_idx = idx % cols
        fig[row_idx*112:(row_idx+1)*112, col_idx*112:(col_idx+1)*112,:] = image[:,:,::-1] # to bgr
    cv2.imwrite("./datasets.png", fig)
