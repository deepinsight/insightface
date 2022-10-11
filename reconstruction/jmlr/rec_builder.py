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
from dataset import FaceDataset, Rt26dof

class RecBuilder():
    def __init__(self, path, image_size=(112, 112), is_train=True):
        self.path = path
        self.image_size = image_size
        self.widx = 0
        self.wlabel = 0
        self.max_label = -1
        #assert not osp.exists(path), '%s exists' % path
        if is_train:
            rec_file = osp.join(path, 'train.rec')
            idx_file = osp.join(path, 'train.idx')
        else:
            rec_file = osp.join(path, 'val.rec')
            idx_file = osp.join(path, 'val.idx')
        #assert not osp.exists(rec_file), '%s exists' % rec_file
        if not osp.exists(path):
            os.makedirs(path)
        self.writer = mx.recordio.MXIndexedRecordIO(idx_file,
                                                    rec_file,
                                                    'w')
        self.meta = []

    def add(self, imgs):
        #!!! img should be BGR!!!!
        #assert label >= 0
        #assert label > self.last_label
        assert len(imgs) > 0
        label = self.wlabel
        for img in imgs:
            idx = self.widx
            image_meta = {'image_index': idx, 'image_classes': [label]}
            header = mx.recordio.IRHeader(0, label, idx, 0)
            if isinstance(img, np.ndarray):
                s = mx.recordio.pack_img(header,img,quality=95,img_fmt='.jpg')
            else:
                s = mx.recordio.pack(header, img)
            self.writer.write_idx(idx, s)
            self.meta.append(image_meta)
            self.widx += 1
        self.max_label = label
        self.wlabel += 1
        return label


    def add_image(self, img, label):
        #!!! img should be BGR!!!!
        #assert label >= 0
        #assert label > self.last_label
        idx = self.widx
        header = mx.recordio.IRHeader(0, label, idx, 0)
        if isinstance(img, np.ndarray):
            s = mx.recordio.pack_img(header,img,quality=100,img_fmt='.jpg')
        else:
            s = mx.recordio.pack(header, img)
        self.writer.write_idx(idx, s)
        self.widx += 1

    def close(self):
        print('stat:', self.widx, self.wlabel)

if __name__ == "__main__":
    #cfg = get_config('configs/s1.py')
    cfg = get_config('configs/s2.py')
    cfg.task = 0
    cfg.input_size = 512
    for is_train in [True, False]:
        dataset = FaceDataset(cfg, is_train=is_train, local_rank=0)
        dataset.transform = None
        writer = RecBuilder(cfg.cache_dir, is_train=is_train)
        #writer = RecBuilder("temp", is_train=is_train)
        print('total:', len(dataset))
        #meta = np.zeros( (len(dataset), 3), dtype=np.float32 )
        meta = []
        subset_name = 'train' if is_train else 'val'
        meta_path = osp.join(cfg.cache_dir, '%s.meta'%subset_name)
        eye_missing = 0
        for idx in range(len(dataset)):
            #img_local, img_global, label_verts, label_Rt, tform = dataset[idx]
            #img_local, label_verts, label_Rt, tform = dataset[idx]
            data = dataset[idx]
            img_local = data['img_local']
            label_verts = data['verts']
            label_Rt = data['rt']
            tform = data['tform']
            label_verts = label_verts.numpy()
            label_Rt = label_Rt.numpy()
            tform = tform.numpy()
            label_6dof = Rt26dof(label_Rt, True)
            pose = label_6dof[:3]
            #print(image.shape, label_verts.shape, label_6dof.shape)
            #print(image.__class__, label_verts.__class__)
            img_local = img_local[:,:,::-1]
            #img_global = img_global[:,:,::-1]
            #image = np.concatenate( (img_local, img_global), axis=1 )
            image = img_local
            label = list(label_verts.flatten()) + list(label_Rt.flatten()) + list(tform.flatten())
            expect_len = 1220*3+16+6
            if 'eye_world_left' in data:
                if idx==0:
                    print('find eye')
                eyel = data['eye_world_left'].numpy()
                eyer = data['eye_world_right'].numpy()
                label += list(eyel.flatten())
                label += list(eyer.flatten())
                expect_len += 481*6
            else:
                eye_missing += 1
                continue
            meta.append(pose)
            assert len(label)==expect_len
            writer.add_image(image, label)
            if idx%100==0:
                print('processing:', idx, image.shape, len(label))
            if idx<10:
                cv2.imwrite("temp/%d.jpg"%idx, image)
        writer.close()
        meta = np.array(meta, dtype=np.float32)
        np.save(meta_path, meta)
        print('Eye missing:', eye_missing, is_train)

