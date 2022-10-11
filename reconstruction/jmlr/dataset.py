import numbers
import os
import os.path as osp
import pickle
import queue as Queue
import threading
import logging
import numbers
import math
import pandas as pd
from scipy.spatial.transform import Rotation

import mxnet as mx
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from skimage import transform as sktrans
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from augs import *


def Rt26dof(R_t, degrees=False):
    yaw_gt, pitch_gt, roll_gt = Rotation.from_matrix(R_t[:3, :3].T).as_euler('yxz', degrees=degrees)
    label_euler = np.array([pitch_gt, yaw_gt, roll_gt])
    label_translation = R_t[3, :3]
    label_6dof = np.concatenate([label_euler, label_translation])
    return label_6dof


def gen_target_pip(target, target_map, target_local_x, target_local_y):
    map_channel, map_height, map_width = target_map.shape
    target = target.reshape(-1, 2)
    assert map_channel == target.shape[0]

    for i in range(map_channel):
        mu_x = int(math.floor(target[i][0] * map_width))
        mu_y = int(math.floor(target[i][1] * map_height))
        mu_x = max(0, mu_x)
        mu_y = max(0, mu_y)
        mu_x = min(mu_x, map_width-1)
        mu_y = min(mu_y, map_height-1)
        target_map[i, mu_y, mu_x] = 1
        shift_x = target[i][0] * map_width - mu_x
        shift_y = target[i][1] * map_height - mu_y
        target_local_x[i, mu_y, mu_x] = shift_x
        target_local_y[i, mu_y, mu_x] = shift_y


    return target_map, target_local_x, target_local_y

def get_tris(cfg):
    import trimesh
    data_root = Path(cfg.root_dir)
    obj_path = data_root / 'resources/example.obj'
    mesh = trimesh.load(obj_path, process=False)
    verts_template = np.array(mesh.vertices, dtype=np.float32)
    tris = np.array(mesh.faces, dtype=np.int32)
    #print(verts_template.shape, tris.shape)
    return tris


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
    def __init__(self, cfg, is_train=True, is_test=False, local_rank=0):
        super(FaceDataset, self).__init__()

        
        self.data_root = Path(cfg.root_dir)
        self.input_size = cfg.input_size
        self.transform = get_aug_transform(cfg)
        self.local_rank = local_rank
        self.is_test = is_test
        txt_path = self.data_root / 'resources/projection_matrix.txt'
        self.M_proj = np.loadtxt(txt_path, dtype=np.float32)
        if is_test:
            data_root = Path(cfg.root_dir)
            csv_path = data_root / 'list/WCPA_track2_test.csv'
            self.df = pd.read_csv(csv_path, dtype={'subject_id': str, 'facial_action': str, 'img_id': str})
        else:
            if is_train:
                self.df = pd.read_csv(osp.join(cfg.cache_dir, 'train_list.csv'), dtype={'subject_id': str, 'facial_action': str, 'img_id': str})
            else:
                self.df = pd.read_csv(osp.join(cfg.cache_dir, 'val_list.csv'), dtype={'subject_id': str, 'facial_action': str, 'img_id': str})
        self.label_6dof_mean = [-0.018197, -0.017891, 0.025348, -0.005368, 0.001176, -0.532206]   # mean of pitch, yaw, roll, tx, ty, tz
        self.label_6dof_std = [0.314015, 0.271809, 0.081881, 0.022173, 0.048839, 0.065444]        # std of pitch, yaw, roll, tx, ty, tz
        self.align_face = cfg.align_face
        if not self.align_face:
            self.dst_pts = np.float32([
                [0, 0],
                [0, cfg.input_size- 1],
                [cfg.input_size- 1, 0]
            ])
        else:
            dst_pts = np.array([
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041] ], dtype=np.float32 )

            new_size = 144
            dst_pts[:,0] += ((new_size-112)//2)
            dst_pts[:,1] += 8
            dst_pts[:,:] *= (self.input_size/float(new_size))
            self.dst_pts = dst_pts

        if local_rank==0:
            logging.info('data_transform_list:%s'%self.transform)
            logging.info('len:%d'%len(self.df))
        self.is_test_aug = False
        self.eye_dataset = None
        if cfg.eyes is not None:
            from eye_dataset import EyeDataset
            self.eye_dataset = EyeDataset(cfg.eyes['root'])

    def set_test_aug(self):
        if not self.is_test_aug:
            from easydict import EasyDict as edict
            cfg = edict()
            cfg.aug_modes = ['test-aug']
            cfg.input_size = self.input_size
            cfg.task = 0
            self.transform = get_aug_transform(cfg)
            self.is_test_aug = True

    def get_names(self, index):
        subject_id = self.df['subject_id'][index]
        facial_action = self.df['facial_action'][index]
        img_id = self.df['img_id'][index]
        return subject_id, facial_action, img_id

    def __getitem__(self, index):
        subject_id = self.df['subject_id'][index]
        facial_action = self.df['facial_action'][index]
        img_id = self.df['img_id'][index]

        img_path = self.data_root / 'image' / subject_id / facial_action / f'{img_id}_ar.jpg'
        npz_path = self.data_root / 'info' / subject_id / facial_action / f'{img_id}_info.npz'
        txt_path = self.data_root / '68landmarks' / subject_id / facial_action / f'{img_id}_68landmarks.txt'
        #if not osp.exists(img_path):
        #    continue

        #print(img_path)
        img_raw = cv2.imread(str(img_path))
        #if img_raw is None:
        #    print('XXX ERR:', img_path)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        #print(img_raw.shape)
        img_h, img_w, _ = img_raw.shape
        pts68 = np.loadtxt(txt_path, dtype=np.int32)

        x_min, y_min = pts68.min(axis=0)
        x_max, y_max = pts68.max(axis=0)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min


        if not self.align_face:
            size = max(w, h)
            ss = np.array([0.75, 0.75, 0.85, 0.65])     # predefined expand size

            left = x_center - ss[0] * size
            right = x_center + ss[1] * size
            top = y_center - ss[2] * size
            bottom = y_center + ss[3] * size

            src_pts = np.float32([
                [left, top],
                [left, bottom],
                [right, top]
            ])
            tform = cv2.getAffineTransform(src_pts, self.dst_pts)
        else:
            src_pts = np.float32([
                (pts68[36] + pts68[39])/2,
                (pts68[42] + pts68[45])/2,
                pts68[30],
                pts68[48],
                pts68[54]
            ])
            tf = sktrans.SimilarityTransform()
            tf.estimate(src_pts, self.dst_pts)
            tform = tf.params[0:2,:]

        img_local = cv2.warpAffine(img_raw, tform, (self.input_size,)*2, flags=cv2.INTER_CUBIC)
        fake_points2d = np.ones( (1,2), dtype=np.float32) * (self.input_size//2)

        #tform_inv = cv2.invertAffineTransform(tform)
        #img_global = cv2.warpAffine(img_local, tform_inv, (img_w, img_h), borderValue=0.0)
        #img_global = cv2.resize(img_global, (self.input_size, self.input_size))
        if self.transform is not None:
            t = self.transform(image=img_local, keypoints=fake_points2d)
            img_local = t['image']
            if self.is_test_aug:
                height, width = img_local.shape[:2]
                for trans in t["replay"]["transforms"]:
                    if trans['__class_fullname__']=='ShiftScaleRotate' and trans['applied']:
                        param = trans['params']
                        dx, dy, angle, scale = param['dx'], param['dy'], param['angle'], param['scale']
                        center = (width / 2, height / 2)
                        matrix = cv2.getRotationMatrix2D(center, angle, scale)
                        matrix[0, 2] += dx * width
                        matrix[1, 2] += dy * height
                        new_matrix = np.identity(3)
                        new_matrix[:2,:3] = matrix
                        old_tform = np.identity(3)
                        old_tform[:2,:3] = tform
                        #new_tform = np.dot(old_tform, new_matrix)
                        new_tform = np.dot(new_matrix, old_tform)
                        #print('label_tform:')
                        #print(label_tform.flatten())
                        #print(new_matrix.flatten())
                        #print(new_tform.flatten())
                        tform = new_tform[:2,:3]
                        break
                            #print('trans param:', param)
            #img_global = self.transform(image=img_global)['image']

        tform_tensor = torch.tensor(tform, dtype=torch.float32)
        d = {'img_local': img_local, 'tform': tform_tensor}
        if self.eye_dataset is not None:
            eye_key = str(Path('image') / subject_id / facial_action / f'{img_id}_ar.jpg')
            #print(eye_key)
            eyel, eyer = self.eye_dataset.get(eye_key, to_homo=True)
            if eyel is not None:
                #print(eye_key, el_inv.shape, er_inv.shape)
                d['eye_world_left'] = torch.tensor(eyel, dtype=torch.float32)
                d['eye_world_right'] = torch.tensor(eyer, dtype=torch.float32)
        if not self.is_test:
            M = np.load(npz_path)
            #yaw_gt, pitch_gt, roll_gt = Rotation.from_matrix(M['R_t'][:3, :3].T).as_euler('yxz', degrees=False)
            #label_euler = np.array([pitch_gt, yaw_gt, roll_gt])
            #label_translation = M['R_t'][3, :3]
            #label_6dof = np.concatenate([label_euler, label_translation])
            #label_6dof = (label_6dof - self.label_6dof_mean) / self.label_6dof_std
            #label_6dof_tensor = torch.tensor(label_6dof, dtype=torch.float32)
            #label_verts = M['verts'] * 10.0     # roughly [-1, 1]
            #label_verts_tensor = torch.tensor(label_verts, dtype=torch.float32)
            #return img_local, label_verts_tensor, label_6dof_tensor
            label_verts_tensor = torch.tensor(M['verts'], dtype=torch.float32)
            label_Rt_tensor = torch.tensor(M['R_t'], dtype=torch.float32)
            d['verts'] = label_verts_tensor
            d['rt'] = label_Rt_tensor
            #return img_local, img_global, label_verts_tensor, label_Rt_tensor, tform_tensor
            #return img_local, label_verts_tensor, label_Rt_tensor, tform_tensor
        else:
            #return img_local, img_global, tform_tensor
            index_tensor = torch.tensor(index, dtype=torch.long)
            d['index'] = index_tensor
            #return img_local, tform_tensor, index_tensor
        return d


    def __len__(self):
        return len(self.df)

class MXFaceDataset(Dataset):
    def __init__(self, cfg, is_train=True, norm_6dof=True, degrees_6dof=False, local_rank=0):
        super(MXFaceDataset, self).__init__()

        
        self.is_train = is_train
        self.data_root = Path(cfg.root_dir)
        self.input_size = cfg.input_size
        self.transform = get_aug_transform(cfg)
        self.local_rank = local_rank
        self.use_trainval = cfg.use_trainval
        self.use_eye = cfg.eyes is not None
        if is_train:
            #self.df = pd.read_csv(osp.join(cfg.cache_dir, 'train_list.csv'), dtype={'subject_id': str, 'facial_action': str, 'img_id': str})
            path_imgrec = os.path.join(cfg.cache_dir, 'train.rec')
            path_imgidx = os.path.join(cfg.cache_dir, 'train.idx')
            self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
            self.imgidx = list(self.imgrec.keys)
            self.imggroup = [0] * len(self.imgidx)
            self.size_train = len(self.imgidx)
            if self.use_trainval:
                assert not cfg.sampling_hard
                path_imgrec = os.path.join(cfg.cache_dir, 'val.rec')
                path_imgidx = os.path.join(cfg.cache_dir, 'val.idx')
                self.imgrec2 = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
                imgidx2 = list(self.imgrec2.keys)
                self.imggroup += [1] * len(imgidx2)
                self.imgidx += imgidx2
        else:
            #self.df = pd.read_csv(osp.join(cfg.cache_dir, 'val_list.csv'), dtype={'subject_id': str, 'facial_action': str, 'img_id': str})
            path_imgrec = os.path.join(cfg.cache_dir, 'val.rec')
            path_imgidx = os.path.join(cfg.cache_dir, 'val.idx')
            self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
            self.imgidx = list(self.imgrec.keys)
            self.imggroup = [0] * len(self.imgidx)
        self.imgidx = np.array(self.imgidx)
        self.imggroup = np.array(self.imggroup)
        if cfg.sampling_hard and is_train:
            meta = np.load(os.path.join(cfg.cache_dir, 'train.meta.npy'))
            assert meta.shape[0]==len(self.imgidx)
            new_imgidx = []
            for i in range(len(self.imgidx)):
                idx = self.imgidx[i]
                assert i==idx
                pose = np.abs(meta[i,:2])
                #repeat = np.sum(pose>=35)*3+1
                if np.max(pose)<15:
                    repeat = 2
                else:
                    repeat = 1
                new_imgidx += [idx]*repeat
            if local_rank==0:
                print('new-imgidx:', len(self.imgidx), len(new_imgidx))
            self.imgidx = np.array(new_imgidx)
        self.label_6dof_mean = [-0.018197, -0.017891, 0.025348, -0.005368, 0.001176, -0.532206]   # mean of pitch, yaw, roll, tx, ty, tz
        self.label_6dof_std = [0.314015, 0.271809, 0.081881, 0.022173, 0.048839, 0.065444]        # std of pitch, yaw, roll, tx, ty, tz
        txt_path = self.data_root / 'resources/projection_matrix.txt'
        self.M_proj = np.loadtxt(txt_path, dtype=np.float32)
        self.M1 = np.array([
            [400.0,       0, 0, 0],
            [      0, 400.0, 0, 0],
            [      0,       0, 1, 0],
            [400.0, 400.0, 0, 1]
        ])
        self.dst_pts = np.float32([
            [0, 0],
            [0, cfg.input_size- 1],
            [cfg.input_size- 1, 0]
        ])
        self.norm_6dof = norm_6dof
        self.degrees_6dof = degrees_6dof
        self.task = cfg.task
        self.num_verts = cfg.num_verts
        self.loss_pip = cfg.loss_pip
        self.net_stride = 32
        if local_rank==0:
            logging.info('data_transform_list:%s'%self.transform)
            logging.info('len:%d'%len(self.imgidx))
            logging.info('glen:%d'%len(self.imggroup))
        self.is_test_aug = False

        self.enable_flip = cfg.enable_flip
        self.flipindex = cfg.flipindex.copy()
        self.verts3d_central_index = cfg.verts3d_central_index
        self.eye_dataset = None
        self.use_eye = False
        if cfg.eyes is not None:
            #from eye_dataset import EyeDataset
            #self.eye_dataset = EyeDataset(cfg.eyes['root'], load_data=False)
            self.use_eye = True

    def set_test_aug(self):
        if not self.is_test_aug:
            from easydict import EasyDict as edict
            cfg = edict()
            cfg.aug_modes = ['test-aug']
            cfg.input_size = self.input_size
            cfg.task = 0
            self.transform = get_aug_transform(cfg)
            self.is_test_aug = True

    def __getitem__(self, index):
        idx = self.imgidx[index]
        group = self.imggroup[index]
        if group==0:
            imgrec = self.imgrec
        elif group==1:
            imgrec = self.imgrec2
        elif group==2:
            imgrec = self.imgrec3

        s = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        hlabel = header.label
        img = mx.image.imdecode(img).asnumpy() #rgb numpy

        label_verts = np.array(hlabel[:1220*3], dtype=np.float32).reshape(-1,3)
        label_Rt = np.array(hlabel[1220*3:1220*3+16], dtype=np.float32).reshape(4,4)
        label_tform = np.array(hlabel[1220*3+16:1220*3+16+6], dtype=np.float32).reshape(2,3)
        label_6dof = Rt26dof(label_Rt, self.degrees_6dof)
        if self.norm_6dof:
            label_6dof = (label_6dof - self.label_6dof_mean) / self.label_6dof_std
        label_6dof_tensor = torch.tensor(label_6dof, dtype=torch.float32)
        el_inv = None
        er_inv = None
        if self.use_eye:
            a = 1220*3+16+6
            el_inv = np.array(hlabel[a:a+481*3], dtype=np.float32).reshape(-1,3)
            a+=481*3
            er_inv = np.array(hlabel[a:a+481*3], dtype=np.float32).reshape(-1,3)
            #el_inv = torch.tensor(el_inv, dtype=torch.float32)
            #er_inv = torch.tensor(er_inv, dtype=torch.float32)
            #eye_verts = [el_inv, er_inv]
            eye_verts = np.concatenate( (el_inv, er_inv), axis=0 )

        #img_local = None
        img_raw = None
        #if self.task==0 or self.task==2:
        #    img_raw = img[:,self.input_size:,:]
        #if self.task==0 or self.task==1 or self.task==3:
        #    img_local = img[:,:self.input_size,:]
        assert img.shape[0]==img.shape[1] and img.shape[0]>=self.input_size
        if img.shape[0]>self.input_size:
            scale = float(self.input_size) / img.shape[0]
            #print('scale:', scale)
            #src_pts = np.float32([
            #    [0, 0],
            #    [0, 799],
            #    [799, 0]
            #])
            #tform = cv2.getAffineTransform(src_pts, self.dst_pts)
            #new_tform = np.identity(3)
            #new_tform[:2,:3] = tform
            #label_tform = np.dot(new_tform, label_tform.T).T

            src_pts = np.float32([
                [0, 0, 1],
                [0, 799, 1],
                [799, 0, 1]
            ])
            dst_pts = np.dot(label_tform, src_pts.T).T
            dst_pts *= scale
            dst_pts = dst_pts.copy()
            src_pts = src_pts[:,:2].copy()
            #print('index:', index)
            #print(src_pts.shape, dst_pts.shape)
            #print(label_tform.shape)
            #print(src_pts.dtype)
            #print(dst_pts.dtype)
            tform = cv2.getAffineTransform(src_pts, dst_pts)
            label_tform = tform

            img = cv2.resize(img, (self.input_size, self.input_size))

        img_local = img
        need_points2d = (self.task==0 or self.task==3)

        if need_points2d:
            ones = np.ones([label_verts.shape[0], 1])
            verts_homo = np.concatenate([label_verts, ones], axis=1)
            verts = verts_homo @ label_Rt @ self.M_proj @ self.M1
            w_ = verts[:, [3]]
            verts = verts / w_
            points2d = verts[:, :3]
            points2d[:, 1] = 800.0 - points2d[:, 1]
            verts2d = points2d[:,:2].copy()
            points2d[:,2] = 1.0
            points2d = np.dot(label_tform, points2d.T).T
        else:
            points2d = np.ones( (1,2), dtype=np.float32) * (self.input_size//2)
        if self.use_eye:
            verts_homo = eye_verts
            if verts_homo.shape[1] == 3:
                ones = np.ones([verts_homo.shape[0], 1])
                verts_homo = np.concatenate([verts_homo, ones], axis=1)
            verts_out = verts_homo @ label_Rt @ self.M_proj @ self.M1
            w_ = verts_out[:, [3]]
            verts_out = verts_out / w_
            _points2d = verts_out[:, :3]
            _points2d[:, 1] = 800.0 - _points2d[:, 1]
            _points2d[:,2] = 1.0
            _points2d = np.dot(label_tform, _points2d.T).T
            eye_points = _points2d
        #if img.shape[0]!=self.input_size:
        #    assert img.shape[0]>self.input_size
            #img = cv2.resize(img, (self.input_size, self.input_size))
            #scale = float(self.input_size) / img.shape[0]
            #points2d *= scale

        if self.transform is not None:
            if img_raw is not None:
                img_raw = self.transform(image=img_raw, keypoints=points2d)['image']
            if img_local is not None:
                height, width = img_local.shape[:2]
                x = self.transform(image=img_local, keypoints=points2d)
                img_local = x['image']
                points2d = x['keypoints']
                points2d = np.array(points2d, dtype=np.float32)
                if self.is_test_aug:
                    for trans in x["replay"]["transforms"]:
                        if trans['__class_fullname__']=='ShiftScaleRotate' and trans['applied']:
                            param = trans['params']
                            dx, dy, angle, scale = param['dx'], param['dy'], param['angle'], param['scale']
                            center = (width / 2, height / 2)
                            matrix = cv2.getRotationMatrix2D(center, angle, scale)
                            matrix[0, 2] += dx * width
                            matrix[1, 2] += dy * height
                            new_matrix = np.identity(3)
                            new_matrix[:2,:3] = matrix
                            old_tform = np.identity(3)
                            old_tform[:2,:3] = label_tform
                            #new_tform = np.dot(old_tform, new_matrix)
                            new_tform = np.dot(new_matrix, old_tform)
                            #print('label_tform:')
                            #print(label_tform.flatten())
                            #print(new_matrix.flatten())
                            #print(new_tform.flatten())
                            label_tform = new_tform[:2,:3]
                            break
                            #print('trans param:', param)


        if self.loss_pip:
            target_map = np.zeros((self.num_verts, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
            target_local_x = np.zeros((self.num_verts, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
            target_local_y = np.zeros((self.num_verts, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
            target = points2d / self.input_size
            target_map, target_local_x, target_local_y = gen_target_pip(target, target_map, target_local_x, target_local_y)
            target_map_tensor = torch.tensor(target_map, dtype=torch.float32)
            target_x_tensor = torch.tensor(target_local_x, dtype=torch.float32)
            target_y_tensor = torch.tensor(target_local_y, dtype=torch.float32)
            d['pip_map'] = target_map_tensor
            d['pip_x'] = target_x_tensor
            d['pip_y'] = target_y_tensor

        if self.is_train and self.enable_flip and np.random.random()<0.5:
            #if self.local_rank==0:
            #    print('XXX:', label_verts[:5,:2])
            img_local = img_local.flip([2])
            x_of_central = 0.0
            #x_of_central = label_verts[self.verts3d_central_index,0]
            #x_of_central = np.mean(x_of_central)
            label_verts = label_verts[self.flipindex,:]
            label_verts[:,0] -= x_of_central
            label_verts[:,0] *= -1.0
            label_verts[:,0] += x_of_central

            if need_points2d:
                flipped_p2d = points2d[self.flipindex,:].copy()
                flipped_p2d[:,0] = self.input_size - 1 - flipped_p2d[:,0]
                points2d = flipped_p2d
            if self.use_eye:
                flipped_p2d = eye_points[self.flipindex,:].copy()
                flipped_p2d[:,0] = self.input_size - 1 - flipped_p2d[:,0]
                eye_points = flipped_p2d
        label_verts_tensor = torch.tensor(label_verts*10.0, dtype=torch.float32)
        d = {}
        d['img_local'] = img_local
        d['verts'] = label_verts_tensor
        d['6dof'] = label_6dof_tensor
        d['rt'] = torch.tensor(label_Rt, dtype=torch.float32)
        if need_points2d:
            points2d = points2d / (self.input_size//2) - 1.0
            points2d_tensor = torch.tensor(points2d, dtype=torch.float32)
            d['points2d'] = points2d_tensor
        if self.use_eye:
            d['eye_verts'] = torch.tensor(eye_verts, dtype=torch.float32)
            eye_points = eye_points / (self.input_size//2) - 1.0
            eye_points_tensor = torch.tensor(eye_points, dtype=torch.float32)
            d['eye_points'] = eye_points_tensor

        loss_weight = 1.0
        if group!=0:
            loss_weight = 0.0
        loss_weight_tensor = torch.tensor(loss_weight, dtype=torch.float32)
        d['loss_weight'] = loss_weight_tensor
        label_tform_tensor = torch.tensor(label_tform, dtype=torch.float32)
        d['tform'] = label_tform_tensor

        #if img_local is None:
        #    image = (img_raw,)
        #elif img_raw is None:
        #    image = (img_local,)
        #else:
        #    image = (img_local,img_raw)
        #ret = image + (label_verts_tensor, label_6dof_tensor, points2d_tensor)
        if not self.is_train:
            idx_tensor = torch.tensor([idx], dtype=torch.long)
            d['idx'] = idx_tensor
            d['verts2d'] = torch.tensor(verts2d, dtype=torch.float32)
        return d


    def __len__(self):
        return len(self.imgidx)

def test_dataset1(cfg):
    cfg.task = 0
    is_train = False
    center_axis = []
    dataset = MXFaceDataset(cfg, is_train=is_train, norm_6dof=False, local_rank=0)
    for i in range(len(dataset.flipindex)):
        if i==dataset.flipindex[i]:
            center_axis.append(i)
    print(center_axis)
    #dataset.transform = None
    print('total:', len(dataset))
    total = len(dataset)
    #total = 50
    list_6dof = []
    all_mean_xs = []
    for idx in range(total):
        #img_local, img_raw, label_verts, label_6dof, = dataset[idx]
        #img_local, img_raw, label_verts, label_6dof, points2d, tform, data_idx = dataset[idx]
        #img_local, label_verts, label_6dof, points2d, tform, data_idx = dataset[idx]
        d = dataset[idx]
        img_local = d['img_local']
        label_verts = d['verts']
        label_6dof = d['6dof']
        points2d = d['points2d']
        label_verts = label_verts.numpy()
        label_6dof = label_6dof.numpy()
        points2d = points2d.numpy()
        #print(img_local.shape, label_verts.shape, label_6dof.shape, points2d.shape)
        verts3d = label_verts / 10.0
        xs = []
        for c in center_axis:
            _x = verts3d[c,0]
            xs.append(_x)
        _std = np.std(xs)
        print(xs)
        print(_std)
        #print(np.mean(xs))
        all_mean_xs.append(np.mean(xs))
        if idx%100==0:
            print('processing:', idx, np.mean(all_mean_xs))
        #print(label_verts[:3,:], label_6dof)
        #list_6dof.append(label_6dof)
        #print(image.__class__, label_verts.__class__)
        #label = list(label_verts.numpy().flatten()) + list(label_6dof.numpy().flatten())
        #points2d = label_verts2[:,:2]
        #points2d = (points2d+1) * 128.0
        #img_local = img_local.numpy()
        #img_local = (img_local+1.0) * 128.0
        #draw = img_local.astype(np.uint8).transpose( (1,2,0) )[:,:,::-1].copy()
        #for i in range(points2d.shape[0]):
        #    pt = points2d[i].astype(np.int)
        #    cv2.circle(draw, pt, 2, (255,0,0), 2)
        ##output_path = "outputs/%d_%.3f_%.3f_%.3f.jpg"%(idx, label_6dof[0], label_6dof[1], label_6dof[2])
        #output_path = "outputs/%06d.jpg"%(idx)
        #cv2.imwrite(output_path, draw)
    #list_6dof = np.array(list_6dof)
    #print('MEAN:')
    #print(np.mean(list_6dof, axis=0))

def test_loader1(cfg):
    cfg.task = 0
    is_train = True
    dataset = MXFaceDataset(cfg, is_train=is_train, norm_6dof=False, local_rank=0)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for index, d in enumerate(loader):
        #img_local = d['img_local']
        label_verts = d['verts']
        points2d = d['points2d']
        tform = d['tform']
        label_verts /= 10.0
        points2d = (points2d + 1.0) * (cfg.input_size//2)
        tform = tform.numpy()
        verts = label_verts.numpy()
        points2d = points2d.numpy()
        print(verts.shape, points2d.shape, tform.shape)
        np.save("temp/verts3d.npy", verts)
        np.save("temp/points2d.npy", points2d)
        np.save("temp/tform.npy", tform)
        break

def test_facedataset1(cfg):
    cfg.task = 0
    cfg.input_size = 512
    dataset = FaceDataset(cfg, is_train=True, local_rank=0)
    for idx in range(100000):
        img_local, label_verts, label_Rt, tform = dataset[idx]
        label_Rt = label_Rt.numpy()
        if label_Rt[0,0]>1.0:
            print(idx, label_Rt.shape)
            print(label_Rt)
            break

def test_arcface(cfg):
    cfg.task = 0
    is_train = True
    dataset = MXFaceDataset(cfg, is_train=is_train, norm_6dof=False, local_rank=0)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for index, d in enumerate(loader):
        img = d['img_local'].numpy()
        img /= 2.0
        img += 0.5
        img *= 255.0
        img = img[0]
        img = img.transpose( (1,2,0) )
        img = img.astype(np.uint8)
        img = cv2.resize(img, (144,144))
        img = img[:,:,::-1]
        img = img[8:120,16:128,:]
        print(img.shape)
        cv2.imwrite("temp/arc_%d.jpg"%index, img)
        #np.save("temp/verts3d.npy", verts)
        #np.save("temp/points2d.npy", points2d)
        #np.save("temp/tform.npy", tform)
        if index>100:
            break

def test_dataset2(cfg):
    cfg.task = 0
    is_train = False
    center_axis = []
    dataset = MXFaceDataset(cfg, is_train=is_train, norm_6dof=False, local_rank=0)
    for i in range(len(dataset.flipindex)):
        if i==dataset.flipindex[i]:
            center_axis.append(i)
    print(center_axis)
    #dataset.transform = None
    print('total:', len(dataset))
    total = len(dataset)
    total = 50
    list_6dof = []
    all_mean_xs = []
    for idx in range(total):
        d = dataset[idx]
        img_local = d['img_local']
        label_verts = d['verts']
        label_6dof = d['6dof']
        points2d = d['points2d']
        label_verts = label_verts.numpy()
        label_6dof = label_6dof.numpy()
        points2d = points2d.numpy()
        eye_points = d['eye_points'].numpy()
        eye_verts = d['eye_verts'].numpy()
        print(eye_verts[:5,:])
        #print(img_local.shape, label_verts.shape, label_6dof.shape, points2d.shape)
        verts3d = label_verts / 10.0
        #print(label_verts[:3,:], label_6dof)
        #list_6dof.append(label_6dof)
        #print(image.__class__, label_verts.__class__)
        #label = list(label_verts.numpy().flatten()) + list(label_6dof.numpy().flatten())
        #points2d = label_verts2[:,:2]
        points2d = (points2d+1) * 128.0
        eye_points = (eye_points+1) * 128.0
        img_local = img_local.numpy()
        img_local = (img_local+1.0) * 128.0
        draw = img_local.astype(np.uint8).transpose( (1,2,0) )[:,:,::-1].copy()
        for i in range(points2d.shape[0]):
            pt = points2d[i].astype(np.int)
            cv2.circle(draw, pt, 2, (255,0,0), 2)
        for i in range(eye_points.shape[0]):
            pt = eye_points[i].astype(np.int)
            cv2.circle(draw, pt, 2, (0,255,0), 2)
        ##output_path = "outputs/%d_%.3f_%.3f_%.3f.jpg"%(idx, label_6dof[0], label_6dof[1], label_6dof[2])
        output_path = "outputs/%06d.jpg"%(idx)
        cv2.imwrite(output_path, draw)
    #list_6dof = np.array(list_6dof)
    #print('MEAN:')
    #print(np.mean(list_6dof, axis=0))

if __name__ == "__main__":
    from utils.utils_config import get_config
    #cfg = get_config('configs/r0_a1.py')
    cfg = get_config('configs/s2')
    #test_loader1(cfg)
    #test_facedataset1(cfg)
    #test_arcface(cfg)
    test_dataset2(cfg)


