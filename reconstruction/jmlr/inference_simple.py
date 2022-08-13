
import os
import time
import timm
import glob
import numpy as np
import os.path as osp
import cv2

import torch
import torch.distributed as dist
from torch import nn
from pathlib import Path
from backbones import get_network
from skimage import transform as sktrans
from scipy.spatial.transform import Rotation

def batch_euler2matrix(batch_euler):
    n = batch_euler.shape[0]
    assert batch_euler.shape[1] == 3
    batch_matrix = np.zeros([n, 3, 3], dtype=np.float32)

    for i in range(n):
        pitch, yaw, roll = batch_euler[i]
        R = Rotation.from_euler('yxz', [yaw, pitch, roll], degrees=False).as_matrix().T
        batch_matrix[i] = R

    return batch_matrix

def euler2matrix(euler):
    assert len(euler)==3
    matrix = np.zeros([3, 3], dtype=np.float32)

    pitch, yaw, roll = euler
    R = Rotation.from_euler('yxz', [yaw, pitch, roll], degrees=False).as_matrix().T
    matrix = R
    return matrix

def Rt_from_6dof(pred_6dof):
    assert pred_6dof.ndim==1 or pred_6dof.ndim==2
    if pred_6dof.ndim==1:
        R_pred = euler2matrix(pred_6dof[:3])
        t_pred = pred_6dof[-3:]
        return R_pred, t_pred
    else:
        R_pred = batch_euler2matrix(pred_6dof[:,:3])
        t_pred = pred_6dof[:,-3:].reshape(-1,1,3)
        return R_pred, t_pred

def solver_rigid(pts_3d , pts_2d , camera_matrix):
    # pts_3d  Nx3
    # pts_2d  Nx2
    # camera_matrix 4x4
    dist_coeffs = np.zeros((4,1))
    pts_3d = pts_3d.copy()
    pts_2d = pts_2d.copy()
    #print(pts_3d.shape, pts_3d.dtype, pts_2d.shape, pts_2d.dtype)
    success, rotation_vector, translation_vector = cv2.solvePnP(pts_3d, pts_2d, camera_matrix, dist_coeffs, flags=0)
    assert success
    R, _ = cv2.Rodrigues(rotation_vector)
    R = R.T
    R[:,1:3] *= -1
    T = translation_vector.flatten()
    T[1:] *= -1

    return R,T


class JMLRInference(nn.Module):
    def __init__(self, cfg, local_rank=0):
        super(JMLRInference, self).__init__()
        backbone = get_network(cfg)
        if cfg.ckpt is None:
            ckpts = list(glob.glob(osp.join(cfg.output, "backbone*.pth")))
            backbone_pth = sorted(ckpts)[-1]
        else:
            backbone_pth = cfg.ckpt
        if local_rank==0:
            print(backbone_pth)
        backbone_ckpt = torch.load(backbone_pth, map_location=torch.device(local_rank))
        if 'model' in backbone_ckpt:
            backbone_ckpt = backbone_ckpt['model']
        backbone.load_state_dict(backbone_ckpt)
        backbone.eval()
        backbone.requires_grad_(False)
        self.backbone = backbone
        self.num_verts = cfg.num_verts
        self.input_size = cfg.input_size
        self.flipindex = cfg.flipindex.copy()
        self.data_root = Path(cfg.root_dir)
        txt_path = self.data_root / 'resources/projection_matrix.txt'
        self.M_proj = np.loadtxt(txt_path, dtype=np.float32)
        M1 = np.array([
            [400.0,       0, 0, 0],
            [      0, 400.0, 0, 0],
            [      0,       0, 1, 0],
            [400.0, 400.0, 0, 1]
        ])
        camera_matrix = self.M_proj @ M1
        camera_matrix =  camera_matrix[:3,:3].T
        camera_matrix[0,2] = 400
        camera_matrix[1,2] = 400
        self.camera_matrix = camera_matrix.copy()

    def set_raw_image_size(self, width, height):
        w = width / 2.0
        h = height / 2.0
        M1 = np.array([
            [w,       0, 0, 0],
            [      0, h, 0, 0],
            [      0,       0, 1, 0],
            [w, h, 0, 1]
        ])
        camera_matrix = self.M_proj @ M1
        camera_matrix =  camera_matrix[:3,:3].T
        camera_matrix[0,2] = w
        camera_matrix[1,2] = h
        self.camera_matrix = camera_matrix


    def forward(self, img_local, is_flip=False):
        if is_flip:
            img_local = img_local.flip([3])
        pred = self.backbone(img_local)
        pred1 = pred[:,:1220*3]
        pred2 = pred[:,1220*3:]
        meta = {'flip': is_flip}
        return pred1, pred2, meta


    def convert_verts(self, pred1, meta):
        is_flip = meta['flip']
        pred1 = pred1.cpu().numpy()
        pred1 = pred1[:,:1220*3]
        pred_verts = pred1.reshape(-1,1220,3) / 10.0
        if is_flip:
            pred_verts = pred_verts[:,self.flipindex,:]
            pred_verts[:,:,0] *= -1.0
        return pred_verts

    def convert_2d(self, pred2, tforms, meta):
        is_flip = meta['flip']
        tforms = tforms.cpu().numpy()
        pred2 = pred2.cpu().numpy()
        points2d = (pred2.reshape(-1,1220,2)+1.0) * self.input_size//2
        if is_flip:
            points2d = points2d[:,self.flipindex,:]
            points2d[:,:,0] = self.input_size - 1 - points2d[:,:,0]
        B = points2d.shape[0]
        points2de = np.ones( (points2d.shape[0], points2d.shape[1], 3), dtype=points2d.dtype)
        points2de[:,:,:2] = points2d
        verts2d = np.zeros((B,1220,2), dtype=np.float32)
        for n in range(B):
            tform = tforms[n]
            tform_inv = cv2.invertAffineTransform(tform)
            _points2d = np.dot(tform_inv, points2de[n].T).T
            verts2d[n] = _points2d
        return verts2d, points2d

    def solve(self, verts3d, verts2d):
        print(verts3d.shape, verts2d.shape)
        B = verts3d.shape[0]
        R = np.zeros([B, 3, 3], dtype=np.float32)
        t = np.zeros([B, 1, 3], dtype=np.float32)
        for n in range(B):
            _R, _t = solver_rigid(verts3d[n], verts2d[n], self.camera_matrix)
            R[n] = _R
            t[n,0] = _t
        return R, t

    def solve_one(self, verts3d, verts2d):
        R, t = solver_rigid(verts3d, verts2d, self.camera_matrix)
        return R, t


def get(net, img, keypoints):
    dst_pts = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041] ], dtype=np.float32 )
    input_size = 256
    local_rank = 0

    new_size = 144
    dst_pts[:,0] += ((new_size-112)//2)
    dst_pts[:,1] += 8
    dst_pts[:,:] *= (input_size/float(new_size))
    tf = sktrans.SimilarityTransform()
    tf.estimate(keypoints, dst_pts)
    tform = tf.params[0:2,:]
    img = cv2.warpAffine(img, tform, (input_size,)*2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    img_local = img.to(local_rank)
    with torch.no_grad():
        pred1, pred2, meta = net(img_local, is_flip=False)
    pred_verts = net.convert_verts(pred1, meta)
    tform = torch.from_numpy(tform.reshape(1,2,3))
    pred_verts2d, pred_points2d = net.convert_2d(pred2, tform, meta)
    return pred_verts[0], pred_verts2d[0]


if __name__ == "__main__":
    import argparse
    from utils.utils_config import get_config
    from insightface.app import FaceAnalysis
    parser = argparse.ArgumentParser(description='JMLR inference')
    #parser.add_argument('config', type=str, help='config file')
    config_file = 'configs/s1.py'
    args = parser.parse_args()
    cfg = get_config(config_file)
    cfg2 = None
    local_rank = 0
    img = cv2.imread('sample.jpg')
    net = JMLRInference(cfg, local_rank)
    print(img.shape)
    net.set_raw_image_size(img.shape[1], img.shape[0])
    net = net.to(local_rank)
    net.eval()
    app = FaceAnalysis(allowed_modules='detection')
    app.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.5)
    draw = img.copy()
    faces = app.get(img)
    for face in faces:
        verts3d, verts2d = get(net, img, face.kps)
        R, t = net.solve_one(verts3d, verts2d)
        print(verts3d.shape, verts2d.shape, R.shape, t.shape)
        for i in range(verts2d.shape[0]):
            pt = verts2d[i].astype(np.int)
            cv2.circle(draw, pt, 2, (255,0,0), 2)
    cv2.imwrite('./draw.jpg', draw)

