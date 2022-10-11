
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
        self.M1 = M1
        camera_matrix = self.M_proj @ M1
        camera_matrix =  camera_matrix[:3,:3].T
        camera_matrix[0,2] = 400
        camera_matrix[1,2] = 400
        self.camera_matrix = camera_matrix.copy()
        self.use_eyes = False
        if cfg.eyes is not None:
            self.use_eyes = True
            from eye_dataset import EyeDataset
            eye_dataset = EyeDataset(cfg.eyes['root'], load_data=False)
            self.iris_idx_481 = eye_dataset.iris_idx_481

    
    def project_shape_in_image(self, verts, R_t):
        verts_homo = verts
        if verts_homo.shape[1] == 3:
            ones = np.ones([verts_homo.shape[0], 1])
            verts_homo = np.concatenate([verts_homo, ones], axis=1)
        verts_out = verts_homo @ R_t @ self.M_proj @ self.M1
        w_ = verts_out[:, [3]]
        verts_out = verts_out / w_
        return verts_out

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
        self.raw_width = width
        self.raw_height = height


    def forward(self, img_local, is_flip=False):
        if is_flip:
            img_local = img_local.flip([3])
        pred = self.backbone(img_local)
        pred1 = pred[:,:1220*3]
        pred2 = pred[:,1220*3:1220*5]
        meta = {'flip': is_flip}
        if not self.use_eyes:
            return pred1, pred2, meta
        else:
            eye_verts = pred[:,1220*5:1220*5+481*2*3]
            eye_points = pred[:,1220*5+481*2*3:]
            return pred1, pred2, meta, eye_verts, eye_points


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
        B = pred2.shape[0]
        points2d = (pred2.reshape(B,-1,2)+1.0) * self.input_size//2
        if is_flip:
            points2d = points2d[:,self.flipindex,:]
            points2d[:,:,0] = self.input_size - 1 - points2d[:,:,0]
        #B = points2d.shape[0]
        points2de = np.ones( (points2d.shape[0], points2d.shape[1], 3), dtype=points2d.dtype)
        points2de[:,:,:2] = points2d
        verts2d = np.zeros((B,points2d.shape[1],2), dtype=np.float32)
        for n in range(B):
            tform = tforms[n]
            tform_inv = cv2.invertAffineTransform(tform)
            _points2d = np.dot(tform_inv, points2de[n].T).T
            verts2d[n] = _points2d
        #return verts2d, points2d
        return verts2d


    def convert_eyes(self, eye_verts3d, eye_verts2d, R_t, tforms):
        meta = {'flip': False}
        eye_verts3d = eye_verts3d.cpu().numpy().reshape(-1, 481*2, 3)[0]
        eye_verts2d = self.convert_2d(eye_verts2d, tforms, meta)[0]
        el_inv = eye_verts3d[:481,:]
        er_inv = eye_verts3d[481:,:]
        v_el = eye_verts2d[:481,:]
        v_er = eye_verts2d[481:,:]
        vector_norm = 0.035
        # gaze vector of left eye in world space
        gl_vector = el_inv[self.iris_idx_481].mean(axis=0) - el_inv[-1]
        gl_vector = (gl_vector / np.linalg.norm(gl_vector)) * vector_norm
        gl_point = el_inv[self.iris_idx_481].mean(axis=0) + gl_vector
        # gaze vector of right eye in world space
        gr_vector = er_inv[self.iris_idx_481].mean(axis=0) - er_inv[-1]
        gr_vector = (gr_vector / np.linalg.norm(gr_vector)) * vector_norm
        gr_point = er_inv[self.iris_idx_481].mean(axis=0) + gr_vector
        g_el = self.project_shape_in_image(gl_point[None, :], R_t)
        g_er = self.project_shape_in_image(gr_point[None, :], R_t)
        g_el = g_el[:, :3].copy()
        g_el[:, 1] = self.raw_height - g_el[:, 1]
        g_er = g_er[:, :3].copy()
        g_er[:, 1] = self.raw_height - g_er[:, 1]
        pt1_l = v_el[self.iris_idx_481][:, [0, 1]].mean(axis=0).astype(np.int32)
        pt2_l = g_el[0, [0, 1]].astype(np.int32)
        pt1_r = v_er[self.iris_idx_481][:, [0, 1]].mean(axis=0).astype(np.int32)
        pt2_r = g_er[0, [0, 1]].astype(np.int32)
        return eye_verts3d, eye_verts2d, (pt1_l, pt2_l), (pt1_r, pt2_r)

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
        if not net.use_eyes:
            pred1, pred2, meta = net(img_local, is_flip=False)
        else:
            pred1, pred2, meta, eye_verts, eye_points = net(img_local, is_flip=False)
    pred_verts = net.convert_verts(pred1, meta)
    tform = torch.from_numpy(tform.reshape(1,2,3))
    pred_verts2d = net.convert_2d(pred2, tform, meta)
    verts = pred_verts[0]
    verts2d = pred_verts2d[0]
    R, t = net.solve_one(verts, verts2d)
    if not net.use_eyes:
        return verts, verts2d
    else:
        R_t = np.zeros( (4,4), dtype=np.float32)
        R_t[:3,:3] = R
        R_t[3,:3] = t
        R_t[3,3] = 1.0
        eye_verts, eye_verts2d, gaze_l, gaze_r = net.convert_eyes(eye_verts, eye_points, R_t, tform)
        return verts, verts2d, eye_verts, eye_verts2d, gaze_l, gaze_r


if __name__ == "__main__":
    import argparse
    from utils.utils_config import get_config
    from insightface.app import FaceAnalysis
    parser = argparse.ArgumentParser(description='JMLR inference')
    #parser.add_argument('config', type=str, help='config file')
    config_file = 'configs/s1.py'
    #config_file = 'configs/s2.py'
    args = parser.parse_args()
    cfg = get_config(config_file)
    cfg2 = None
    local_rank = 0
    #img = cv2.imread('sample.jpg')
    net = JMLRInference(cfg, local_rank)
    net = net.to(local_rank)
    net.eval()
    app = FaceAnalysis(allowed_modules='detection')
    app.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.5)
    index = -1
    for img_path in glob.glob('/data/insightface/wcpa/image/222714/01_LeftToRight_Neutral/*.jpg'):
        index+=1
        img = cv2.imread(img_path)
        if index==0:
            net.set_raw_image_size(img.shape[1], img.shape[0])
        draw = img.copy()
        faces = app.get(img)
        for face in faces:
            if not net.use_eyes:
                verts3d, verts2d = get(net, img, face.kps)
            else:
                verts3d, verts2d, eye_verts3d, eye_verts2d, gaze_l, gaze_r = get(net, img, face.kps)
            #print(verts3d.shape, verts2d.shape, R.shape, t.shape, eye_verts3d.shape, eye_verts2d.shape)
            for i in range(verts2d.shape[0]):
                pt = verts2d[i].astype(np.int32)
                cv2.circle(draw, pt, 2, (255,0,0), 2)
            #eye_verts2d = eye_verts2d[:481,:]
            if net.use_eyes:
                for i in range(eye_verts2d.shape[0]):
                    pt = eye_verts2d[i].astype(np.int32)
                    cv2.circle(draw, pt, 2, (0,255,0), 2)
                for gaze in [gaze_l, gaze_r]:
                    pt1, pt2 = gaze
                    cv2.arrowedLine(draw, pt1, pt2, [0, 0, 255], 10)
        cv2.imwrite('./outputs/%04d.jpg'%index, draw)

