import argparse
import cv2
import sys
import numpy as np
import pickle
import os
import mxnet as mx
import datetime
from skimage import transform as trans
import insightface


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation)*np.pi/180.0
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0]*scale_ratio
    cy = center[1]*scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1*cx, -1*cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size/2, output_size/2))
    t = t1+t2+t3+t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(
        data, M, (output_size, output_size), borderValue=0.0)
    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0]*M[0][0] + M[0][1]*M[0][1])
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2]*scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)


class Handler:
    def __init__(self, prefix, epoch, im_size=128, ctx_id=0):
        print('loading', prefix, epoch)
        if ctx_id >= 0:
            ctx = mx.gpu(ctx_id)
        else:
            ctx = mx.cpu()
        image_size = (im_size, im_size)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        self.image_size = image_size
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(for_training=False, data_shapes=[
                   ('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model
        self.detector = insightface.model_zoo.get_model('retinaface_r50_v1')
        self.detector.prepare(ctx_id=ctx_id)

    def detect(self, img, S=128):
        if img.shape[1] > img.shape[0]:
            det_scale = float(S) / img.shape[1]
            width = S
            height = float(img.shape[0]) / img.shape[1] * S
            height = int(height)
        else:
            det_scale = float(S) / img.shape[0]
            height = S
            width = float(img.shape[1]) / img.shape[0] * S
            width = int(width)
        img_resize = cv2.resize(img, (width, height))
        img_det = np.zeros((S, S, 3), dtype=np.uint8)
        img_det[:height, :width, :] = img_resize
        bboxes, _ = self.detector.detect(img_det, threshold=0.5)
        bboxes /= det_scale
        return bboxes

    def get(self, img):
        bboxes = self.detect(img)
        if bboxes.shape[0] == 0:
            return None
        det = bboxes
        area = (det[:, 2]-det[:, 0])*(det[:, 3]-det[:, 1])
        bindex = np.argmax(area)
        bbox = bboxes[bindex]
        w, h = (bbox[2]-bbox[0]), (bbox[3]-bbox[1])
        center = (bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2
        rotate = 0
        _scale = self.image_size[0]/(1.5*max(w, h))
        input_blob = np.zeros( (1, 3)+self.image_size,dtype=np.uint8 )

        rimg, M = transform(img, center, self.image_size[0], _scale, rotate)
        rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
        rimg = np.transpose(rimg, (2, 0, 1))
        input_blob[0] = rimg
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        x = self.model.get_outputs()[-1].asnumpy()[0]
        x = x.reshape((-1, 3))
        x = x[-68:, :]
        x[:, 0:2] += 1
        x[:, 0:2] *= (self.image_size[0]//2)
        x[:, 2] *= (self.image_size[0]//2)
        IM = cv2.invertAffineTransform(M)
        x = trans_points(x, IM)
        return x

