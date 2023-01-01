#!/usr/bin/env python

import os
import os.path as osp
import argparse
import cv2
import numpy as np
import onnxruntime
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX

onnxruntime.set_default_logger_severity(3)

assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')

detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
detector.prepare(0)
model_path = os.path.join(assets_dir, 'w600k_r50.onnx')
rec = ArcFaceONNX(model_path)
rec.prepare(0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('img1', type=str)
    parser.add_argument('img2', type=str)
    return parser.parse_args()


def func(args):
    image1 = cv2.imread(args.img1)
    image2 = cv2.imread(args.img2)
    bboxes1, kpss1 = detector.autodetect(image1, max_num=1)
    if bboxes1.shape[0]==0:
        return -1.0, "Face not found in Image-1"
    bboxes2, kpss2 = detector.autodetect(image2, max_num=1)
    if bboxes2.shape[0]==0:
        return -1.0, "Face not found in Image-2"
    kps1 = kpss1[0]
    kps2 = kpss2[0]
    feat1 = rec.get(image1, kps1)
    feat2 = rec.get(image2, kps2)
    sim = rec.compute_sim(feat1, feat2)
    if sim<0.2:
        conclu = 'They are NOT the same person'
    elif sim>=0.2 and sim<0.28:
        conclu = 'They are LIKELY TO be the same person'
    else:
        conclu = 'They ARE the same person'
    return sim, conclu



if __name__ == '__main__':
    args = parse_args()
    output = func(args)
    print('sim: %.4f, message: %s'%(output[0], output[1]))

