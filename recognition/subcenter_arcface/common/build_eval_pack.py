from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#import mxnet as mx
#from mxnet import ndarray as nd
import argparse
import cv2
import pickle
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', '..', 'RetinaFace'))
import face_align
from retinaface import RetinaFace


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def IOU(Reframe, GTframe):
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2] - Reframe[0]
    height1 = Reframe[3] - Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2] - GTframe[0]
    height2 = GTframe[3] - GTframe[1]

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)
    return ratio


parser = argparse.ArgumentParser(description='Package eval images')
# general
parser.add_argument('--data-dir', default='', help='')
parser.add_argument('--image-size', type=int, default=112, help='')
parser.add_argument('--gpu', type=int, default=0, help='')
parser.add_argument('--det-prefix', type=str, default='./model/R50', help='')
parser.add_argument('--output', default='./', help='path to save.')
parser.add_argument('--align-mode', default='arcface', help='align mode.')
args = parser.parse_args()

gpu_id = args.gpu

detector = RetinaFace(args.det_prefix, 0, gpu_id, network='net3')
target_size = 400
max_size = 800


def get_norm_crop(image_path):
    im = cv2.imread(image_path)
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    bbox, landmark = detector.detect(im, threshold=0.5, scales=[im_scale])
    #print(im.shape, bbox.shape, landmark.shape)
    if bbox.shape[0] == 0:
        bbox, landmark = detector.detect(
            im,
            threshold=0.05,
            scales=[im_scale * 0.75, im_scale, im_scale * 2.0])
        print('refine', im.shape, bbox.shape, landmark.shape)
    nrof_faces = bbox.shape[0]
    if nrof_faces > 0:
        det = bbox[:, 0:4]
        img_size = np.asarray(im.shape)[0:2]
        bindex = 0
        if nrof_faces > 1:
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                           det[:, 1])
            img_center = img_size / 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            bindex = np.argmax(bounding_box_size - offset_dist_squared *
                               2.0)  # some extra weight on the centering
        #_bbox = bounding_boxes[bindex, 0:4]
        _landmark = landmark[bindex]
        warped = face_align.norm_crop(im,
                                      landmark=_landmark,
                                      image_size=args.image_size,
                                      mode=args.align_mode)
        return warped
    else:
        return None


bins = []
issame_list = []
pp = 0
for line in open(os.path.join(args.data_dir, 'pairs_label.txt'), 'r'):
    pp += 1
    if pp % 100 == 0:
        print('processing', pp)
    line = line.strip().split()
    assert len(line) == 3
    path1 = os.path.join(args.data_dir, line[0])
    path2 = os.path.join(args.data_dir, line[1])
    im1 = get_norm_crop(path1)
    im2 = get_norm_crop(path2)
    issame = True
    if line[2] == '0':
        issame = False
    issame_list.append(issame)
    for im in [im1, im2]:
        _, s = cv2.imencode('.jpg', im)
        bins.append(s)

with open(args.output, 'wb') as f:
    pickle.dump((bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
