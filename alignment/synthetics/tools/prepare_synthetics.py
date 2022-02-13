
import sys
import glob
import torch
import pickle
import os
import numpy as np
import cv2
import os.path as osp
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(224, 224))
output_size = 384

input_dir = '/root/codebase/FaceSynthetics'
output_dir = 'data/synthetics'

if not osp.exists(output_dir):
    os.makedirs(output_dir)

X = []
Y = []

for i in range(0, 100000):
    if i%1000==0:
        print('loading', i)
    x = "%06d.png"%i
    img_path = osp.join(input_dir, x)
    img = cv2.imread(img_path)
    dimg = img.copy()
    ylines = open(osp.join(input_dir, "%06d_ldmks.txt"%i)).readlines()
    ylines = ylines[:68]
    y = []
    for yline in ylines:
        lmk = [float(x) for x in yline.strip().split()]
        y.append( tuple(lmk) )
    pred = np.array(y)
    faces = app.get(img, max_num=1)
    if len(faces)!=1:
        continue
    bbox = faces[0].bbox
    w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
    center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
    rotate = 0
    _scale = output_size  / (max(w, h)*1.5)
    aimg, M = face_align.transform(dimg, center, output_size, _scale, rotate)
    pred = face_align.trans_points(pred, M)
    #box = bbox.astype(np.int)
    #color = (0, 0, 255)
    #cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)

    #kps = pred.astype(np.int)
    #for l in range(kps.shape[0]):
    #    color = (0, 0, 255)
    #    cv2.circle(aimg, (kps[l][0], kps[l][1]), 1, color, 2)
    x = x.replace('png', 'jpg')
    X.append(x)
    y = []
    for k in range(pred.shape[0]):
        y.append( (pred[k][0], pred[k][1]) )
    Y.append(y)
    cv2.imwrite("%s/%s"%(output_dir, x), aimg)


with open(osp.join(output_dir, 'annot.pkl'), 'wb') as pfile:
    pickle.dump((X, Y), pfile, protocol=pickle.HIGHEST_PROTOCOL)

