# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np


image_root = '/data/victor/cvte_baby/CVTE_Baby_En/'
info_root = '/data/victor/cvte_baby/Label_of_Baby/'
save_root = '/data/victor/cvte_baby/landmark_test/'

with open(os.path.join(info_root, 'baby_label.txt'), 'r') as f:
    lines_label = map(lambda s: s.strip().split(), f.readlines())

with open(os.path.join(info_root, 'baby_landmark.txt'), 'r') as f:
    lines_landmark = map(lambda s: s.strip().split(), f.readlines())


def drawPoints(img_path, landmark):
    img = cv2.imread(img_path)
    for x, y in landmark:
        cv2.circle(img, (x,y), 5, (0, 255, 0), 2)
    return img


lines = []
i = 0
for line_label, line_landmark in zip(lines_label, lines_landmark):
    image_name1, label = line_label
    image_name = line_landmark[0]
    landmark = np.array(line_landmark[1:]).astype(np.float32).reshape((5,2))
    assert image_name1 == image_name
    # print image_name, landmark
    if i % 100 == 0:
        cv2.imwrite(os.path.join(save_root, '{}.jpg'.format(i)), drawPoints(
                    os.path.join(image_root,image_name), landmark))
    lines.append('\t'.join(['0', os.path.realpath(os.path.join(image_root,image_name)), label,
                            '\t'.join(['-1']*4),'\t'.join(landmark.T.flatten().astype(np.str))]))
    i += 1

with open('/data/victor/insightface/datasets/faces_cvtebaby_112x112/train.lst', 'w') as f:
    f.write('\n'.join(lines))


