import os

import cv2
import numpy as np
from skimage import transform as trans

src = np.array([[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
                [33.5493, 92.3655], [62.7299, 92.2041]],
               dtype=np.float32)
src[:, 0] += 8.0

img_path = '/data/anxiang/datasets/IJB_release/IJBC/loose_crop'
img_path_align = '/data/anxiang/datasets/IJB_release/IJBC/loose_crop_align'

img_list_path = '/data/anxiang/datasets/IJB_release/IJBC/meta/ijbc_name_5pts_score.txt'
img_list = open(img_list_path)
files = img_list.readlines()

for img_index, each_line in enumerate(files):
    if img_index % 500 == 0:
        print('processing', img_index)
    name_lmk_score = each_line.strip().split(' ')
    img_name = os.path.join(img_path, name_lmk_score[0])
    img = cv2.imread(img_name)
    landmark = np.array([float(x) for x in name_lmk_score[1:-1]],
                        dtype=np.float32)
    landmark = landmark.reshape((5, 2))

    if landmark.shape[0] == 68:
        landmark5 = np.zeros((5, 2), dtype=np.float32)
        landmark5[0] = (landmark[36] + landmark[39]) / 2
        landmark5[1] = (landmark[42] + landmark[45]) / 2
        landmark5[2] = landmark[30]
        landmark5[3] = landmark[48]
        landmark5[4] = landmark[54]
    else:
        landmark5 = landmark
    tform = trans.SimilarityTransform()
    tform.estimate(landmark5, src)
    M = tform.params[0:2, :]
    img = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    cv2.imwrite(os.path.join(img_path_align, name_lmk_score[0]), img)
