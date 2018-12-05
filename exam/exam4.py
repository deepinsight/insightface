#

from __future__ import print_function

import os
import pickle

import mxnet as mx
from PIL import Image

target = "cfp_fp"

bins, issame_list = pickle.load(open("/home/lijc08/datasets/glintasia/faces_glintasia/%s.bin" % target, 'rb'))
if os.path.exists(target):
    os.rmdir(target)
os.mkdir(target)
for i in range(0, len(issame_list), 100):
    issame = issame_list[i]
    dirPath = os.path.join(target, str(i) + "_" + str(issame))
    os.mkdir(dirPath)

    for index in [i * 2, i * 2 + 1]:
        img = mx.image.imdecode(bins[index])
        imgData = Image.fromarray(img.asnumpy(), 'RGB')
        imgData.save(dirPath + "/" + str(index) + ".jpeg", format='JPEG')
