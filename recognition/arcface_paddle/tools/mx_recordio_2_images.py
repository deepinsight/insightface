# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import numpy as np
import numbers
import mxnet as mx
import cv2
import tqdm
import shutil


def main(args):
    path_imgrec = os.path.join(args.root_dir, 'train.rec')
    path_imgidx = os.path.join(args.root_dir, 'train.idx')
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    if header.flag > 0:
        header0 = (int(header.label[0]), int(header.label[1]))
        imgidx = np.array(range(1, int(header.label[0])))
    else:
        imgidx = np.array(list(imgrec.keys))

    classes = set()
    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)
    fp = open(os.path.join(args.output_dir, 'label.txt'), 'w')
    for idx in tqdm.tqdm(imgidx):
        s = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        img = mx.image.imdecode(img).asnumpy()[..., ::-1]
        label = int(label)
        classes.add(label)

        filename = 'images/%08d.jpg' % idx
        fp.write('%s\t%d\n' % (filename, label))
        cv2.imwrite(
            os.path.join(args.output_dir, filename), img,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    fp.close()
    shutil.copy(
        os.path.join(args.root_dir, 'agedb_30.bin'),
        os.path.join(args.output_dir, 'agedb_30.bin'))
    shutil.copy(
        os.path.join(args.root_dir, 'cfp_fp.bin'),
        os.path.join(args.output_dir, 'cfp_fp.bin'))
    shutil.copy(
        os.path.join(args.root_dir, 'lfw.bin'),
        os.path.join(args.output_dir, 'lfw.bin'))
    print('num_image: ', len(imgidx), 'num_classes: ', len(classes))
    with open(os.path.join(args.output_dir, 'README.md'), 'w') as f:
        f.write('num_image: {}\n'.format(len(imgidx)))
        f.write('num_classes: {}\n'.format(len(classes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        help="Root directory to mxnet dataset.", )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to output.", )
    args = parser.parse_args()
    main(args)
