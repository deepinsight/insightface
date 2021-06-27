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
import cv2
import paddle
import backbones


def read_img(img_path=None):
    if img_path is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scale = 1. / 255.
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    mean = np.array(mean).reshape((1, 1, 3)).astype('float32')
    std = np.array(std).reshape((1, 1, 3)).astype('float32')
    img = (img.astype('float32') * scale - mean) / std
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    return img


def main(args):
    backbone = eval("backbones.{}".format(args.network))()
    model_params = args.network + '.pdparams'
    print('INFO:' + args.network + ' chose! ' + model_params + ' loaded!')
    state_dict = paddle.load(os.path.join(args.checkpoint, model_params))
    backbone.set_state_dict(state_dict)
    backbone.eval()
    img = read_img(args.img)
    input_tensor = paddle.to_tensor(img)
    feat = backbone(input_tensor).numpy()
    return feat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paddle ArcFace Testing')
    parser.add_argument(
        '--network',
        type=str,
        default='MobileFaceNet_128',
        help='backbone network')
    parser.add_argument(
        '--img', type=str, default='None', help='backbone network')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='emore_arcface',
        help='checkpoint dir')
    args = parser.parse_args()
    main(args)
