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
import pickle
import struct
import random
import multiprocessing
import numpy as np
import cv2
import json


def readkv(f):
    """readkv"""
    keylendata = f.read(4)
    if len(keylendata) != 4:
        return None
    keylen = struct.unpack('I', keylendata)[0]
    if keylen > 5000:
        raise Exception('wrong key len' + str(keylen))
    key = f.read(keylen)
    valuelen = struct.unpack('I', f.read(4))[0]
    value = f.read(valuelen)
    return key, value


def writekv(f, k, v, flush=True):
    """writekv"""
    f.write(struct.pack('I', len(k)))
    f.write(k)
    f.write(struct.pack('I', len(v)))
    f.write(v)
    if flush:
        f.flush()
    return


def trans_img_to_bin(img_name, output_path):
    with open(img_name, "rb") as fin:
        img = fin.read()
    key = os.path.split(img_name)[-1]
    with open(output_path, "wb") as fout:
        writekv(fout, key.encode(), pickle.dumps(img, -1))
    return


def read_img_from_bin(input_path):
    # the file can exist many key-vals, but it just save one in fact.
    with open(input_path, "rb") as fin:
        r = readkv(fin)
        assert r is not None
        _, value = r
        value = pickle.loads(value)
        value = np.frombuffer(value, dtype='uint8')
        img = cv2.imdecode(value, 1)
    return img
