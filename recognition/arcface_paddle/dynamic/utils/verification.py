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

import time
import os
import numpy as np
import sklearn
import paddle
import logging
from typing import List

from utils.verification import evaluate
from datasets import load_bin


@paddle.no_grad()
def test(data_set, backbone, batch_size, fp16=False, nfolds=10):
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size:bb]
            # 将numpy转Tensor
            img = paddle.to_tensor(
                _data, dtype='float16' if fp16 else 'float32')
            net_out: paddle.Tensor = backbone(img)
            _embeddings = net_out.detach().cpu().numpy()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0].copy()
    try:
        embeddings = sklearn.preprocessing.normalize(embeddings)
    except:
        print(embeddings)
    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    _, _, accuracy, val, val_std, far = evaluate(
        embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list


class CallBackVerification(object):
    def __init__(self,
                 frequent,
                 rank,
                 batch_size,
                 val_targets,
                 rec_prefix,
                 fp16=False,
                 image_size=(112, 112)):
        self.frequent: int = frequent
        self.rank: int = rank
        self.batch_size: int = batch_size
        self.fp16 = fp16
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        if self.rank == 0:
            self.init_dataset(
                val_targets=val_targets,
                data_dir=rec_prefix,
                image_size=image_size)

    def ver_test(self, backbone: paddle.nn.Layer, global_step: int):
        for i in range(len(self.ver_list)):
            test_start = time.time()
            acc1, std1, acc2, std2, xnorm, embeddings_list = test(
                self.ver_list[i],
                backbone,
                self.batch_size,
                fp16=self.fp16,
                nfolds=10)
            logging.info('[%s][%d]XNorm: %f' %
                         (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' %
                         (self.ver_name_list[i], global_step, acc2, std2))
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info('[%s][%d]Accuracy-Highest: %1.5f' % (
                self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            test_end = time.time()
            logging.info("test time: {:.4f}".format(test_end - test_start))

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: paddle.nn.Layer):
        if self.rank == 0 and num_update > 0 and num_update % self.frequent == 0:
            backbone.eval()
            with paddle.no_grad():
                self.ver_test(backbone, num_update)
            backbone.train()
