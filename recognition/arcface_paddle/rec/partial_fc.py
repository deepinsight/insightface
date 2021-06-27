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
import paddle
import paddle.nn as nn
from paddle.nn.functional import normalize, linear
import pickle


class PartialFC(nn.Layer):
    """
    Author: {Xiang An, Yang Xiao, XuHan Zhu} in DeepGlint,
    Partial FC: Training 10 Million Identities on a Single Machine
    See the original paper:
    https://arxiv.org/abs/2010.05222
    """

    @paddle.no_grad()
    def __init__(self,
                 rank,
                 world_size,
                 batch_size,
                 resume,
                 margin_softmax,
                 num_classes,
                 sample_rate=1.0,
                 embedding_size=512,
                 prefix="./"):
        super(PartialFC, self).__init__()
        self.num_classes: int = num_classes
        self.rank: int = rank
        self.world_size: int = world_size
        self.batch_size: int = batch_size
        self.margin_softmax: callable = margin_softmax
        self.sample_rate: float = sample_rate
        self.embedding_size: int = embedding_size
        self.prefix: str = prefix
        self.num_local: int = num_classes // world_size + int(
            rank < num_classes % world_size)
        self.class_start: int = num_classes // world_size * rank + min(
            rank, num_classes % world_size)
        self.num_sample: int = int(self.sample_rate * self.num_local)

        self.weight_name = os.path.join(
            self.prefix, "rank:{}_softmax_weight.pkl".format(self.rank))
        self.weight_mom_name = os.path.join(
            self.prefix, "rank:{}_softmax_weight_mom.pkl".format(self.rank))

        if resume:
            try:
                self.weight: paddle.Tensor = paddle.load(self.weight_name)
                print("softmax weight resume successfully!")
            except (FileNotFoundError, KeyError, IndexError):
                self.weight = paddle.normal(0, 0.01, (self.num_local,
                                                      self.embedding_size))
                print("softmax weight resume fail!")

            try:
                self.weight_mom: paddle.Tensor = paddle.load(
                    self.weight_mom_name)
                print("softmax weight mom resume successfully!")
            except (FileNotFoundError, KeyError, IndexError):
                self.weight_mom: paddle.Tensor = paddle.zeros_like(self.weight)
                print("softmax weight mom resume fail!")
        else:
            self.weight = paddle.normal(0, 0.01,
                                        (self.num_local, self.embedding_size))
            self.weight_mom: paddle.Tensor = paddle.zeros_like(self.weight)
            print("softmax weight init successfully!")
            print("softmax weight mom init successfully!")

        self.index = None
        if int(self.sample_rate) == 1:
            self.update = lambda: 0
            self.sub_weight = paddle.create_parameter(
                shape=self.weight.shape,
                dtype='float32',
                default_initializer=paddle.nn.initializer.Assign(self.weight))
            self.sub_weight_mom = self.weight_mom
        else:
            self.sub_weight = paddle.create_parameter(
                shape=[1, 1],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Assign(
                    paddle.empty((1, 1))))

    def save_params(self):
        with open(self.weight_name, 'wb') as file:
            pickle.dump(self.weight.numpy(), file)
        with open(self.weight_mom_name, 'wb') as file:
            pickle.dump(self.weight_mom.numpy(), file)

    @paddle.no_grad()
    def sample(self, total_label):
        index_positive = (self.class_start <= total_label).numpy() & (
            total_label < self.class_start + self.num_local).numpy()
        total_label = total_label.numpy()
        total_label[~index_positive] = -1
        total_label[index_positive] -= self.class_start
        total_label = paddle.to_tensor(total_label)

    def forward(self, total_features, norm_weight):
        logits = linear(total_features, paddle.t(norm_weight))
        return logits

    @paddle.no_grad()
    def update(self):
        self.weight_mom[self.index] = self.sub_weight_mom
        self.weight[self.index] = self.sub_weight

    def prepare(self, label, optimizer):
        # label [64, 1]
        total_label = label.detach()
        self.sample(total_label)
        optimizer._parameter_list[0] = self.sub_weight
        norm_weight = normalize(self.sub_weight)
        return total_label, norm_weight

    def forward_backward(self, label, features, optimizer):
        total_label, norm_weight = self.prepare(label, optimizer)
        total_features = features.detach()
        total_features.stop_gradient = False

        logits = self.forward(total_features, norm_weight)
        logits = self.margin_softmax(logits, total_label)

        with paddle.no_grad():
            max_fc = paddle.max(logits, axis=1, keepdim=True)

            # calculate exp(logits) and all-reduce
            logits_exp = paddle.exp(logits - max_fc)
            logits_sum_exp = logits_exp.sum(axis=1, keepdim=True)

            # calculate prob
            logits_exp = logits_exp.divide(logits_sum_exp)

            # get one-hot
            grad = logits_exp
            one_hot = paddle.nn.functional.one_hot(
                total_label.astype('long'), num_classes=85742)

            # calculate loss
            loss = paddle.nn.functional.one_hot(
                total_label.astype('long'),
                num_classes=85742).multiply(grad).sum(axis=1)
            loss_v = paddle.clip(loss, 1e-30).log().mean() * (-1)

            # calculate grad
            grad -= one_hot
            grad = grad.divide(
                paddle.to_tensor(
                    self.batch_size * self.world_size, dtype='float32'))
        (logits.multiply(grad)).backward()

        x_grad = paddle.to_tensor(total_features.grad, stop_gradient=False)
        return x_grad, loss_v
