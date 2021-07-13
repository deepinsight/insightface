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

import paddle
from paddle import nn


class CosFace(nn.Layer):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        m_hot = paddle.nn.functional.one_hot(
            label.astype('long'), num_classes=85742) * self.m
        cosine -= m_hot
        ret = cosine * self.s
        return ret


class ArcFace(nn.Layer):
    def __init__(self, s=64.0, m=0.50):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: paddle.Tensor, label):
        m_hot = paddle.nn.functional.one_hot(
            label.astype('long'), num_classes=85742) * self.m
        cosine = cosine.acos()
        cosine += m_hot
        cosine = cosine.cos() * self.s
        return cosine
