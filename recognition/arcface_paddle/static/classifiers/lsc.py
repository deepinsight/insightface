# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from six.moves import reduce
from collections import OrderedDict

import paddle

__all__ = ["LargeScaleClassifier"]


class LargeScaleClassifier(object):
    """
    Author: {Xiang An, Yang Xiao, XuHan Zhu} in DeepGlint,
    Partial FC: Training 10 Million Identities on a Single Machine
    See the original paper:
    https://arxiv.org/abs/2010.05222
    """

    def __init__(self,
                 feature,
                 label,
                 rank,
                 world_size,
                 num_classes,
                 margin1=1.0,
                 margin2=0.5,
                 margin3=0.0,
                 scale=64.0,
                 sample_ratio=1.0,
                 embedding_size=512,
                 name=None):
        super(LargeScaleClassifier, self).__init__()
        self.num_classes: int = num_classes
        self.rank: int = rank
        self.world_size: int = world_size
        self.sample_ratio: float = sample_ratio
        self.embedding_size: int = embedding_size
        self.num_local: int = (num_classes + world_size - 1) // world_size
        if num_classes % world_size != 0 and rank == world_size - 1:
            self.num_local = num_classes % self.num_local
        self.num_sample: int = int(self.sample_ratio * self.num_local)
        self.margin1 = margin1
        self.margin2 = margin2
        self.margin3 = margin3
        self.logit_scale = scale

        self.input_dict = OrderedDict()
        self.input_dict['feature'] = feature
        self.input_dict['label'] = label

        self.output_dict = OrderedDict()

        if name is None:
            name = 'dist@fc@rank@%05d.w' % rank
        assert '.w' in name

        stddev = math.sqrt(2.0 / (self.embedding_size + self.num_local))
        param_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Normal(std=stddev))

        weight_dtype = 'float16' if feature.dtype == paddle.float16 else 'float32'
        weight = paddle.static.create_parameter(
            shape=[self.embedding_size, self.num_local],
            dtype=weight_dtype,
            name=name,
            attr=param_attr,
            is_bias=False)

        # avoid allreducing gradients for distributed parameters
        weight.is_distributed = True
        # avoid broadcasting distributed parameters in startup program
        paddle.static.default_startup_program().global_block().vars[
            weight.name].is_distributed = True

        if self.world_size > 1:
            feature_list = []
            paddle.distributed.all_gather(feature_list, feature)
            total_feature = paddle.concat(feature_list, axis=0)

            label_list = []
            paddle.distributed.all_gather(label_list, label)
            total_label = paddle.concat(label_list, axis=0)
            total_label.stop_gradient = True
        else:
            total_feature = feature
            total_label = label

        total_label.stop_gradient = True

        if self.sample_ratio < 1.0:
            # partial fc sample process
            total_label, sampled_class_index = paddle.nn.functional.class_center_sample(
                total_label, self.num_local, self.num_sample)
            sampled_class_index.stop_gradient = True
            weight = paddle.gather(weight, sampled_class_index, axis=1)

        norm_feature = paddle.fluid.layers.l2_normalize(total_feature, axis=1)
        norm_weight = paddle.fluid.layers.l2_normalize(weight, axis=0)

        local_logit = paddle.matmul(norm_feature, norm_weight)

        loss = paddle.nn.functional.margin_cross_entropy(
            local_logit,
            total_label,
            margin1=self.margin1,
            margin2=self.margin2,
            margin3=self.margin3,
            scale=self.logit_scale,
            return_softmax=False,
            reduction=None, )

        loss.desc.set_dtype(paddle.fluid.core.VarDesc.VarType.FP32)
        loss = paddle.mean(loss)

        self.output_dict['loss'] = loss
