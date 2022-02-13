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

import paddle
from collections import OrderedDict

__all__ = [
    "FresResNet", "FresResNet50", "FresResNet100", "FresResNet101",
    "FresResNet152"
]


class FresResNet(object):
    def __init__(self,
                 layers=50,
                 num_features=512,
                 is_train=True,
                 fp16=False,
                 fc_type='E',
                 dropout=0.4):
        super(FresResNet, self).__init__()
        self.layers = layers
        self.num_features = num_features
        self.fc_type = fc_type

        self.input_dict = OrderedDict()
        self.output_dict = OrderedDict()

        image = paddle.static.data(
            name='image',
            shape=[-1, 3, 112, 112],
            dtype='float16' if fp16 else 'float32')
        self.input_dict['image'] = image
        if is_train:
            label = paddle.static.data(name='label', shape=[-1], dtype='int32')
            self.input_dict['label'] = label

        supported_layers = [50, 100, 101, 152]
        assert layers in supported_layers, \
            "supported layers {}, but given {}".format(supported_layers, layers)

        if layers == 50:
            units = [3, 4, 14, 3]
        elif layers == 100:
            units = [3, 13, 30, 3]
        elif layers == 101:
            units = [3, 4, 23, 3]
        elif layers == 152:
            units = [3, 8, 36, 3]
        filter_list = [64, 64, 128, 256, 512]
        num_stages = 4

        input_blob = paddle.static.nn.conv2d(
            input=image,
            num_filters=filter_list[0],
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            param_attr=paddle.ParamAttr(),
            bias_attr=False)
        input_blob = paddle.static.nn.batch_norm(
            input=input_blob,
            act=None,
            epsilon=1e-05,
            momentum=0.9,
            is_test=False if is_train else True)
        # input_blob = paddle.nn.functional.relu6(input_blob)
        input_blob = paddle.static.nn.prelu(
            input_blob,
            mode="all",
            param_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.25)))

        for i in range(num_stages):
            for j in range(units[i]):
                input_blob = self.residual_unit_v3(
                    input_blob,
                    filter_list[i + 1],
                    3,
                    2 if j == 0 else 1,
                    1,
                    is_train, )
        fc1 = self.get_fc1(input_blob, is_train, dropout)

        self.output_dict['feature'] = fc1

    def residual_unit_v3(self, in_data, num_filter, filter_size, stride, pad,
                         is_train):

        bn1 = paddle.static.nn.batch_norm(
            input=in_data,
            act=None,
            epsilon=1e-05,
            momentum=0.9,
            is_test=False if is_train else True)
        conv1 = paddle.static.nn.conv2d(
            input=bn1,
            num_filters=num_filter,
            filter_size=filter_size,
            stride=1,
            padding=1,
            groups=1,
            param_attr=paddle.ParamAttr(),
            bias_attr=False)
        bn2 = paddle.static.nn.batch_norm(
            input=conv1,
            act=None,
            epsilon=1e-05,
            momentum=0.9,
            is_test=False if is_train else True)
        # prelu = paddle.nn.functional.relu6(bn2)
        prelu = paddle.static.nn.prelu(
            bn2,
            mode="all",
            param_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.25)))
        conv2 = paddle.static.nn.conv2d(
            input=prelu,
            num_filters=num_filter,
            filter_size=filter_size,
            stride=stride,
            padding=pad,
            groups=1,
            param_attr=paddle.ParamAttr(),
            bias_attr=False)
        bn3 = paddle.static.nn.batch_norm(
            input=conv2,
            act=None,
            epsilon=1e-05,
            momentum=0.9,
            is_test=False if is_train else True)

        if stride == 1:
            input_blob = in_data
        else:
            input_blob = paddle.static.nn.conv2d(
                input=in_data,
                num_filters=num_filter,
                filter_size=1,
                stride=stride,
                padding=0,
                groups=1,
                param_attr=paddle.ParamAttr(),
                bias_attr=False)

            input_blob = paddle.static.nn.batch_norm(
                input=input_blob,
                act=None,
                epsilon=1e-05,
                momentum=0.9,
                is_test=False if is_train else True)

        identity = paddle.add(bn3, input_blob)
        return identity

    def get_fc1(self, last_conv, is_train, dropout=0.4):
        body = last_conv
        if self.fc_type == "Z":
            body = paddle.static.nn.batch_norm(
                input=body,
                act=None,
                epsilon=1e-05,
                is_test=False if is_train else True)
            if dropout > 0:
                body = paddle.nn.functional.dropout(
                    x=body,
                    p=dropout,
                    training=is_train,
                    mode='upscale_in_train')
            fc1 = body
        elif self.fc_type == "E":
            body = paddle.static.nn.batch_norm(
                input=body,
                act=None,
                epsilon=1e-05,
                is_test=False if is_train else True)
            if dropout > 0:
                body = paddle.nn.functional.dropout(
                    x=body,
                    p=dropout,
                    training=is_train,
                    mode='upscale_in_train')
            fc1 = paddle.static.nn.fc(
                x=body,
                size=self.num_features,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.XavierNormal(
                        fan_in=0.0)),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant()))
            fc1 = paddle.static.nn.batch_norm(
                input=fc1,
                act=None,
                epsilon=1e-05,
                is_test=False if is_train else True)

        elif self.fc_type == "FC":
            body = paddle.static.nn.batch_norm(
                input=body,
                act=None,
                epsilon=1e-05,
                is_test=False if is_train else True)
            fc1 = paddle.static.nn.fc(
                x=body,
                size=self.num_features,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.XavierNormal(
                        fan_in=0.0)),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant()))
            fc1 = paddle.static.nn.batch_norm(
                input=fc1,
                act=None,
                epsilon=1e-05,
                is_test=False if is_train else True)

        return fc1


def FresResNet50(**args):
    model = FresResNet(layers=50, **args)
    return model


def FresResNet100(**args):
    model = FresResNet(layers=100, **args)
    return model


def FresResNet101(**args):
    model = FresResNet(layers=101, **args)
    return model


def FresResNet152(**args):
    model = FresResNet(layers=152, **args)
    return model
