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

import paddle
import paddle.nn.functional as F
from paddle.jit import to_static

import backbones


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str)
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--output_path", type=str, default="./inference")

    return parser.parse_args()


def load_dygraph_pretrain(model, path=None):
    if not os.path.exists(path):
        raise ValueError(f"The path of pretrained model file does not exists: {path}.")
    param_state_dict = paddle.load(path)
    model.set_dict(param_state_dict)
    return


def main():
    args = parse_args()

    net = eval("backbones.{}".format(args.network))()
    load_dygraph_pretrain(net, path=args.pretrained_model)
    net.eval()

    net = to_static(net, input_spec=[paddle.static.InputSpec(shape=[None, 3, 112, 112], dtype='float32')])
    paddle.jit.save(net, os.path.join(args.output_path, "inference"))


if __name__ == "__main__":
    main()