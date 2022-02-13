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
import numpy as np
import paddle

from .utils.io import Checkpoint
from . import backbones


def export(args):
    checkpoint = Checkpoint(
        rank=0,
        world_size=1,
        embedding_size=args.embedding_size,
        num_classes=None,
        checkpoint_dir=args.checkpoint_dir, )

    backbone = eval("backbones.{}".format(args.backbone))(
        num_features=args.embedding_size)
    checkpoint.load(backbone, for_train=False, dtype='float32')

    print("Load checkpoint from '{}'.".format(args.checkpoint_dir))
    backbone.eval()

    path = os.path.join(args.output_dir, args.backbone)

    if args.export_type == 'onnx':
        paddle.onnx.export(
            backbone,
            path,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, 3, 112, 112], dtype='float32')
            ])
    else:
        paddle.jit.save(
            backbone,
            path,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, 3, 112, 112], dtype='float32')
            ])
    print("Save exported model to '{}'.".format(args.output_dir))
