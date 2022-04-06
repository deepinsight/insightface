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

from .utils.verification import CallBackVerification
from .utils.io import Checkpoint
from . import backbones


def validation(args):
    checkpoint = Checkpoint(
        rank=0,
        world_size=1,
        embedding_size=args.embedding_size,
        num_classes=None,
        checkpoint_dir=args.checkpoint_dir, )

    backbone = eval("backbones.{}".format(args.backbone))(
        num_features=args.embedding_size)
    checkpoint.load(backbone, for_train=False, dtype='float32')
    backbone.eval()

    callback_verification = CallBackVerification(
        1, 0, args.batch_size, args.val_targets, args.data_dir)

    callback_verification(1, backbone)
