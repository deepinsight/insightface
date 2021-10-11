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

import errno
import os
import numpy as np
import paddle

from .utils.verification import CallBackVerification
from .utils.io import Checkpoint
from .static_model import StaticModel

from . import backbones


def validation(args):
    checkpoint = Checkpoint(
        rank=0,
        world_size=1,
        embedding_size=args.embedding_size,
        num_classes=None,
        checkpoint_dir=args.checkpoint_dir, )

    test_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    test_model = StaticModel(
        main_program=test_program,
        startup_program=startup_program,
        backbone_class_name=args.backbone,
        embedding_size=args.embedding_size,
        mode='test', )

    gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
    place = paddle.CUDAPlace(gpu_id)
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    checkpoint.load(program=test_program, for_train=False)

    callback_verification = CallBackVerification(
        1, 0, args.batch_size, test_program,
        list(test_model.backbone.input_dict.values()),
        list(test_model.backbone.output_dict.values()), args.val_targets,
        args.data_dir)

    callback_verification(1)
