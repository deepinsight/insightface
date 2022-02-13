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

from .utils.io import Checkpoint
from . import backbones
from .static_model import StaticModel


def export_onnx(path_prefix, feed_vars, fetch_vars, executor, program):

    from paddle2onnx.graph import PaddleGraph, ONNXGraph
    from paddle2onnx.passes import PassManager

    opset_version = 10
    enable_onnx_checker = True
    verbose = False

    paddle_graph = PaddleGraph.build_from_program(program, feed_vars,
                                                  fetch_vars,
                                                  paddle.fluid.global_scope())

    onnx_graph = ONNXGraph.build(paddle_graph, opset_version, verbose)
    onnx_graph = PassManager.run_pass(onnx_graph, ['inplace_node_pass'])

    onnx_proto = onnx_graph.export_proto(enable_onnx_checker)

    try:
        # mkdir may conflict if pserver and trainer are running on the same machine
        dirname = os.path.dirname(path_prefix)
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    model_path = path_prefix + ".onnx"
    if os.path.isdir(model_path):
        raise ValueError("'{}' is an existing directory.".format(model_path))

    with open(model_path, 'wb') as f:
        f.write(onnx_proto.SerializeToString())


def export(args):
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

    checkpoint.load(program=test_program, for_train=False, dtype='float32')
    print("Load checkpoint from '{}'.".format(args.checkpoint_dir))

    path = os.path.join(args.output_dir, args.backbone)
    if args.export_type == 'onnx':
        feed_vars = [test_model.backbone.input_dict['image'].name]
        fetch_vars = [test_model.backbone.output_dict['feature']]
        export_onnx(path, feed_vars, fetch_vars, exe, program=test_program)
    else:
        feed_vars = [test_model.backbone.input_dict['image']]
        fetch_vars = [test_model.backbone.output_dict['feature']]
        paddle.static.save_inference_model(
            path, feed_vars, fetch_vars, exe, program=test_program)
    print("Save exported model to '{}'.".format(args.output_dir))
