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
import time
import argparse
from paddle.inference import Config
from paddle.inference import create_predictor


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # general params
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=1000)

    # params for predict
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--params_file", type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_mkldnn", type=str2bool, default=True)
    parser.add_argument("--cpu_num_threads", type=int, default=10)
    parser.add_argument("--model", type=str)

    return parser.parse_args()


def create_paddle_predictor(args):
    config = Config(args.model_file, args.params_file)

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()

    if args.use_mkldnn:
        config.enable_mkldnn()
        config.set_cpu_math_library_num_threads(args.cpu_num_threads)
        config.set_mkldnn_cache_capacity(100)

    config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

    return predictor


class Predictor(object):
    def __init__(self, args):

        self.args = args

        self.paddle_predictor = create_paddle_predictor(args)
        input_names = self.paddle_predictor.get_input_names()
        self.input_tensor = self.paddle_predictor.get_input_handle(input_names[
            0])

        output_names = self.paddle_predictor.get_output_names()
        self.output_tensor = self.paddle_predictor.get_output_handle(
            output_names[0])

    def predict(self, batch_input):
        self.input_tensor.copy_from_cpu(batch_input)
        self.paddle_predictor.run()
        batch_output = self.output_tensor.copy_to_cpu()
        return batch_output

    def benchmark_predict(self):
        test_num = 500
        test_time = 0.0
        for i in range(0, test_num + 10):
            inputs = np.random.rand(args.batch_size, 3, 112,
                                    112).astype(np.float32)
            start_time = time.time()
            batch_output = self.predict(inputs).flatten()
            if i >= 10:
                test_time += time.time() - start_time
            # time.sleep(0.01)  # sleep for T4 GPU

        print("{0}\tbatch size: {1}\ttime(ms): {2}".format(
            args.model, args.batch_size, 1000 * test_time / test_num))


if __name__ == "__main__":
    args = parse_args()
    assert os.path.exists(
        args.model_file), "The path of 'model_file' does not exist: {}".format(
            args.model_file)
    assert os.path.exists(
        args.params_file
    ), "The path of 'params_file' does not exist: {}".format(args.params_file)

    predictor = Predictor(args)
    assert args.model is not None
    predictor.benchmark_predict()
