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

import sys
import os
import cv2
import time
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

def str2bool(v):
    return v.lower() in ("True","true", "t", "1")

def parse_args():
    parser = argparse.ArgumentParser(description='Paddle Face Predictor')

    parser.add_argument(
        '--export_type', type=str, help='export type, paddle or onnx')
    parser.add_argument(
        "--model_file",
        type=str,
        required=False,
        help="paddle save inference model filename")
    parser.add_argument(
        "--params_file",
        type=str,
        required=False,
        help="paddle save inference parameter filename")
    parser.add_argument(
        "--onnx_file", type=str, required=False, help="onnx model filename")
    parser.add_argument("--image_path", type=str, help="path to test image")
    parser.add_argument("--benchmark", type=str2bool, default=False, help="Is benchmark mode")
    # params for paddle inferece engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--min_subgraph_size", type=int, default=15)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=10)
    args = parser.parse_args()
    return args

def get_infer_gpuid():
    cmd = "nvidia-smi"
    res = os.popen(cmd).readlines()
    if len(res) == 0:
        return None
    cmd = "env | grep CUDA_VISIBLE_DEVICES"
    env_cuda = os.popen(cmd).readlines()
    if len(env_cuda) == 0:
        return 0
    else:
        gpu_id = env_cuda[0].strip().split("=")[1]
        return int(gpu_id[0])


def init_paddle_inference_config(args):
    import paddle.inference as paddle_infer
    config = paddle_infer.Config(args.model_file, args.params_file)
    if hasattr(args, 'precision'):
        if args.precision == "fp16" and args.use_tensorrt:
            precision = paddle_infer.PrecisionType.Half
        elif args.precision == "int8":
            precision = paddle_infer.PrecisionType.Int8
        else:
            precision = paddle_infer.PrecisionType.Float32
    else:
        precision = paddle_infer.PrecisionType.Float32

    if args.use_gpu:
        gpu_id = get_infer_gpuid()
        if gpu_id is None:
            raise ValueError(
                "Not found GPU in current device. Please check your device or set args.use_gpu as False"
            )
        config.enable_use_gpu(args.gpu_mem, 0)
        if args.use_tensorrt:
            config.enable_tensorrt_engine(
                precision_mode=precision,
                max_batch_size=args.max_batch_size,
                min_subgraph_size=args.min_subgraph_size)
            # skip the minmum trt subgraph
            min_input_shape = {"x": [1, 3, 10, 10]}
            max_input_shape = {"x": [1, 3, 1000, 1000]}
            opt_input_shape = {"x": [1, 3, 112, 112]}
            config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
                                            opt_input_shape)

    else:
        config.disable_gpu()
        cpu_threads = args.cpu_threads if  hasattr(args, "cpu_threads") else 10
        config.set_cpu_math_library_num_threads(cpu_threads)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.enable_mkldnn()
            config.set_mkldnn_cache_capacity(10)
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()
    return config


def get_image_file_list(img_file):
    import imghdr
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'GIF'}
    if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and imghdr.what(file_path) in img_end:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists

def paddle_inference(args):
    import paddle.inference as paddle_infer

    config =  init_paddle_inference_config(args)
    predictor = paddle_infer.create_predictor(config)

    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    if args.benchmark:
        import auto_log
        pid = os.getpid()
        autolog = auto_log.AutoLogger(
            model_name="det",
            model_precision='fp32',
            batch_size=1,
            data_shape="dynamic",
            save_path="./output/auto_log.log",
            inference_config=config,
            pids=pid,
            process_name=None,
            gpu_ids=0,
            time_keys=[
                'preprocess_time', 'inference_time','postprocess_time'
            ],
            warmup=0)
        img = np.random.uniform(0, 255, [1, 3, 112,112]).astype(np.float32)
        input_handle.copy_from_cpu(img)
        for i in range(10):
            predictor.run()


    img_list = get_image_file_list(args.image_path)
    for img_path in img_list:
        img = cv2.imread(img_path)
        st = time.time()
        if args.benchmark:
            autolog.times.start()

        # normalize to mean 0.5, std 0.5
        img = (img - 127.5) * 0.00784313725
        # BGR2RGB
        img = img[:, :, ::-1]
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, 0)
        img = img.astype('float32')

        if args.benchmark:
            autolog.times.stamp()

        input_handle.copy_from_cpu(img)

        predictor.run()

        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()
        if args.benchmark:
            autolog.times.stamp()
            autolog.times.end(stamp=True)
            print('{}\t{}'.format(img_path,json.dumps(output_data.tolist())))
        print('paddle inference result: ', output_data.shape)
    if args.benchmark:
        autolog.report()

def onnx_inference(args):
    import onnxruntime

    ort_sess = onnxruntime.InferenceSession(args.onnx_file)

    img = cv2.imread(args.image_path)
    # normalize to mean 0.5, std 0.5
    img = (img - 127.5) * 0.00784313725
    # BGR2RGB
    img = img[:, :, ::-1]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = img.astype('float32')

    ort_inputs = {ort_sess.get_inputs()[0].name: img}
    ort_outs = ort_sess.run(None, ort_inputs)

    print('onnx inference result: ', ort_outs[0].shape)


if __name__ == '__main__':

    args = parse_args()

    assert args.export_type in ['paddle', 'onnx']
    if args.export_type == 'onnx':
        assert os.path.exists(args.onnx_file)
        onnx_inference(args)
    else:
        assert os.path.exists(args.model_file)
        assert os.path.exists(args.params_file)
        paddle_inference(args)
