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
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath('.'))


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

    args = parser.parse_args()
    return args


def paddle_inference(args):
    import paddle.inference as paddle_infer

    config = paddle_infer.Config(args.model_file, args.params_file)
    predictor = paddle_infer.create_predictor(config)

    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    img = cv2.imread(args.image_path)
    # normalize to mean 0.5, std 0.5
    img = (img - 127.5) * 0.00784313725
    # BGR2RGB
    img = img[:, :, ::-1]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = img.astype('float32')

    input_handle.copy_from_cpu(img)

    predictor.run()

    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()

    print('paddle inference result: ', output_data.shape)


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
