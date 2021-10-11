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
import logging
sys.path.insert(0, os.path.abspath('.'))

import argparse


def str2bool(v):
    return str(v).lower() in ("true", "t", "1")


def tostrlist(v):
    if isinstance(v, list):
        return v
    elif isinstance(v, str):
        return [e.strip() for e in v.split(',')]


def parse_args():
    parser = argparse.ArgumentParser(description='Paddle Face Exporter')

    # Model setting
    parser.add_argument(
        '--is_static',
        type=str2bool,
        default='False',
        help='whether to use static mode')
    parser.add_argument(
        '--backbone',
        type=str,
        default='FresResNet50',
        help='backbone network')
    parser.add_argument(
        '--embedding_size', type=int, default=512, help='embedding size')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='MS1M_v3_arcface/FresResNet50/24/',
        help='checkpoint direcotry')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./MS1M_v3_bin',
        help='train dataset directory')
    parser.add_argument(
        '--val_targets',
        type=tostrlist,
        default=["lfw", "cfp_fp", "agedb_30"],
        help='val targets, list or str split by comma')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='test batch size')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format="Validation: %(asctime)s - %(message)s")

    args = parse_args()
    if args.is_static:
        import paddle
        paddle.enable_static()
        from static.validation import validation
    else:
        from dynamic.validation import validation

    validation(args)
