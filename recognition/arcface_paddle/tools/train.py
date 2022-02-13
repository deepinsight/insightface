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
sys.path.insert(0, os.path.abspath('.'))

import paddle
from configs import argparser as parser
from utils.logging import init_logging

if __name__ == '__main__':
    args = parser.parse_args()
    if args.is_static:
        from static.train import train
        paddle.enable_static()
    else:
        from dynamic.train import train

    rank = int(os.getenv("PADDLE_TRAINER_ID", 0))
    os.makedirs(args.output, exist_ok=True)
    init_logging(rank, args.output)
    parser.print_args(args)
    train(args)
