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
import paddle
import argparse
import backbones
from utils.utils_callbacks import CallBackVerification


def main(args):
    '''
    For the CallBackVerification class, you can place you val_dataset,
    like ["lfw"], also you can use ["lfw", "cplfw", "calfw"].
    
    For the callback_verification function, the batch_size must be divisible by 12000!
    Cause the length of dataset is 12000.
    '''
    backbone = eval("backbones.{}".format(args.network))()
    model_params = args.network + '.pdparams'
    print('INFO:' + args.network + ' chose! ' + model_params + ' loaded!')
    state_dict = paddle.load(os.path.join(args.checkpoint, model_params))
    backbone.set_state_dict(state_dict)
    callback_verification = CallBackVerification(
        1, 0, ["lfw", "cfp_fp", "agedb_30"], "MS1M_v2")
    callback_verification(1, backbone, batch_size=50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paddle ArcFace Testing')
    parser.add_argument(
        '--network',
        type=str,
        default='MobileFaceNet_128',
        help='backbone network')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='emore_arcface',
        help='checkpoint dir')
    args = parser.parse_args()
    main(args)
