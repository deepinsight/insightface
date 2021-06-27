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

from easydict import EasyDict as edict

config = edict()
config.dataset = "emore"
config.sample_rate = 1
config.momentum = 0.9

config.data_dir = "./MS1M_bin"
config.file_list = "MS1M_bin/label.txt"
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 32
config.warmup_epoch = 1
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]


def lr_step_func(epoch):
    return ((epoch + 1) / (4 + 1))**2 if epoch < -1 else 0.1**len(
        [m for m in [6, 12, 18, 24] if m - 1 <= epoch])


config.lr_func = lr_step_func
