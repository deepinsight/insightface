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
config.is_static = False
config.backbone = 'MobileFaceNet_128'
config.classifier = 'LargeScaleClassifier'
config.embedding_size = 128
config.model_parallel = True
config.sample_ratio = 1.0
config.loss = 'ArcFace'
config.dropout = 0.0

config.lr = 0.1  # for global batch size = 512
config.lr_decay = 0.1
config.weight_decay = 5e-4
config.momentum = 0.9
config.train_unit = 'epoch'  # 'step' or 'epoch'
config.warmup_num = 0
config.train_num = 25
config.decay_boundaries = [10, 16, 22]

config.use_synthetic_dataset = False
config.dataset = "MS1M_v2"
config.data_dir = "./MS1M_v2"
config.label_file = "./MS1M_v2/label.txt"
config.is_bin = False
config.num_classes = 85742  # 85742 for MS1M_v2, 93431 for MS1M_v3
config.batch_size = 128  # global batch size 1024 of 8 GPU
config.num_workers = 8

config.do_validation_while_train = True
config.validation_interval_step = 2000
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

config.logdir = './log'
config.log_interval_step = 100
config.output = './MS1M_v2_arcface_MobileFaceNet_128_0.1'
config.resume = False
config.checkpoint_dir = None
config.max_num_last_checkpoint = 1
