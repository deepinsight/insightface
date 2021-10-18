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

python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 tools/train.py \
    --config_file configs/ms1mv3_r50.py \
    --is_static True \
    --backbone FresResNet50 \
    --classifier LargeScaleClassifier \
    --embedding_size 512 \
    --model_parallel True \
    --dropout 0.0 \
    --sample_ratio 0.1 \
    --loss ArcFace \
    --batch_size 128 \
    --dataset MS1M_v3 \
    --num_classes 93431 \
    --data_dir MS1M_v3/ \
    --label_file MS1M_v3/label.txt \
    --is_bin False \
    --log_interval_step 100 \
    --validation_interval_step 2000 \
    --fp16 True \
    --use_dynamic_loss_scaling True \
    --init_loss_scaling 27648.0 \
    --num_workers 8 \
    --train_unit 'epoch' \
    --warmup_num 0 \
    --train_num 25 \
    --decay_boundaries "10,16,22" \
    --output MS1M_v3_arcface_static_0.1
