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

set -ex

gpus=${1:-0,1,2,3,4,5,6,7}
config_file=${2:-configs/ms1mv3_r50.py}
mode=${3:-static}
num_classes=${4:-93431}
dtype=${5:-fp16}
num_nodes=${6:-1}
batch_size_per_device=${7:-128}
sample_ratio=${8:-0.1}
test_id=${9:-1}

if [ $mode = "static" ]; then
    is_static=True
else
    is_static=False
fi

if [ $dtype = "fp16" ]; then
    fp16=True
else
    fp16=False
fi

if [[ $config_file =~ r50 ]]; then
    backbone=r50
else
    backbone=r100
fi

gpu_num_per_node=`expr ${#gpus} / 2 + 1`

log_dir=./logs/arcface_paddle_${backbone}_${mode}_${dtype}_r${sample_ratio}_bz${batch_size_per_device}_${num_nodes}n${gpu_num_per_node}g_id${test_id}

python -m paddle.distributed.launch --gpus=${gpus} --log_dir=${log_dir} tools/train.py \
    --config_file ${config_file} \
    --is_static ${is_static} \
    --num_classes ${num_classes} \
    --fp16 ${fp16} \
    --sample_ratio ${sample_ratio} \
    --log_interval_step 1 \
    --train_unit 'step' \
    --train_num 200 \
    --warmup_num 0 \
    --use_synthetic_dataset True \
    --do_validation_while_train False
