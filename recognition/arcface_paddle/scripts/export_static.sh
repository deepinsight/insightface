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

python tools/export.py \
    --is_static True \
    --export_type paddle \
    --backbone FresResNet50 \
    --embedding_size 512 \
    --checkpoint_dir MS1M_v3_arcface_static_128_fp16_0.1/FresResNet50/24 \
    --output_dir MS1M_v3_arcface_static_128_fp16_0.1/FresResNet50/exported_model
