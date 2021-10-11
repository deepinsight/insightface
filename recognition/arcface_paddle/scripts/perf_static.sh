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

set -e

num_test=5
num_nodes=1

configs=(configs/ms1mv3_r50.py configs/ms1mv3_r100.py)
dtypes=(fp16 fp32)
gpus=("0" "0,1,2,3" "0,1,2,3,4,5,6,7")

for config in "${configs[@]}"
do
    for dtype in "${dtypes[@]}"
    do
        for gpu in "${gpus[@]}"
        do
            i=1
            while [ $i -le ${num_test} ]
            do
                bash scripts/perf_runner.sh $gpu $config static 93431 $dtype $num_nodes 128 0.1 ${i}
                echo " >>>>>>Finished Test Case $config, $dtype, $gpu, ${i} <<<<<<<"
                let i++
                sleep 20s
            done
        done
    done
done
