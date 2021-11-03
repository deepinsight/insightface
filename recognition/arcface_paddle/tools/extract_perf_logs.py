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
import re
import sys
import glob
import json
import argparse
import pprint

import numpy as np

pp = pprint.PrettyPrinter(indent=1)

parser = argparse.ArgumentParser(description="flags for benchmark")
parser.add_argument("--log_dir", type=str, default="./logs/", required=True)
parser.add_argument(
    "--output_dir", type=str, default="./logs/", required=False)
parser.add_argument('--warmup_batches', type=int, default=50)
parser.add_argument('--train_batches', type=int, default=150)

args = parser.parse_args()


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
        return value


def compute_median(iter_dict):
    speed_list = [i for i in iter_dict.values()]
    return round(np.median(speed_list), 2)


def compute_average(iter_dict):
    i = 0
    total_speed = 0
    for iter in iter_dict:
        i += 1
        total_speed += iter_dict[iter]
    return round(total_speed / i, 2)


def extract_info_from_file(log_file, result_dict, speed_dict):
    # extract info from file name
    exp_config = log_file.split("/")[-2]
    model = exp_config.split("_")[2]
    mode = exp_config.split("_")[3]
    precision = exp_config.split("_")[4]
    batch_size_per_device = exp_config.split("_")[6]
    run_case = exp_config.split("_")[7]  # eg: 1n1g
    test_iter = int(exp_config.split("_")[8][2:])
    node_num = int(run_case[0])
    if len(run_case) == 4:
        card_num = int(run_case[-2])
    elif len(run_case) == 5:
        card_num = int(run_case[-3:-1])

    avg_speed_list = []
    # extract info from file content
    with open(log_file) as f:
        lines = f.readlines()
        for line in lines:
            if "throughput:" in line:
                p1 = re.compile(r" ips: ([0-9]+\.[0-9]+)", re.S)
                item = re.findall(p1, line)
                a = float(item[0].strip())
                avg_speed_list.append(a)

    # compute avg throughoutput
    avg_speed = round(
        np.mean(avg_speed_list[args.warmup_batches:args.train_batches]), 2)

    speed_dict[mode][model][run_case][precision][batch_size_per_device][
        test_iter] = avg_speed
    average_speed = compute_average(speed_dict[mode][model][run_case][
        precision][batch_size_per_device])
    median_speed = compute_median(speed_dict[mode][model][run_case][precision][
        batch_size_per_device])

    result_dict[mode][model][run_case][precision][batch_size_per_device][
        'average_speed'] = average_speed
    result_dict[mode][model][run_case][precision][batch_size_per_device][
        'median_speed'] = median_speed

    # print(log_file, speed_dict[mode][model][run_case])


def compute_speedup(result_dict, speed_dict):
    mode_list = [key for key in result_dict]  # eg. ['static', 'dynamic']
    for md in mode_list:
        model_list = [key for key in result_dict[md]]  # eg.['vgg16', 'r50']
        for m in model_list:
            run_case = [key for key in result_dict[md][m]
                        ]  # eg.['4n8g', '2n8g', '1n8g', '1n4g', '1n1g']
            for d in run_case:
                precision = [key for key in result_dict[md][m][d]]
                for p in precision:
                    batch_size_per_device = [
                        key for key in result_dict[md][m][d][p]
                    ]
                    for b in batch_size_per_device:
                        speed_up = 1.0
                        if result_dict[md][m]['1n1g'][p][b]['median_speed']:
                            speed_up = result_dict[md][m][d][p][b][
                                'median_speed'] / result_dict[md][m]['1n1g'][
                                    p][b]['median_speed']
                        result_dict[md][m][d][p][b]['speedup'] = round(
                            speed_up, 2)


def extract_result():
    result_dict = AutoVivification()
    speed_dict = AutoVivification()
    logs_list = glob.glob(os.path.join(args.log_dir, "*/workerlog.0"))
    for l in logs_list:
        extract_info_from_file(l, result_dict, speed_dict)

    # compute speedup
    compute_speedup(result_dict, speed_dict)

    # print result
    pp.pprint(result_dict)

    # write to file as JSON format
    os.makedirs(args.output_dir, exist_ok=True)
    result_file_name = os.path.join(args.output_dir,
                                    "arcface_paddle_result.json")
    print("Saving result to {}".format(result_file_name))
    with open(result_file_name, 'w') as f:
        json.dump(result_dict, f)


if __name__ == "__main__":
    extract_result()
