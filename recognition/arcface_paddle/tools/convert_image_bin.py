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
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import argparse
import cv2
from datasets.kv_helper import read_img_from_bin
from datasets.kv_helper import trans_img_to_bin


def get_file_list(img_file, end=('jpg', 'png', 'jpeg', 'JPEG', 'JPG', 'bmp')):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    if os.path.isfile(img_file) and img_file.split('.')[-1] in end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            if single_file.split('.')[-1] in end:
                imgs_lists.append(os.path.join(img_file, single_file))
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    return imgs_lists


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--bin_path", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="image2bin",
        help="conversion mode, image2bin or bin2image")
    return parser.parse_args()


def main(args):
    assert args.mode in ["image2bin", "bin2image"]
    os.makedirs(args.image_path, exist_ok=True)
    os.makedirs(args.bin_path, exist_ok=True)
    assert os.path.isdir(args.image_path)
    assert os.path.isdir(args.bin_path)

    if args.mode == "image2bin":
        img_list = get_file_list(args.image_path)
        for idx, img_fp in enumerate(img_list):
            if idx % len(img_list) == 1000:
                print("conversion process: [{}]/[{}]".format(idx,
                                                             len(img_list)))
            img_name = os.path.basename(img_fp)
            output_path = os.path.join(args.bin_path,
                                       os.path.splitext(img_name)[0] + ".bin")
            trans_img_to_bin(img_fp, output_path)
    elif args.mode == "bin2image":
        bin_list = get_file_list(args.bin_path, end=("bin", ))
        for idx, bin_fp in enumerate(bin_list):
            if idx % len(bin_list) == 1000:
                print("conversion process: [{}]/[{}]".format(idx,
                                                             len(bin_list)))
            bin_name = os.path.basename(bin_fp)
            output_path = os.path.join(args.image_path,
                                       os.path.splitext(bin_name)[0] + ".jpg")
            img = read_img_from_bin(bin_fp)
            cv2.imwrite(output_path, img)

    print("ok..")


if __name__ == "__main__":
    args = parse_args()
    main(args)
