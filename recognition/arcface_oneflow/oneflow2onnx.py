import os
from os import mkdir
import oneflow.typing as tp
import onnx
import onnxruntime as ort
import numpy as np
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check
import oneflow as flow

import logging
from easydict import EasyDict as edict
from backbones import get_model
from utils.utils_config import get_config
import argparse


def convert_func(cfg, model_path, out_path):
    @flow.global_function()
    def InferenceNet(images: tp.Numpy.Placeholder((1, 3, 112, 112))):

        logits = get_model(cfg.network, images, cfg)
        return logits
    print(convert_to_onnx_and_check(InferenceNet,
          flow_weight_dir=None, onnx_model_path=out_path))


def main(args):
    logging.basicConfig(level=logging.NOTSET)
    logging.info(args.model_path)
    cfg = get_config(args.config)
    if not os.path.exists(args.out_path):
        mkdir(args.out_path)
    convert_func(cfg, args.model_path, args.out_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='OneFlow ArcFace val')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--model_path', type=str, help='model path')
    parser.add_argument('--out_path', type=str,
                        default="onnx_model", help='out path')
    main(parser.parse_args())
