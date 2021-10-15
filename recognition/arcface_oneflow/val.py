from utils.utils_logging import AverageMeter, init_logging
import argparse
from function import Validator
from utils.utils_config import get_config
import logging
import os
from backbones import get_model
from utils.utils_callbacks import CallBackVerification
from eval import verification
import oneflow as flow
import sys


def main(args):

    cfg = get_config(args.config)

    logging.basicConfig(level=logging.NOTSET)
    logging.info(args.model_path)
    val_infer = Validator(cfg)
    val_callback = CallBackVerification(
        1, cfg.val_targets, cfg.eval_ofrecord_path, image_nums=cfg.val_image_num)
    val_infer.load_checkpoint(args.model_path)

    val_callback(1000, val_infer.get_symbol_val_fn)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='OneFlow ArcFace val')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--model_path', type=str, help='model path')
    main(parser.parse_args())
