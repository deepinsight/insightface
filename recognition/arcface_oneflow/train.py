import argparse
import logging
import os
import oneflow as flow

from function import Trainer
from utils.utils_logging import init_logging
from utils.utils_config import get_config


def str2bool(v):
    return str(v).lower() in ("true", "t", "1")


def main(args):
    cfg = get_config(args.config)
    cfg.graph = args.graph
    cfg.batch_size = args.batch_size
    cfg.fp16 = args.fp16
    cfg.model_parallel = args.model_parallel
    cfg.train_num = args.train_num
    cfg.channel_last = args.channel_last
    rank = flow.env.get_rank()
    world_size = flow.env.get_world_size()
    placement = flow.env.all_device_placement("cuda")

    os.makedirs(cfg.output, exist_ok=True)
    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)

    # root dir of loading checkpoint
    load_path = args.load_path

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    trainer = Trainer(cfg, placement, load_path, world_size, rank)
    trainer()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="OneFlow ArcFace Training")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Train batch size per device",
    )
    parser.add_argument(
        "--fp16", type=str2bool, default="True", help="Whether to use fp16",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Run model in graph mode,else run model in ddp mode.",
    )
    parser.add_argument(
        "--model_parallel", type=str2bool, default="True", help="Train use model_parallel",
    )
    parser.add_argument(
        "--train_num", type=int, default=1000000, help="Train total num",
    )
    parser.add_argument(
        "--channel_last", type=str2bool, default="False", help="use NHWC",
    )
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    parser.add_argument("--load_path", type=str, default=None,
                        help="root dir of loading checkpoint")
    main(parser.parse_args())
