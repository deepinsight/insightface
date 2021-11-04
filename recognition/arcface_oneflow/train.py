import argparse
import logging
import os
import oneflow as flow

from function import Trainer
from utils.utils_logging import init_logging
from utils.utils_config import get_config


def main(args):
    cfg = get_config(args.config)
    cfg.graph = args.graph
    rank = flow.env.get_rank()
    world_size = flow.env.get_world_size()
    placement = flow.env.all_device_placement("cuda")

    os.makedirs(cfg.output, exist_ok=True)
    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)

    # root dir of loading checkpoint
    load_path = None

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    trainer = Trainer(cfg, placement, load_path, world_size, rank)
    trainer()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="OneFlow ArcFace Training")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Run model in graph mode,else run model in ddp mode.",
    )
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
