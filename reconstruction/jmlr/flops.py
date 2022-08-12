from ptflops import get_model_complexity_info
import os
import argparse
from utils.utils_config import get_config
from backbones import get_network

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='JMLR FLOPs')
    parser.add_argument('config', type=str, help='input config file')
    args = parser.parse_args()
    args = parser.parse_args()
    cfg = get_config(args.config)
    #backbone = get_model(cfg.network, num_features=cfg.embedding_size, input_size=cfg.input_size, dropout=cfg.dropout, stem_type=cfg.stem_type, fp16=0)
    net = get_network(cfg)
    macs, params = get_model_complexity_info(
        net, (3, cfg.input_size, cfg.input_size), as_strings=True,
        print_per_layer_stat=True, verbose=True)
    print(macs)
    print(params)

    # from torch import distributed
    # distributed.AllreduceOptions
    # distributed.AllreduceCoalescedOptions
    # distributed.all_reduce
