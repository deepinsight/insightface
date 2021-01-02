"""
Author: {Xiang An, XuHan Zhu, Yang Xiao} in DeepGlint,
Partial FC: Training 10 Million Identities on a Single Machine
See the original paper:
https://arxiv.org/abs/2010.05222
"""

import argparse
import logging
import os
import sys
import time

import horovod.mxnet as hvd
import mxnet as mx

import default
from callbacks import CallBackModelSave, CallBackLogging, CallBackCenterSave, CallBackVertification
from default import config
from image_iter import FaceImageIter, DummyIter
from memory_module import SampleDistributeModule
from memory_bank import MemoryBank
from memory_scheduler import get_scheduler
from memory_softmax import MarginLoss
from optimizer import MemoryBankSGDOptimizer
from symbol import resnet

sys.path.append(os.path.join(os.path.dirname(__file__), 'symbol'))
os.environ['MXNET_BACKWARD_DO_MIRROR'] = '0'
os.environ['MXNET_UPDATE_ON_KVSTORE'] = "0"
os.environ['MXNET_EXEC_ENABLE_ADDTO'] = "1"
os.environ['MXNET_USE_TENSORRT'] = "0"
os.environ['MXNET_GPU_WORKER_NTHREADS'] = "2"
os.environ['MXNET_GPU_COPY_NTHREADS'] = "1"
os.environ['MXNET_OPTIMIZER_AGGREGATION_SIZE'] = "54"
os.environ['HOROVOD_CYCLE_TIME'] = "0.1"
os.environ['HOROVOD_FUSION_THRESHOLD'] = "67108864"
os.environ['HOROVOD_NUM_NCCL_STREAMS'] = "2"
os.environ['MXNET_HOROVOD_NUM_GROUPS'] = "16"
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD'] = "999"
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD'] = "25"


def parse_args():
    parser = argparse.ArgumentParser(description='Train parall face network')
    # general
    parser.add_argument('--dataset', default='emore', help='dataset config')
    parser.add_argument('--network', default='r100', help='network config')
    parser.add_argument('--loss', default='cosface', help='loss config')

    args, rest = parser.parse_known_args()
    default.generate_config(args.loss, args.dataset, args.network)
    parser.add_argument('--models-root',
                        default="./test",
                        help='root directory to save model.')
    args = parser.parse_args()
    return args


def set_logger(logger, rank, models_root):
    formatter = logging.Formatter("rank-id:" + str(rank) +
                                  ":%(asctime)s-%(message)s")
    file_handler = logging.FileHandler(
        os.path.join(models_root, "%d_hist.log" % rank))
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info('rank_id: %d' % rank)


def get_symbol_embedding():
    embedding = eval(config.net_name).get_symbol()
    all_label = mx.symbol.Variable('softmax_label')
    all_label = mx.symbol.BlockGrad(all_label)
    out_list = [embedding, all_label]
    out = mx.symbol.Group(out_list)
    return out, embedding


def train_net():
    args = parse_args()
    hvd.init()

    # Size is the number of total GPU, rank is the unique process(GPU) ID from 0 to size,
    # local_rank is the unique process(GPU) ID within this server
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    size = hvd.size()

    prefix = os.path.join(args.models_root, 'model')
    prefix_dir = os.path.dirname(prefix)
    if not os.path.exists(prefix_dir) and not local_rank:
        os.makedirs(prefix_dir)
    else:
        time.sleep(2)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    set_logger(logger, rank, prefix_dir)

    data_shape = (3, config.image_size, config.image_size)

    # We equally store the class centers (softmax linear transformation matrix) on all GPUs in order.
    num_local = (config.num_classes + size - 1) // size
    num_sample = int(num_local * config.sample_ratio)
    memory_bank = MemoryBank(
        num_sample=num_sample,
        num_local=num_local,
        rank=rank,
        local_rank=local_rank,
        embedding_size=config.embedding_size,
        prefix=prefix_dir,
        gpu=True)

    if config.debug:
        train_iter = DummyIter(config.batch_size, data_shape, 1000 * 10000)
    else:
        train_iter = FaceImageIter(
            batch_size=config.batch_size,
            data_shape=data_shape,
            path_imgrec=config.rec,
            shuffle=True,
            rand_mirror=True,
            context=rank,
            context_num=size)
    train_data_iter = mx.io.PrefetchingIter(train_iter)

    esym, save_symbol = get_symbol_embedding()
    margins = (config.loss_m1, config.loss_m2, config.loss_m3)
    fc7_model = MarginLoss(margins, config.loss_s, config.embedding_size)

    # optimizer
    # backbone  lr_scheduler & optimizer
    backbone_lr_scheduler, memory_bank_lr_scheduler = get_scheduler()

    backbone_kwargs = {
        'learning_rate': config.backbone_lr,
        'momentum': 0.9,
        'wd': 5e-4,
        'rescale_grad': 1.0 / (config.batch_size * size) * size,
        'multi_precision': config.fp16,
        'lr_scheduler': backbone_lr_scheduler,
    }

    # memory_bank lr_scheduler & optimizer
    memory_bank_optimizer = MemoryBankSGDOptimizer(
        lr_scheduler=memory_bank_lr_scheduler,
        rescale_grad=1.0 / config.batch_size / size,
    )
    #
    train_module = SampleDistributeModule(
        symbol=esym,
        fc7_model=fc7_model,
        memory_bank=memory_bank,
        memory_optimizer=memory_bank_optimizer)
    #
    if not config.debug and local_rank == 0:
        cb_vert = CallBackVertification(esym, train_module)
    cb_speed = CallBackLogging(rank, size, prefix_dir)
    cb_save = CallBackModelSave(save_symbol, train_module, prefix, rank)
    cb_center_save = CallBackCenterSave(memory_bank)

    def call_back_fn(params):
        cb_speed(params)
        if not config.debug and local_rank == 0:
            cb_vert(params)
        cb_center_save(params)
        cb_save(params)

    train_module.fit(
        train_data_iter,
        optimizer_params=backbone_kwargs,
        initializer=mx.init.Normal(0.1),
        batch_end_callback=call_back_fn)


if __name__ == '__main__':
    train_net()
