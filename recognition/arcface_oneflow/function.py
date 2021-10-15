
import argparse
import logging
import os

import oneflow as flow
import oneflow.nn as nn

import sys
from backbones import get_model
import math
from utils.utils_config import get_config
import numpy as np
import pickle
import time
from utils.ofrecord_data_utils import load_train_dataset, load_synthetic


class Validator(object):
    def __init__(self, cfg):
        self.cfg = cfg

        def get_val_config():
            config = flow.function_config()
            config.default_logical_view(flow.scope.consistent_view())
            config.default_data_type(flow.float)
            return config
        function_config = get_val_config()

        @flow.global_function(type="predict", function_config=function_config)
        def get_symbol_val_job(
            images: flow.typing.Numpy.Placeholder(
                (self.cfg.val_batch_size, 3, 112, 112)
            )
        ):
            print("val batch data: ", images.shape)
            embedding = get_model(cfg.network, images, cfg)
            return embedding

        self.get_symbol_val_fn = get_symbol_val_job

    def load_checkpoint(self, model_path):
        flow.load_variables(flow.checkpoint.get(model_path))


def get_train_config(cfg):

    cfg.cudnn_conv_heuristic_search_algo = False
    cfg.enable_fuse_model_update_ops = True
    cfg.enable_fuse_add_to_output = True
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.consistent_view())
    func_config.default_data_type(flow.float)
    func_config.cudnn_conv_heuristic_search_algo(
        cfg.cudnn_conv_heuristic_search_algo
    )

    func_config.enable_fuse_model_update_ops(
        cfg.enable_fuse_model_update_ops)
    func_config.enable_fuse_add_to_output(cfg.enable_fuse_add_to_output)
    if cfg.fp16:
        logging.info("Training with FP16 now.")
        func_config.enable_auto_mixed_precision(True)
    if cfg.partial_fc:
        func_config.enable_fuse_model_update_ops(False)
        func_config.indexed_slices_optimizer_conf(
            dict(include_op_names=dict(op_name=['fc7-weight'])))
    if cfg.fp16 and (cfg.num_nodes * cfg.device_num_per_node) > 1:
        flow.config.collective_boxing.nccl_fusion_all_reduce_use_buffer(False)
    if cfg.nccl_fusion_threshold_mb:
        flow.config.collective_boxing.nccl_fusion_threshold_mb(
            cfg.nccl_fusion_threshold_mb)
    if cfg.nccl_fusion_max_ops:
        flow.config.collective_boxing.nccl_fusion_max_ops(
            cfg.nccl_fusion_max_ops)

    return func_config


def make_train_func(cfg):
    @flow.global_function(type="train", function_config=get_train_config(cfg))
    def get_symbol_train_job():
        if cfg.use_synthetic_data:
            (labels, images) = load_synthetic(cfg)
        else:
            labels, images = load_train_dataset(cfg)
        image_size = images.shape[2:]
        assert len(
            image_size) == 2, "The length of image size must be equal to 2."
        assert image_size[0] == image_size[1], "image_size[0] should be equal to image_size[1]."

        embedding = get_model(cfg.network, images, cfg)

        def _get_initializer():
            return flow.random_normal_initializer(mean=0.0, stddev=0.01)

        trainable = True

        if cfg.model_parallel and cfg.device_num_per_node > 1:
            logging.info("Training is using model parallelism now.")
            labels = labels.with_distribute(flow.distribute.broadcast())
            fc1_distribute = flow.distribute.broadcast()
            fc7_data_distribute = flow.distribute.split(1)
            fc7_model_distribute = flow.distribute.split(0)
        else:
            fc1_distribute = flow.distribute.split(0)
            fc7_data_distribute = flow.distribute.split(0)
            fc7_model_distribute = flow.distribute.broadcast()
        weight_regularizer = flow.regularizers.l2(0.0005)
        fc7_weight = flow.get_variable(
            name="fc7-weight",
            shape=(cfg.num_classes, embedding.shape[1]),
            dtype=embedding.dtype,
            initializer=_get_initializer(),
            regularizer=weight_regularizer,
            trainable=trainable,
            model_name="weight",
            distribute=fc7_model_distribute,
        )
        if cfg.partial_fc and cfg.model_parallel:
            logging.info(
                "Training is using model parallelism and optimized by partial_fc now."
            )

            size = cfg.device_num_per_node * cfg.num_nodes
            num_local = (cfg.num_classes + size - 1) // size
            num_sample = int(num_local * cfg.sample_rate)
            total_num_sample = num_sample * size
            (
                mapped_label,
                sampled_label,
                sampled_weight,
            ) = flow.distributed_partial_fc_sample(
                weight=fc7_weight, label=labels, num_sample=total_num_sample,
            )
            labels = mapped_label
            fc7_weight = sampled_weight
        fc7_weight = flow.math.l2_normalize(
            input=fc7_weight, axis=1, epsilon=1e-10)
        fc1 = flow.math.l2_normalize(
            input=embedding, axis=1, epsilon=1e-10)
        fc7 = flow.matmul(
            a=fc1.with_distribute(fc1_distribute), b=fc7_weight, transpose_b=True
        )
        fc7 = fc7.with_distribute(fc7_data_distribute)

        if cfg.loss == "cosface":
            fc7 = (flow.combined_margin_loss(
                fc7, labels, m1=1, m2=0.0, m3=0.4) * 64)
        elif cfg.loss == "arcface":
            fc7 = (flow.combined_margin_loss(
                fc7, labels, m1=1, m2=0.5, m3=0.0) * 64)
        else:
            raise ValueError()

        fc7 = fc7.with_distribute(fc7_data_distribute)

        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, fc7, name="softmax_loss"
        )

        lr_scheduler = flow.optimizer.PiecewiseScalingScheduler(
            base_lr=cfg.lr,
            boundaries=cfg.lr_steps,
            scale=cfg.lr_scales,
            warmup=None
        )
        flow.optimizer.SGD(lr_scheduler,
                           momentum=cfg.momentum if cfg.momentum > 0 else None,
                           ).minimize(loss)

        return loss

    return get_symbol_train_job
