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

import time
import os
import sys
import numpy as np

import paddle
from visualdl import LogWriter

from utils.logging import AverageMeter, init_logging, CallBackLogging
from utils import losses

from .utils.optimization_pass import gather_optimization_pass, amp_pass

from . import classifiers
from . import backbones


class StaticModel(object):
    def __init__(self,
                 main_program,
                 startup_program,
                 backbone_class_name,
                 embedding_size,
                 classifier_class_name=None,
                 num_classes=None,
                 sample_ratio=0.1,
                 lr_scheduler=None,
                 momentum=0.9,
                 weight_decay=2e-4,
                 dropout=0.4,
                 mode='train',
                 fp16=False,
                 fp16_configs=None,
                 margin_loss_params=None):

        rank = int(os.getenv("PADDLE_TRAINER_ID", 0))
        world_size = int(os.getenv("PADDLE_TRAINERS_NUM", 1))
        if world_size > 1:
            import paddle.distributed.fleet as fleet

        self.main_program = main_program
        self.startup_program = startup_program
        self.backbone_class_name = backbone_class_name
        self.embedding_size = embedding_size
        self.classifier_class_name = classifier_class_name
        self.num_classes = num_classes
        self.sample_ratio = sample_ratio
        self.lr_scheduler = lr_scheduler
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.mode = mode
        self.fp16 = fp16
        self.fp16_configs = fp16_configs
        self.margin_loss_params = margin_loss_params

        if self.mode == 'train':
            assert self.classifier_class_name is not None
            assert self.num_classes is not None
            assert self.lr_scheduler is not None
            assert self.margin_loss_params is not None
            with paddle.static.program_guard(self.main_program,
                                             self.startup_program):
                with paddle.utils.unique_name.guard():
                    self.backbone = eval("backbones.{}".format(
                        self.backbone_class_name))(
                            num_features=self.embedding_size,
                            is_train=True,
                            fp16=self.fp16,
                            dropout=dropout)
                    assert 'label' in self.backbone.input_dict
                    assert 'feature' in self.backbone.output_dict
                    self.classifier = eval("classifiers.{}".format(
                        self.classifier_class_name))(
                            feature=self.backbone.output_dict['feature'],
                            label=self.backbone.input_dict['label'],
                            rank=rank,
                            world_size=world_size,
                            num_classes=self.num_classes,
                            margin1=self.margin_loss_params.margin1,
                            margin2=self.margin_loss_params.margin2,
                            margin3=self.margin_loss_params.margin3,
                            scale=self.margin_loss_params.scale,
                            sample_ratio=self.sample_ratio,
                            embedding_size=self.embedding_size)
                    assert 'loss' in self.classifier.output_dict

                    self.optimizer = paddle.optimizer.Momentum(
                        learning_rate=self.lr_scheduler,
                        momentum=self.momentum,
                        weight_decay=paddle.regularizer.L2Decay(
                            self.weight_decay))
                    if self.fp16:
                        assert self.fp16_configs is not None
                        self.optimizer = paddle.static.amp.decorate(
                            optimizer=self.optimizer,
                            init_loss_scaling=self.fp16_configs[
                                'init_loss_scaling'],
                            incr_every_n_steps=self.fp16_configs[
                                'incr_every_n_steps'],
                            decr_every_n_nan_or_inf=self.fp16_configs[
                                'decr_every_n_nan_or_inf'],
                            incr_ratio=self.fp16_configs['incr_ratio'],
                            decr_ratio=self.fp16_configs['decr_ratio'],
                            use_dynamic_loss_scaling=self.fp16_configs[
                                'use_dynamic_loss_scaling'],
                            use_pure_fp16=self.fp16_configs['use_pure_fp16'],
                            amp_lists=paddle.static.amp.
                            AutoMixedPrecisionLists(
                                custom_white_list=self.fp16_configs[
                                    'custom_white_list'],
                                custom_black_list=self.fp16_configs[
                                    'custom_black_list'], ),
                            use_fp16_guard=False)

                    if world_size > 1:
                        dist_optimizer = fleet.distributed_optimizer(
                            self.optimizer)
                        dist_optimizer.minimize(self.classifier.output_dict[
                            'loss'])
                    else:
                        self.optimizer.minimize(self.classifier.output_dict[
                            'loss'])
                    if self.fp16:
                        self.optimizer = self.optimizer._optimizer
                    if self.sample_ratio < 1.0:
                        gather_optimization_pass(self.main_program,
                                                 'dist@fc@rank')
                    if self.fp16:
                        amp_pass(self.main_program, 'dist@fc@rank')

        elif self.mode == 'test':
            with paddle.static.program_guard(self.main_program,
                                             self.startup_program):
                with paddle.utils.unique_name.guard():
                    self.backbone = eval("backbones.{}".format(
                        self.backbone_class_name))(
                            num_features=self.embedding_size,
                            is_train=False,
                            fp16=self.fp16,
                            dropout=dropout)
                    assert 'feature' in self.backbone.output_dict

        else:
            raise ValueError(
                "mode is error, only support 'train' and 'test' now.")
