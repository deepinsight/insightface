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
from typing import List
import paddle
import logging
from eval import verification
from utils.utils_logging import AverageMeter
from partial_fc import PartialFC
import time


class CallBackVerification(object):
    def __init__(self,
                 frequent,
                 rank,
                 val_targets,
                 rec_prefix,
                 image_size=(112, 112)):
        self.frequent: int = frequent
        self.rank: int = rank
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        if self.rank == 0:
            self.init_dataset(
                val_targets=val_targets,
                data_dir=rec_prefix,
                image_size=image_size)

    def ver_test(self,
                 backbone: paddle.nn.Layer,
                 global_step: int,
                 batch_size: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, batch_size, 10)
            logging.info('[%s][%d]XNorm: %f' %
                         (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' %
                         (self.ver_name_list[i], global_step, acc2, std2))
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info('[%s][%d]Accuracy-Highest: %1.5f' % (
                self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: paddle.nn.Layer, batch_size=10):
        if self.rank == 0 and num_update > 0 and num_update % self.frequent == 0:
            backbone.eval()
            self.ver_test(backbone, num_update, batch_size)
            backbone.train()


class CallBackLogging(object):
    def __init__(self,
                 frequent,
                 rank,
                 total_step,
                 batch_size,
                 world_size,
                 writer=None):
        self.frequent: int = frequent
        self.rank: int = rank
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size
        self.world_size: int = world_size
        self.writer = writer

        self.init = False
        self.tic = 0

    def __call__(self,
                 global_step,
                 loss: AverageMeter,
                 epoch: int,
                 lr_backbone_value,
                 lr_pfc_value):
        if self.rank is 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (
                        time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                time_now = (time.time() - self.time_start) / 3600
                time_total = time_now / ((global_step + 1) / self.total_step)
                time_for_end = time_total - time_now
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end,
                                           global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                msg = "Speed %.2f samples/sec   Loss %.4f   Epoch: %d   Global Step: %d   Required: %1.f hours, lr_backbone_value: %f, lr_pfc_value: %f" % (
                    speed_total, loss.avg, epoch, global_step, time_for_end,
                    lr_backbone_value, lr_pfc_value)
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()


class CallBackModelCheckpoint(object):
    def __init__(self, rank, output="./", model_name="mobilefacenet"):
        self.rank: int = rank
        self.output: str = output
        self.model_name: str = model_name

    def __call__(self,
                 global_step,
                 backbone: paddle.nn.Layer,
                 partial_fc: PartialFC=None):
        if global_step > 100 and self.rank is 0:
            paddle.save(backbone.state_dict(),
                        os.path.join(self.output,
                                     self.model_name + ".pdparams"))
        if global_step > 100 and partial_fc is not None:
            partial_fc.save_params()
