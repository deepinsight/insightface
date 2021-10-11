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

import logging
import os
import sys
import time


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_logging(rank, models_root):
    if rank is 0:
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)
        formatter = logging.Formatter("Training: %(asctime)s - %(message)s")
        handler_file = logging.FileHandler(
            os.path.join(models_root, "training.log"))
        handler_stream = logging.StreamHandler(sys.stdout)
        handler_file.setFormatter(formatter)
        handler_stream.setFormatter(formatter)
        log_root.addHandler(handler_file)
        log_root.addHandler(handler_stream)
        log_root.info('rank: %d' % rank)


class CallBackLogging(object):
    def __init__(self,
                 frequent,
                 rank,
                 world_size,
                 total_step,
                 batch_size,
                 writer=None):
        self.frequent: int = frequent
        self.rank: int = rank
        self.world_size: int = world_size
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size
        self.writer = writer

        self.tic = time.time()

    def __call__(self, global_step, loss: AverageMeter, epoch: int, lr_value):

        if self.rank is 0 and global_step > 0 and global_step % self.frequent == 0:
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
            msg = "loss %.4f, lr: %f, epoch: %d, step: %d, eta: %1.2f hours, throughput: %.2f imgs/sec" % (
                loss.avg, lr_value, epoch, global_step, time_for_end,
                speed_total)
            logging.info(msg)
            loss.reset()
            self.tic = time.time()
