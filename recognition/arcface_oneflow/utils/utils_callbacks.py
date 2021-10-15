import logging
import os
import time
from typing import List

import oneflow as flow

from eval import verification
from utils.utils_logging import AverageMeter



class CallBackVerification(object):
    def __init__(self, frequent,  val_targets, rec_prefix, image_size=(112, 112),world_size=1):
        self.frequent: int = frequent

        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.world_size=world_size
        
        self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)


    def ver_test(self, backbone: flow.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
   
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            results.append(acc2)


    def init_dataset(self, val_targets, data_dir, image_size):

        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")           
            if os.path.exists(path):
                data_set = verification.load_bin_cv(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)


    def __call__(self, num_update, backbone):
        if num_update > 0 and num_update % self.frequent == 0:
            self.ver_test(backbone, num_update)



class CallBackLogging(object):
    def __init__(self, frequent,  total_step, batch_size, world_size, writer=None):
        self.frequent: int = frequent

        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size
        self.world_size: int = world_size
        self.writer = writer

        self.init = False
        self.tic = 0
        self.losses=AverageMeter()

    def  metric_cb(self,
                 global_step: int,
                 epoch: int,
                 learning_rate: float):
        def callback(loss):
            loss=loss.mean()
            self.losses.update(loss, 1)
            if  global_step % self.frequent == 0:

                if self.init:
                    try:
                        speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                        speed_total = speed * self.world_size
                    except ZeroDivisionError:
                        speed_total = float('inf')
    
                    time_now = (time.time() - self.time_start) / 3600
                    time_total = time_now / ((global_step + 1) / self.total_step)
                    time_for_end = time_total - time_now
                    if self.writer is not None:
                        self.writer.add_scalar('time_for_end', time_for_end, global_step)
                        self.writer.add_scalar('learning_rate', learning_rate, global_step)
                        self.writer.add_scalar('loss', loss.avg, global_step)
                    else:
                        msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.4f   Epoch: %d   Global Step: %d   " \
                              "Required: %1.f hours" % (
                                  speed_total, self.losses.avg, learning_rate, epoch, global_step, time_for_end
                              )
                    logging.info(msg)
                    self.losses.reset()
                    self.tic = time.time()
                else:
                    self.init = True
                    self.tic = time.time()
        return callback


