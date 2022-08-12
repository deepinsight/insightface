import logging
import os
import time
from typing import List

import torch
import psutil

#from eval import verification
#from partial_fc import PartialFC
#from torch2onnx import convert_onnx
from utils.utils_logging import AverageMeter


class CallBackVerification(object):
    def __init__(self, frequent, rank, val_targets, rec_prefix, image_size=(112, 112)):
        self.frequent: int = frequent
        self.rank: int = rank
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        if self.rank is 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
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
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.rank is 0 and num_update > 0 and num_update % self.frequent == 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()


class CallBackLogging(object):
    def __init__(self, frequent, rank, total_step, batch_size, world_size, writer=None):
        self.frequent: int = frequent
        self.rank: int = rank
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size
        self.world_size: int = world_size
        self.writer = writer

        self.init = False
        self.tic = 0

    def __call__(self, global_step, loss, epoch, fp16, grad_scaler, opt):
        if self.rank is 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                time_now = (time.time() - self.time_start) / 3600.0
                time_total = time_now / ((global_step + 1) / self.total_step)
                time_for_end = time_total - time_now
                lr = opt.param_groups[0]['lr']
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    #self.writer.add_scalar('loss', loss.avg, global_step)
                mem = psutil.virtual_memory()
                mem_used = mem.used / (1024 ** 3)
                loss_str = ""
                for k,v in loss.items():
                    if len(loss_str)!=0:
                        loss_str += "   "
                    loss_str += "%s:%.4f"%(k, v.avg)
                if fp16:
                    msg = "Speed %.2f samples/sec   %s   Epoch: %d   Global Step: %d   LR: %.8f   " \
                            "Fp16 Grad Scale: %2.f   Required: %.1f hours   MemUsed: %.3f" % (
                              speed_total, loss_str, epoch, global_step, lr, grad_scaler.get_scale(), time_for_end, mem_used
                          )
                else:
                    msg = "Speed %.2f samples/sec   %s   Epoch: %d   Global Step: %d   LR: %.8f   Required: %.1f hours   MemUsed: %.3f" % (
                        speed_total, loss_str, epoch, global_step, lr, time_for_end, mem_used
                    )
                logging.info(msg)
                for k,v in loss.items():
                    v.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()


class CallBackModelCheckpoint(object):
    def __init__(self, rank, cfg):
        self.rank = rank
        self.output = cfg.output
        #self.save_pfc = cfg.save_pfc
        #self.save_onnx = cfg.save_onnx
        self.save_opt = cfg.save_opt

    def __call__(self, epoch, backbone, opt_backbone):
        if self.rank == 0:
            path_module = os.path.join(self.output, "backbone_ep%04d.pth"%epoch)
            if self.save_opt:
                data = {
                        'model': backbone.module.state_dict(),
                        'optimizer': opt_backbone.state_dict(),
                        }
            else:
                data = backbone.module.state_dict()
            torch.save(data, path_module)
            logging.info("Pytorch Model Saved in '{}'".format(path_module))


