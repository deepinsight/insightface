from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import logging
import sys
import numbers
import math
import sklearn
import datetime
import numpy as np
import cv2

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.mxnet import DALIClassificationIterator


logger = logging.getLogger()


class HybridTrainPipe(Pipeline):
    # TODO: 这里还要添加个数据增强，dali提供了很多基础的数据增强方式：https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html?highlight=ops#support-table
    # 因为太懒（菜）这里只添加了 random_mirror
    def __init__(self, path_imgrec, batch_size, num_threads, device_id, num_gpus, initial_fill):
        '''
        initial_fill: 太大会占用内存，太小导致单个 batch id 重复率高而 loss 下降太慢，测试了下 batch_size*1000 基本不影响到训练
        num_threads: 经测试，单核3.5GHz的U，hhd设置为3～4,ssd设置为5～6
        '''
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id)
        logging.info('loading recordio %s...', path_imgrec)
        path_imgidx = path_imgrec[0:-4] + ".idx"
        self.input = ops.MXNetReader(path = [path_imgrec], index_path=[path_imgidx],
                                     random_shuffle = True, shard_id = device_id, num_shards = num_gpus,
                                     prefetch_queue_depth = 5, initial_fill = initial_fill)
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.res = ops.Resize(device="gpu", resize_x=112, resize_y=112)
        self.rrc = ops.RandomResizedCrop(device = "gpu", size = (112, 112))
        # self.cmnp = ops.CropMirrorNormalize(device = "gpu",
        #                                     dtype = types.FLOAT,
        #                                     output_layout = types.NCHW,
        #                                     mean = [0.485 * 255,0.456 * 255,0.406 * 255],
        #                                     std = [0.229 * 255,0.224 * 255,0.225 * 255])
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            dtype = types.FLOAT,
                                            output_layout = types.NCHW)
        self.coin = ops.CoinFlip(probability = 0.5)


    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name = "Reader")
        # TODO: 这部分是问题最大的地方，原始的.rec开始和结尾都记录着其他信息，
        # 一旦读到空图像会 raise RuntimeError，并提示 'pipline broken'，无法 reset pipline，
        # 尝试了加 try 啥的都不行，大佬看看有没有啥解决方案
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror = rng)
        return [output, self.labels]


if __name__ == "__main__":
    path_imgrec = '/home/ps/data/src_data/glint360k/train.rec'
    batch_size = 128
    N = 4
    # 多卡测试，速度和单卡一样，也是18000samples/s，可能主要卡在 SSD 读取速度上了，2080Ti GPU占用20%左右
    # 测试 HHD 8000 samples/s, SSD 18000 samples/s
    # trainpipes = [HybridTrainPipe(path_imgidx, path_imgrec, batch_size=batch_size, num_threads=6, device_id = i, num_gpus = N) for i in range(N)]
    # htp = trainpipes[0]
    # 单卡测试
    htp = HybridTrainPipe(path_imgrec, batch_size, 6, device_id = 0, num_gpus = N, initial_fill = batch_size)
    trainpipes = [htp]

    htp.build()
    print("Training pipeline epoch size: {}".format(htp.epoch_size("Reader")))
    dali_train_iter = DALIClassificationIterator(trainpipes, htp.epoch_size("Reader"))
    print([dali_train_iter.provide_data[0][:2]], [dali_train_iter.provide_label[0][:2]])
    import time
    time_start = time.time()
    batch_num = 0
    while True:
        batch = dali_train_iter.next()
        batch_num += 1
        # # print("batch num:", len(batch))
        # # # print("batch:", batch[0].asnumpy())
        # # print("elem num:", len(batch[0].data))
        # # print("image num:", batch[0].data[0].shape)
        # # print("label num:", batch[0].label[0].shape)
        # 查看图像结果
        # for image, label in zip(batch[0].data[0], batch[0].label[0]):
        #     # image = elem.data[0][0]
        #     # label = elem.data[0][1]
        #     # print(image)
        #     print(image.shape)
        #     print(label.asnumpy)
        #     cv2.imshow("image", image.asnumpy())
        #     cv2.waitKey(0)

        time_now = time.time()
        if time_now - time_start > 1 and batch_num > 0:
            print("\r{:.2f} samples/s".format(batch_num*batch_size/(time_now - time_start)), end='')
            batch_num = 0
            time_start = time_now

