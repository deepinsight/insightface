from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from datetime import datetime
import os.path
from easydict import EasyDict as edict
import time
import json
import glob
import sys
import numpy as np
import importlib
import itertools
import argparse
import struct
import cv2
import sklearn
from sklearn.preprocessing import normalize
import mxnet as mx
from mxnet import ndarray as nd

image_shape = None
net = None
data_size = 203848
emb_size = 0
use_flip = False
ctx_num = 0


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_feature(buffer):
    global emb_size
    input_count = len(buffer)
    if use_flip:
        input_count *= 2
    network_count = input_count
    if input_count % ctx_num != 0:
        network_count = (input_count // ctx_num + 1) * ctx_num

    input_blob = np.zeros((network_count, 3, image_shape[1], image_shape[2]),
                          dtype=np.float32)
    idx = 0
    for item in buffer:
        img = cv2.imread(item)[:, :, ::-1]  #to rgb
        img = np.transpose(img, (2, 0, 1))
        attempts = [0, 1] if use_flip else [0]
        for flipid in attempts:
            _img = np.copy(img)
            if flipid == 1:
                do_flip(_img)
            input_blob[idx] = _img
            idx += 1
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data, ))
    net.model.forward(db, is_train=False)
    _embedding = net.model.get_outputs()[0].asnumpy()
    _embedding = _embedding[0:input_count]
    if emb_size == 0:
        emb_size = _embedding.shape[1]
        print('set emb_size to ', emb_size)
    embedding = np.zeros((len(buffer), emb_size), dtype=np.float32)
    if use_flip:
        embedding1 = _embedding[0::2]
        embedding2 = _embedding[1::2]
        embedding = embedding1 + embedding2
    else:
        embedding = _embedding
    embedding = sklearn.preprocessing.normalize(embedding)
    return embedding


def write_bin(path, m):
    rows, cols = m.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', rows, cols, cols * 4, 5))
        f.write(m.data)


def main(args):
    global image_shape
    global net
    global ctx_num

    print(args)
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd) > 0:
        for i in range(len(cvd.split(','))):
            ctx.append(mx.gpu(i))
    if len(ctx) == 0:
        ctx = [mx.cpu()]
        print('use cpu')
    else:
        print('gpu num:', len(ctx))
    ctx_num = len(ctx)
    image_shape = [int(x) for x in args.image_size.split(',')]
    vec = args.model.split(',')
    assert len(vec) > 1
    prefix = vec[0]
    epoch = int(vec[1])
    print('loading', prefix, epoch)
    net = edict()
    net.ctx = ctx
    net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(
        prefix, epoch)
    #net.arg_params, net.aux_params = ch_dev(net.arg_params, net.aux_params, net.ctx)
    all_layers = net.sym.get_internals()
    net.sym = all_layers['fc1_output']
    net.model = mx.mod.Module(symbol=net.sym,
                              context=net.ctx,
                              label_names=None)
    net.model.bind(data_shapes=[('data', (args.batch_size, 3, image_shape[1],
                                          image_shape[2]))])
    net.model.set_params(net.arg_params, net.aux_params)

    features_all = None

    i = 0
    filelist = os.path.join(args.input, 'filelist.txt')
    #print(filelist)
    buffer_images = []
    buffer_embedding = np.zeros((0, 0), dtype=np.float32)
    aggr_nums = []
    row_idx = 0
    for line in open(filelist, 'r'):
        if i % 1000 == 0:
            print("processing ", i)
        i += 1
        #print('stat', i, len(buffer_images), buffer_embedding.shape, aggr_nums, row_idx)
        videoname = line.strip().split()[0]
        images = glob.glob("%s/%s/*.jpg" % (args.input, videoname))
        assert len(images) > 0
        image_features = []
        for image_path in images:
            buffer_images.append(image_path)
        aggr_nums.append(len(images))
        while len(buffer_images) >= args.batch_size:
            embedding = get_feature(buffer_images[0:args.batch_size])
            buffer_images = buffer_images[args.batch_size:]
            if buffer_embedding.shape[1] == 0:
                buffer_embedding = embedding.copy()
            else:
                buffer_embedding = np.concatenate(
                    (buffer_embedding, embedding), axis=0)
        buffer_idx = 0
        acount = 0
        for anum in aggr_nums:
            if buffer_embedding.shape[0] >= anum + buffer_idx:
                image_features = buffer_embedding[buffer_idx:buffer_idx + anum]
                video_feature = np.sum(image_features, axis=0, keepdims=True)
                video_feature = sklearn.preprocessing.normalize(video_feature)
                if features_all is None:
                    features_all = np.zeros(
                        (data_size, video_feature.shape[1]), dtype=np.float32)
                #print('write to', row_idx, anum, buffer_embedding.shape)
                features_all[row_idx] = video_feature.flatten()
                row_idx += 1
                buffer_idx += anum
                acount += 1
            else:
                break
        aggr_nums = aggr_nums[acount:]
        buffer_embedding = buffer_embedding[buffer_idx:]

    if len(buffer_images) > 0:
        embedding = get_feature(buffer_images)
        buffer_images = buffer_images[args.batch_size:]
        buffer_embedding = np.concatenate((buffer_embedding, embedding),
                                          axis=0)
    buffer_idx = 0
    acount = 0
    for anum in aggr_nums:
        assert buffer_embedding.shape[0] >= anum + buffer_idx
        image_features = buffer_embedding[buffer_idx:buffer_idx + anum]
        video_feature = np.sum(image_features, axis=0, keepdims=True)
        video_feature = sklearn.preprocessing.normalize(video_feature)
        #print('last write to', row_idx, anum, buffer_embedding.shape)
        features_all[row_idx] = video_feature.flatten()
        row_idx += 1
        buffer_idx += anum
        acount += 1

    aggr_nums = aggr_nums[acount:]
    buffer_embedding = buffer_embedding[buffer_idx:]
    assert len(aggr_nums) == 0
    assert buffer_embedding.shape[0] == 0

    write_bin(args.output, features_all)
    print(row_idx, features_all.shape)
    #os.system("bypy upload %s"%args.output)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, help='', default=32)
    parser.add_argument('--image_size', type=str, help='', default='3,112,112')
    parser.add_argument('--input',
                        type=str,
                        help='',
                        default='./testdata-video')
    parser.add_argument('--output', type=str, help='', default='')
    parser.add_argument('--model', type=str, help='', default='')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
