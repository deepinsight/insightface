# coding=utf-8
"""Helper for evaluation on the Labeled Faces in the Wild dataset
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import math
import os
import random

import cv2
import mxnet as mx
import numpy as np
import sklearn
from mxnet import ndarray as nd
from scipy import interpolate
from sklearn.model_selection import KFold


class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    dist = distance(embeddings1, embeddings2)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # print('train_set', train_set)
        # print('test_set', test_set)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        print("fold_idx %s threshold %s acc_train %s" % (
            fold_idx, thresholds[best_threshold_index], acc_train[best_threshold_index]))
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = compare(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    print("calculate_val")
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    dist = distance(embeddings1, embeddings2)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            # Test if cuda could be foun
            try:
                threshold = f(far_target)
            except:
                threshold = 0.0
        else:
            threshold = 0.0
        print("fold_idx %s threshold %s np.max(far_train) %s" % (
            fold_idx, threshold, np.max(far_train)))
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, similarity, actual_issame):
    predict_issame = compare(similarity, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print(true_accept, false_accept)
    # print(n_same, n_diff)
    val = 0 if n_same == 0 else float(true_accept) / float(n_same)
    far = 0 if n_diff == 0 else float(false_accept) / float(n_diff)
    return val, far


def calculate_val_all(thresholds, embeddings1, embeddings2, actual_issame):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_thresholds = len(thresholds)

    dist = distance(embeddings1, embeddings2)
    # Find the threshold that gives FAR = far_target
    val_train = np.zeros(nrof_thresholds)
    far_train = np.zeros(nrof_thresholds)
    for threshold_idx, threshold in enumerate(thresholds):
        val_train[threshold_idx], far_train[threshold_idx] = calculate_val_far(threshold, dist, actual_issame)
    return val_train, far_train


USE_SIM = True


def compare(similarity, threshold):
    if USE_SIM:
        predict_issame = np.greater(similarity, threshold)
    else:
        predict_issame = np.less(similarity, threshold)
    return predict_issame


def distance(embeddings1, embeddings2):
    dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
    norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
    similarity = dot / norm
    similarity_normal = similarity * 0.5 + 0.5
    # print("similarity %s", similarity)
    dist = np.arccos(similarity) / math.pi
    if USE_SIM:
        return similarity_normal
    else:
        return dist


def test_badcase(data_set, mx_model, batch_size, name='', data_extra=None, label_shape=None):
    print('testing verification badcase..')
    data_list = data_set[0]
    issame_list = data_set[1]
    model = mx_model
    embeddings_list = []
    if data_extra is not None:
        _data_extra = nd.array(data_extra)
    time_consumed = 0.0
    if label_shape is None:
        _label = nd.ones((batch_size,))
    else:
        _label = nd.ones(label_shape)
    for i in xrange(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
            # print(_data.shape, _label.shape)
            time0 = datetime.datetime.now()
            if data_extra is None:
                db = mx.io.DataBatch(data=(_data,), label=(_label,))
            else:
                db = mx.io.DataBatch(data=(_data, _data_extra), label=(_label,))
            model.forward(db, is_train=False)
            net_out = model.get_outputs()
            _embeddings = net_out[0].asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    if USE_SIM:
        thresholds = np.arange(0, 1, 0.01)
    else:
        thresholds = np.arange(0, 4, 0.01)
    actual_issame = np.asarray(issame_list)
    nrof_folds = 10
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    dist = distance(embeddings1, embeddings2)
    data = data_list[0]

    pouts = []
    pouts_true = []
    nouts = []
    nouts_true = []

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        # print(train_set)
        # print(train_set.__class__)
        for threshold_idx, threshold in enumerate(thresholds):
            p2 = dist[train_set]
            p3 = actual_issame[train_set]
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, p2, p3)
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])
        best_threshold = thresholds[best_threshold_index]
        for iid in test_set:
            ida = iid * 2
            idb = ida + 1
            asame = actual_issame[iid]
            _dist = dist[iid]
            if USE_SIM:
                violate = _dist - best_threshold
            else:
                violate = best_threshold - _dist

            if not asame:
                violate *= -1.0
            if (USE_SIM and violate < 0.0) or (not USE_SIM and violate > 0.0):
                imga = data[ida].asnumpy().transpose((1, 2, 0))[..., ::-1]  # to bgr
                imgb = data[idb].asnumpy().transpose((1, 2, 0))[..., ::-1]
                print(imga.shape, imgb.shape, violate, asame, _dist)
                if asame:
                    pouts.append((imga, imgb, _dist, best_threshold, ida))
                else:
                    nouts.append((imga, imgb, _dist, best_threshold, ida))
            else:
                if random.random() < 0.05:
                    # 随机保存6000*0.05 = 300条数据查验
                    imga = data[ida].asnumpy().transpose((1, 2, 0))[..., ::-1]  # to bgr
                    imgb = data[idb].asnumpy().transpose((1, 2, 0))[..., ::-1]
                    if asame:
                        pouts_true.append((imga, imgb, _dist, best_threshold, ida))
                    else:
                        nouts_true.append((imga, imgb, _dist, best_threshold, ida))

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    acc = np.mean(accuracy)
    pouts = sorted(pouts, key=lambda x: x[2], reverse=True)
    nouts = sorted(nouts, key=lambda x: x[2], reverse=False)
    print(len(pouts), len(nouts))
    print('acc', acc)
    gap = 10
    image_shape = (112, 224, 3)
    out_dir = "./badcases"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if len(nouts) > 0:
        threshold = nouts[0][3]
    else:
        threshold = pouts[-1][3]

    for item in [(pouts, 'positive(false_negative).png'), (nouts, 'negative(false_positive).png'), (pouts_true, 'positive(true_positive).png'),
                 (nouts_true, 'negative(true_negative).png')]:
        cols = 4
        rows = 8000
        outs = item[0]
        if len(outs) == 0:
            continue
        # if len(outs)==9:
        #  cols = 3
        #  rows = 3

        _rows = int(math.ceil(len(outs) / cols))
        rows = min(rows, _rows)
        hack = {}

        # if name.startswith('cfp') and item[1].startswith('pos'):
        #     hack = {0: 'manual/238_13.jpg.jpg', 6: 'manual/088_14.jpg.jpg', 10: 'manual/470_14.jpg.jpg',
        #             25: 'manual/238_13.jpg.jpg', 28: 'manual/143_11.jpg.jpg'}

        filename = item[1]
        if len(name) > 0:
            filename = name + "_" + filename
        filename = os.path.join(out_dir, filename)
        img = np.zeros((image_shape[0] * rows + 20, image_shape[1] * cols + (cols - 1) * gap, 3), dtype=np.uint8)
        img[:, :, :] = 255
        text_color = (0, 0, 153)
        text_color = (255, 178, 102)
        text_color = (153, 255, 51)
        for outi, out in enumerate(outs):
            row = outi // cols
            col = outi % cols
            if row == rows:
                break
            imga = out[0].copy()
            imgb = out[1].copy()
            if outi in hack:
                idx = out[4]
                print('noise idx', idx)
                aa = hack[outi]
                imgb = cv2.imread(aa)
                # if aa==1:
                #  imgb = cv2.transpose(imgb)
                #  imgb = cv2.flip(imgb, 1)
                # elif aa==3:
                #  imgb = cv2.transpose(imgb)
                #  imgb = cv2.flip(imgb, 0)
                # else:
                #  for ii in xrange(2):
                #    imgb = cv2.transpose(imgb)
                #    imgb = cv2.flip(imgb, 1)
            dist = out[2]
            _img = np.concatenate((imga, imgb), axis=1)
            k = "%.3f" % dist
            # print(k)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(_img, k, (80, image_shape[0] // 2 + 7), font, 0.6, text_color, 2)
            # _filename = filename+"_%d.png"%outi
            # cv2.imwrite(_filename, _img)
            img[row * image_shape[0]:(row + 1) * image_shape[0],
            (col * image_shape[1] + gap * col):((col + 1) * image_shape[1] + gap * col), :] = _img
        # threshold = outs[0][3]
        font = cv2.FONT_HERSHEY_SIMPLEX
        k = "threshold: %.3f" % threshold
        cv2.putText(img, k, (img.shape[1] // 2 - 70, img.shape[0] - 5), font, 0.6, text_color, 2)
        cv2.imwrite(filename, img)


# ---------------------------分割线---------区分新加方法----------------------

def calculate_global(threshold, dist, actual_issame):
    print("calculate_global threshold %s" % (threshold))
    predict_issame = compare(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    print("tp %s fp %s tn %s fn %s" % (tp, fp, tn, fn))
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size

    far = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    frr = 0 if (tp + fn == 0) else float(fn) / float(tp + fn)

    return acc, tpr, fpr, far, frr


def cal_global_mode2(data_set, mx_model, threshold, batch_size):
    print('cal_global_mode2')
    data_list = data_set[0]
    issame_list = data_set[1]

    model = mx_model
    embeddings_list = []

    time_consumed = 0.0
    _label = nd.ones((batch_size,))

    for i in xrange(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
            # print(_data.shape, _label.shape)
            time0 = datetime.datetime.now()
            db = mx.io.DataBatch(data=(_data,), label=(_label,))
            model.forward(db, is_train=False)
            net_out = model.get_outputs()
            _embeddings = net_out[0].asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)

    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    dist = distance(embeddings1, embeddings2)
    val, tpr, fpr, far, frr = calculate_global(threshold, dist, issame_list)
    return val, tpr, fpr, far, frr
