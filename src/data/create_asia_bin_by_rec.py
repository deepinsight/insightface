#!/usr/bin/env python2
import os
import pickle
import random
import shutil

import mxnet as mx
from PIL import Image
from mxnet import recordio

g_SETS_NUMBER = 2
g_SETS_NUMBER = 10
g_NUMBER_FOR_SET = 100
g_NUMBER_FOR_SET = 300
g_same_level = 4

asia_path = "/home/lijc08/datasets/glintasia/faces_glintasia"


def img_and_label(imgrec, index):
    s = imgrec.read_idx(index)
    header, img = recordio.unpack(s)
    label = header.label[0]
    return img, label


def save_bin(bin, path_file):
    decodeImg = mx.image.imdecode(bin)
    img = Image.fromarray(decodeImg.asnumpy(), 'RGB')
    img.save(path_file, format='JPEG')


def save_bin(bin, path_file):
    decodeImg = mx.image.imdecode(bin)
    img = Image.fromarray(decodeImg.asnumpy(), 'RGB')
    img.save(path_file, format='JPEG')


def export(asia_bins, issame_list, skip=100):
    tmp_dir = "asia_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    for i in range(len(issame_list)):
        if i % skip != 0:
            continue
        issame = issame_list[i]
        child_dir = os.path.join(tmp_dir, "%s_%s" % (i, issame))
        os.mkdir(child_dir)

        bin1 = asia_bins[i * 2]
        label1 = labels[i * 2]
        save_bin(bin1, os.path.join(child_dir, "%s_%s.jpg" % (str(label1), str(i * 2))))
        bin2 = asia_bins[i * 2 + 1]
        label2 = labels[i * 2 + 1]
        save_bin(bin2, os.path.join(child_dir, "%s_%s.jpg" % (str(label2), str(i * 2 + 1))))


if __name__ == "__main__":

    asia_bins = []
    labels = []
    issame_list = []

    path_imgrec = os.path.join(asia_path, 'train.rec')
    path_imgidx = os.path.join(asia_path, 'train.idx')
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    assert header.flag > 0
    print('header0 label', header.label)
    header0 = (int(header.label[0]), int(header.label[1]))
    print('identities', header0[1] - header0[0])
    print('images', header0[0])

    # assert(header.flag==1)
    imgidx = range(1, int(header.label[0]))
    pair_files = {}
    seq_identity = range(int(header.label[0]), int(header.label[1]))
    for identity in seq_identity:
        s = imgrec.read_idx(identity)
        header, _ = recordio.unpack(s)
        a, b = int(header.label[0]), int(header.label[1])
        pair_files[identity] = range(a, b)

    random.seed(100)
    print('start random pairs')
    for i in range(g_SETS_NUMBER):
        print("g_SETS_NUMBER %s max %s" % (i, g_SETS_NUMBER))
        names = pair_files.keys()
        random.shuffle(names)
        idx = 0

        print("diff start ")
        while names:
            if idx >= g_NUMBER_FOR_SET:
                break
            idx += 1
            first = names.pop(random.randint(0, len(names) - 1))
            firstIndex = random.choice(pair_files[first])
            img, label = img_and_label(imgrec, firstIndex)
            asia_bins.append(img)
            labels.append(label)

            second = names.pop(random.randint(0, len(names) - 1))
            secondIndex = random.choice(pair_files[second])
            img, label = img_and_label(imgrec, secondIndex)
            asia_bins.append(img)
            labels.append(label)
            issame_list.append(False)

        print("same start ")
        idx = 0
        names = pair_files.keys()
        random.shuffle(names)
        while names:
            if idx >= g_NUMBER_FOR_SET:
                break
            first = names.pop(random.randint(0, len(names) - 1))
            fs = pair_files[first]
            if len(fs) <= 1:
                continue
            random.shuffle(fs)

            same_cnt = 0
            for lft_f, rgt_f in zip(*[iter(fs)] * 2):
                if idx >= g_NUMBER_FOR_SET:
                    break
                if same_cnt > 0:
                    ran = random.random()
                    if len(fs) > 15 and ran < 0.8:
                        continue
                    if len(fs) < 15 and ran < 0.7:
                        continue
                if same_cnt >= g_same_level:
                    break
                idx += 1
                same_cnt += 1

                img, label = img_and_label(imgrec, lft_f)
                asia_bins.append(img)
                labels.append(label)

                img, label = img_and_label(imgrec, rgt_f)
                asia_bins.append(img)
                labels.append(label)
                issame_list.append(True)

    print(len(asia_bins), len(issame_list))
    export(asia_bins, issame_list)

    with open("asia.bin", 'wb') as f:
        pickle.dump((asia_bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
