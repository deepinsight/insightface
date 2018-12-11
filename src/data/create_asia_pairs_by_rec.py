#!/usr/bin/env python2
import os
import random
import sys
from itertools import izip
import mxnet as mx
from mxnet import recordio

g_SETS_NUMBER = 10
g_NUMBER_FOR_SET = 300
g_same_level = 4

asia_path = "/home/lijc08/datasets/glintasia/faces_glintasia"
if __name__ == "__main__":

    ftxt = open("./asia_pairs.txt", "w")
    ftxt.write("{}\t{}\n".format(g_SETS_NUMBER, g_NUMBER_FOR_SET))

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

    print('start random pairs')
    for i in range(g_SETS_NUMBER):
        names = pair_files.keys()
        random.shuffle(names)
        idx = 0
        while names:
            if idx >= g_NUMBER_FOR_SET:
                break
            idx += 1
            first = names.pop(random.randint(0, len(names) - 1))
            firstIndex = random.choice(pair_files[first])
            second = names.pop(random.randint(0, len(names) - 1))
            secondIndex = random.choice(pair_files[second])
            ftxt.write("%s\t%d\t%s\t%d\n" % (first, int(firstIndex), second, int(secondIndex)))

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
                    if len(fs) > 15 and ran < 0.95:
                        continue
                    if len(fs) < 15 and ran < 0.7:
                        continue
                if same_cnt >= g_same_level:
                    break
                idx += 1
                same_cnt += 1
                ftxt.write("%s\t%d\t%d\n" % (first, int(lft_f), int(rgt_f)))
    ftxt.close()
