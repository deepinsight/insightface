#!/usr/bin/env python2
import os
import random
import sys
from itertools import izip

g_SETS_NUMBER = 10
g_NUMBER_FOR_SET = 300
g_same_level = 4

lfw_path = "/home/lijc08/datasets/lfw/lfw_mtcnnpy_160"
if __name__ == "__main__":

    ftxt = open("./pairs.txt", "w")
    ftxt.write("{}\t{}\n".format(g_SETS_NUMBER, g_NUMBER_FOR_SET))

    pair_files = {}
    for p, ds, fs in os.walk(lfw_path):
        if p == lfw_path:
            continue
        for f in fs:
            pair_files.setdefault(os.path.basename(p), []).append(f)

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
            firstIndex = random.choice(pair_files[first]).rsplit("_", 1)[1].split('.')[0]
            second = names.pop(random.randint(0, len(names) - 1))
            secondIndex = random.choice(pair_files[second]).rsplit("_", 1)[1].split('.')[0]
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
                ftxt.write("%s\t%d\t%d\n" % (first, int(lft_f.rsplit("_", 1)[1].split('.')[0]), int(rgt_f.rsplit("_", 1)[1].split('.')[0])))
    ftxt.close()
