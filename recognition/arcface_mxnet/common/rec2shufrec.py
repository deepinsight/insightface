import os
import os.path as osp
import sys
import datetime
import glob
import shutil
import numbers
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
import random
import argparse
import cv2
import time
import numpy as np

def main(args):
    ds = args.input
    path_imgrec = osp.join(ds, 'train.rec')
    path_imgidx = osp.join(ds, 'train.idx')
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
    if not osp.exists(args.output):
        os.makedirs(args.output)
    writer = mx.recordio.MXRecordIO(osp.join(args.output, 'train.rec'), 'w')
    s = imgrec.read_idx(0)
    header, _ = recordio.unpack(s)
    if header.flag > 0:
        print('header0 label', header.label)
        header0 = (int(header.label[0]), int(header.label[1]))
        imgidx = list(range(1, int(header.label[0])))
    else:
        imgidx = list(imgrec.keys)
    random.shuffle(imgidx)
    label_stat = None
    print('total images:', len(imgidx))
    for i, idx in enumerate(imgidx):
        if i%10000==0:
            print('processing', i, idx)
        s = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        if label_stat is None:
            label_stat = [label, label]
        else:
            label_stat[0] = min(label, label_stat[0])
            label_stat[1] = max(label, label_stat[1])
        wheader = mx.recordio.IRHeader(0, label, i, 0)
        ws = mx.recordio.pack(wheader, img)
        writer.write(ws)
    print('label_stat:', label_stat)
    writer.close()
    if args.copy_vers:
        for binfile in glob.glob(osp.join(args.input, '*.bin')):
            target_file = osp.join(args.output, binfile.split('/')[-1])
            shutil.copyfile(binfile, target_file)
    with open(osp.join(args.output, 'property'), 'w') as f:
        f.write("%d,112,112\n"%(int(label_stat[1])+1))
        f.write("%d\n"%len(imgidx))
        f.write("shuffled\n")
        f.write("%s\n"%(datetime.datetime.now()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert rec to shuffled rec')
    # general
    parser.add_argument('--input', default='', type=str, help='')
    parser.add_argument('--output', default='', type=str, help='')
    parser.add_argument('--copy-vers', action='store_true', help='copy verification bins')
    args = parser.parse_args()
    main(args)
