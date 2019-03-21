
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys

import mxnet as mx
import random
import argparse
import cv2
import time
import traceback
import numpy as np
#from builtins import range
from easydict import EasyDict as edict
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_preprocess
import face_image

try:
    import multiprocessing
except ImportError:
    multiprocessing = None



def read_label(path_in):
  identities = []
  last = [-1, -1]
  _id = 1
  dir2label = {}
  for line in open(path_in, 'r'):
    line = line.strip().split()
    item = edict()
    item.flag = 0
    item.image_path = os.path.join(args.input, 'images', line[0])
    image_dir = line[0].split('/')[0]
    if image_dir in dir2label:
      label = dir2label[image_dir]
    else:
      label = len(dir2label)
      dir2label[image_dir] = label
    item.bbox = np.array( [float(x) for x in line[1:5]], dtype=np.float32 )
    item.landmark = np.array( [float(x) for x in line[5:15]], dtype=np.float32 ).reshape( (5,2) )
    item.aligned = False
    item.id = _id
    item.label = label
    yield item
    if label!=last[0]:
      if last[1]>=0:
        identities.append( (last[1], _id) )
      last[0] = label
      last[1] = _id
    _id+=1
  identities.append( (last[1], _id) )
  item = edict()
  item.flag = 2
  item.id = 0
  item.label = [float(_id), float(_id+len(identities))]
  yield item
  for identity in identities:
    item = edict()
    item.flag = 2
    item.id = _id
    _id+=1
    item.label = [float(identity[0]), float(identity[1])]
    yield item



def image_encode(args, i, item, q_out):
    #print('flag', item.flag)
    if item.flag==0:
      oitem = [item.id, item.label]
      fullpath = item.image_path
      header = mx.recordio.IRHeader(item.flag, item.label, item.id, 0)
      #print('write', item.flag, item.id, item.label)
      if item.aligned:
        with open(fullpath, 'rb') as fin:
            img = fin.read()
        s = mx.recordio.pack(header, img)
        q_out.put((i, s, oitem))
      else:
        img = cv2.imread(fullpath, args.color)
        assert item.landmark is not None
        img = face_preprocess.preprocess(img, bbox = item.bbox, landmark=item.landmark, image_size='%d,%d'%(args.image_h, args.image_w))
        s = mx.recordio.pack_img(header, img, quality=args.quality, img_fmt=args.encoding)
        q_out.put((i, s, oitem))
    else: 
      oitem = [item.id, -1]
      header = mx.recordio.IRHeader(item.flag, item.label, item.id, 0)
      #print('write', item.flag, item.id, item.label)
      s = mx.recordio.pack(header, '')
      q_out.put((i, s, oitem))


def read_worker(args, q_in, q_out):
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        image_encode(args, i, item, q_out)

def write_worker(q_out, output_dir):
    pre_time = time.time()
    count = 0
    record = mx.recordio.MXIndexedRecordIO(os.path.join(output_dir, 'train.idx'),
                                           os.path.join(output_dir, 'train.rec'), 'w')
    buf = {}
    more = True
    max_label = 0
    while more:
        deq = q_out.get()
        if deq is not None:
            i, s, item = deq
            buf[i] = (s, item)
        else:
            more = False
        while count in buf:
            s, item = buf[count]
            label = item[1]
            #print('label', label)
            max_label = max(max_label, label)
            del buf[count]
            if s is not None:
                #print('write idx', item[0])
                record.write_idx(item[0], s)

            if count % 1000 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', count)
                pre_time = cur_time
            count += 1
    print('writing', output_dir, 'property', max_label)
    with open(os.path.join(output_dir, 'property'), 'w') as outf:
      outf.write("%d,112,112"%(max_label+1))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    parser.add_argument('--input', help='input data dir.')
    parser.add_argument('--output', help='outputdata dir.')
    #parser.add_argument('root', help='path to folder containing images.')

    cgroup = parser.add_argument_group('Options for creating image lists')
    cgroup.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg'],
                        help='list of acceptable image extensions.')
    cgroup.add_argument('--chunks', type=int, default=1, help='number of chunks.')
    cgroup.add_argument('--train-ratio', type=float, default=1.0,
                        help='Ratio of images to use for training.')
    cgroup.add_argument('--test-ratio', type=float, default=0,
                        help='Ratio of images to use for testing.')

    rgroup = parser.add_argument_group('Options for creating database')
    rgroup.add_argument('--quality', type=int, default=95,
                        help='JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9')
    rgroup.add_argument('--num-thread', type=int, default=1,
                        help='number of thread to use for encoding. order of images will be different\
        from the input list if >1. the input list will be modified to match the\
        resulting order.')
    rgroup.add_argument('--color', type=int, default=1, choices=[-1, 0, 1],
                        help='specify the color mode of the loaded image.\
        1: Loads a color image. Any transparency of image will be neglected. It is the default flag.\
        0: Loads image in grayscale mode.\
        -1:Loads image as such including alpha channel.')
    rgroup.add_argument('--encoding', type=str, default='.jpg', choices=['.jpg', '.png'],
                        help='specify the encoding of the images.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    label_file = os.path.join(args.input, 'label.txt')
    assert os.path.exists(label_file)
    if not os.path.exists(args.output):
      os.makedirs(args.output)
    image_size = [112, 112]
    print('image_size', image_size)
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    image_list = read_label(label_file)
    # -- write_record -- #
    q_in = [multiprocessing.Queue(1024) for i in range(args.num_thread)]
    q_out = multiprocessing.Queue(1024)
    read_process = [multiprocessing.Process(target=read_worker, args=(args, q_in[i], q_out)) \
                    for i in range(args.num_thread)]
    for p in read_process:
        p.start()
    write_process = multiprocessing.Process(target=write_worker, args=(q_out, args.output))
    write_process.start()

    for i, item in enumerate(image_list):
        q_in[i % len(q_in)].put((i, item))
    for q in q_in:
        q.put(None)
    for p in read_process:
        p.join()

    q_out.put(None)
    write_process.join()

