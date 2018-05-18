from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import mxnet as mx
from mxnet import ndarray as nd
import random
import argparse
import cv2
import time
import sklearn
import numpy as np



def main(args):
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    train_writer = mx.recordio.MXIndexedRecordIO(os.path.join(args.output, 'train.idx'), os.path.join(args.output, 'train.rec'), 'w')
    val_writer = mx.recordio.MXIndexedRecordIO(os.path.join(args.output, 'val.idx'), os.path.join(args.output, 'val.rec'), 'w')
    train_widx = [0]
    val_widx = [0]
    stat = [0,0]
    #for ds in ['ms1m', 'megaage', 'imdb']:
    for ds in ['megaage', 'imdb']:
        for n in ['train', 'val']:
            #if ds=='ms1m' or ds=='imdb':
            #  continue
            repeat = 1
            if args.mode=='age':
              if args.lite:
                if ds!='megaage':
                  continue
              if n=='val' and ds!='megaage':
                continue
              if n=='train' and ds=='megaage':
                if args.lite==0:
                  repeat = 10
              if n=='train' and ds=='imdb':
                repeat = 1
            elif args.mode=='gender':
              if ds!='imdb':
                continue
            else:
              if n=='train' and ds=='megaage':
                repeat = 10
            writer = train_writer
            widx = train_widx
            if n=='val':
                writer = val_writer
                widx = val_widx
            path = os.path.join(args.input, ds, '%s.rec'%n)
            if not os.path.exists(path):
                continue
            imgrec = mx.recordio.MXIndexedRecordIO(path[:-3]+'idx', path, 'r')  # pylint: disable=redefined-variable-type
            if ds=='ms1m':
                s = imgrec.read_idx(0)
                header, _ = mx.recordio.unpack(s)
                assert header.flag>0
                print('header0 label', header.label)
                header0 = (int(header.label[0]), int(header.label[1]))
                #assert(header.flag==1)
                imgidx = range(1, int(header.label[0]))
            else:
                imgidx = list(imgrec.keys)
            for idx in imgidx:
                if ds=='megaage' and idx==0:
                  continue
                print('info', ds, n, idx)
                s = imgrec.read_idx(idx)
                _header, _content = mx.recordio.unpack(s)
                stat[0]+=1
                try:
                    img = mx.image.imdecode(_content)
                except:
                    stat[1]+=1
                    print('error', ds, n, idx)
                    continue
                #print(img.shape)
                if ds=='ms1m':
                    nlabel = [_header.label]
                    nlabel += [-1]*101
                elif ds=='megaage':
                    #nlabel = [-1, -1]
                    nlabel = []
                    age_label = [0]*100
                    age = int(_header.label[0])
                    if age>100 or age<0:
                        continue
                    age = max(0, min(100, age))
                    #print('age', age)

                    for a in xrange(0, age):
                        age_label[a] = 1
                    nlabel += age_label
                elif ds=='imdb':
                    gender = int(_header.label[1])
                    if args.mode=='gender':
                      nlabel = [gender]
                    else:
                      age_label = [0]*100
                      age = int(_header.label[0])
                      age = max(0, min(100, age))
                      for a in xrange(0, age):
                          age_label[a] = 1
                      nlabel = age_label
                    #nlabel += age_label
                for r in xrange(repeat):
                    nheader = mx.recordio.IRHeader(0, nlabel, widx[0], 0)
                    s = mx.recordio.pack(nheader, _content)
                    writer.write_idx(widx[0], s)
                    widx[0]+=1
    print('stat', stat)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='do dataset merge')
  # general
  parser.add_argument('--input', default='', type=str, help='')
  parser.add_argument('--output', default='', type=str, help='')
  parser.add_argument('--mode', default='age', type=str, help='')
  parser.add_argument('--lite', default=1, type=int, help='')
  args = parser.parse_args()
  main(args)

