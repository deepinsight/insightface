from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import mxnet as mx

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))


def main(args):
    path_imgrec = os.path.join(args.input, 'train.rec')
    path_imgidx = os.path.join(args.input, 'train.idx')
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    assert header.flag > 0
    print('header0 label', header.label)
    header0 = (int(header.label[0]), int(header.label[1]))
    print('identities', header0[1] - header0[0])
    print('images', header0[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # general
    parser.add_argument('--input', default='/home/lijc08/datasets/glintasia/faces_glintasia', type=str, help='')
    args = parser.parse_args()
    main(args)
