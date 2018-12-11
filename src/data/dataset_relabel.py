from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import mxnet as mx

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))


def main(args):
    include_datasets = args.include.split(',')
    rec_list = []
    for ds in include_datasets:
        path_imgrec = os.path.join(ds, 'train.rec')
        path_imgidx = os.path.join(ds, 'train.idx')
        imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
        rec_list.append(imgrec)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    writer = mx.recordio.MXIndexedRecordIO(os.path.join(args.output, 'train.idx'), os.path.join(args.output, 'train.rec'), 'w')
    for ds_id in xrange(len(rec_list)):
        id_list = []
        imgrec = rec_list[ds_id]
        s = imgrec.read_idx(0)
        writer.write_idx(0, s)
        header, _ = mx.recordio.unpack(s)
        assert header.flag > 0
        print('header0 label', header.label)
        header0 = (int(header.label[0]), int(header.label[1]))
        seq_identity = range(int(header.label[0]), int(header.label[1]))
        pp = 0
        nlabel = -1
        for identity in seq_identity:
            pp += 1
            if pp % 10 == 0:
                print('processing id', pp)
            s = imgrec.read_idx(identity)
            writer.write_idx(identity, s)
            header, _ = mx.recordio.unpack(s)
            nlabel += 1
            for _idx in xrange(int(header.label[0]), int(header.label[1])):
                s = imgrec.read_idx(_idx)
                _header, _content = mx.recordio.unpack(s)
                nheader = mx.recordio.IRHeader(0, nlabel, _idx, 0)
                s = mx.recordio.pack(nheader, _content)
                writer.write_idx(_idx, s)

        print('max label', nlabel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do dataset merge')
    # general
    parser.add_argument('--include', default='', type=str, help='')
    parser.add_argument('--output', default='', type=str, help='')
    args = parser.parse_args()
    main(args)
