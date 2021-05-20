from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import mxnet as mx


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

    #
    imgid = 0
    for ds_id in range(len(rec_list)):
        imgrec = rec_list[ds_id]
        s = imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        assert header.flag > 0
        seq_identity = range(int(header.label[0]), int(header.label[1]))

        for identity in seq_identity:
            s = imgrec.read_idx(identity)
            header, _ = mx.recordio.unpack(s)
            for _idx in range(int(header.label[0]), int(header.label[1])):
                s = imgrec.read_idx(_idx)
                _header, _img = mx.recordio.unpack(s)
                label = int(_header.label[0])
                class_path = os.path.join(args.output, "id_%d" % label)
                if not os.path.exists(class_path):
                    os.makedirs(class_path)

                image_path = os.path.join(class_path, "%d_%d.jpg" % (label, imgid))
                with open(image_path, "wb") as ff:
                    ff.write(_img)

                imgid += 1
                if imgid % 10000 == 0:
                    print(imgid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do dataset merge')
    # general
    parser.add_argument('--include', default='', type=str, help='')
    parser.add_argument('--output', default='', type=str, help='')
    args = parser.parse_args()
    main(args)
