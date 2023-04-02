import argparse
import multiprocessing
import os
import time

import mxnet as mx
import numpy as np


def read_worker(args, q_in):
    path_imgidx = os.path.join(args.input, "train.idx")
    path_imgrec = os.path.join(args.input, "train.rec")
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")

    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    assert header.flag > 0

    imgidx = np.array(range(1, int(header.label[0])))
    np.random.shuffle(imgidx)
    
    for idx in imgidx:
        item = imgrec.read_idx(idx)
        q_in.put(item)

    q_in.put(None)
    imgrec.close()


def write_worker(args, q_out):
    pre_time = time.time()
    
    if args.input[-1] == '/':
        args.input = args.input[:-1]
    dirname = os.path.dirname(args.input)
    basename = os.path.basename(args.input)
    output = os.path.join(dirname, f"shuffled_{basename}")
    os.makedirs(output, exist_ok=True)
    
    path_imgidx = os.path.join(output, "train.idx")
    path_imgrec = os.path.join(output, "train.rec")
    save_record = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "w")
    more = True
    count = 0
    while more:
        deq = q_out.get()
        if deq is None:
            more = False
        else:
            header, jpeg = mx.recordio.unpack(deq)
            # TODO it is currently not fully developed
            if isinstance(header.label, float):
                label = header.label
            else:
                label = header.label[0]

            header = mx.recordio.IRHeader(flag=header.flag, label=label, id=header.id, id2=header.id2)
            save_record.write_idx(count, mx.recordio.pack(header, jpeg))
            count += 1
            if count % 10000 == 0:
                cur_time = time.time()
                print('save time:', cur_time - pre_time, ' count:', count)
                pre_time = cur_time
    print(count)
    save_record.close()


def main(args):
    queue = multiprocessing.Queue(10240)
    read_process = multiprocessing.Process(target=read_worker, args=(args, queue))
    read_process.daemon = True
    read_process.start()
    write_process = multiprocessing.Process(target=write_worker, args=(args, queue))
    write_process.start()
    write_process.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='path to source rec.')
    main(parser.parse_args())
