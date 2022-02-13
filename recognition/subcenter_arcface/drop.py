import os
import shutil
import datetime
import sys
from mxnet import ndarray as nd
import mxnet as mx
import random
import argparse
import numbers
import cv2
import time
import pickle
import sklearn
import sklearn.preprocessing
from easydict import EasyDict as edict
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from rec_builder import *


def get_embedding(args, imgrec, a, b, image_size, model):
    ocontents = []
    for idx in range(a, b):
        s = imgrec.read_idx(idx)
        ocontents.append(s)
    embeddings = None
    #print(len(ocontents))
    ba = 0
    rlabel = -1
    imgs = []
    contents = []
    while True:
        bb = min(ba + args.batch_size, len(ocontents))
        if ba >= bb:
            break
        _batch_size = bb - ba
        #_batch_size2 = max(_batch_size, args.ctx_num)
        _batch_size2 = _batch_size
        if _batch_size % args.ctx_num != 0:
            _batch_size2 = ((_batch_size // args.ctx_num) + 1) * args.ctx_num
        data = np.zeros((_batch_size2, 3, image_size[0], image_size[1]))
        count = bb - ba
        ii = 0
        for i in range(ba, bb):
            header, img = mx.recordio.unpack(ocontents[i])
            contents.append(img)
            label = header.label
            if not isinstance(label, numbers.Number):
                label = label[0]
            if rlabel < 0:
                rlabel = int(label)

            img = mx.image.imdecode(img)
            rgb = img.asnumpy()
            bgr = rgb[:, :, ::-1]
            imgs.append(bgr)
            img = rgb.transpose((2, 0, 1))
            data[ii] = img
            ii += 1
        while ii < _batch_size2:
            data[ii] = data[0]
            ii += 1
        nddata = nd.array(data)
        db = mx.io.DataBatch(data=(nddata, ))
        model.forward(db, is_train=False)
        net_out = model.get_outputs()
        net_out = net_out[0].asnumpy()
        if embeddings is None:
            embeddings = np.zeros((len(ocontents), net_out.shape[1]))
        embeddings[ba:bb, :] = net_out[0:_batch_size, :]
        ba = bb
    embeddings = sklearn.preprocessing.normalize(embeddings)
    return embeddings, rlabel, contents


def main(args):
    print(args)
    image_size = (112, 112)
    print('image_size', image_size)
    vec = args.model.split(',')
    prefix = vec[0]
    epoch = int(vec[1])
    print('loading', prefix, epoch)
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd) > 0:
        for i in range(len(cvd.split(','))):
            ctx.append(mx.gpu(i))
    if len(ctx) == 0:
        ctx = [mx.cpu()]
        print('use cpu')
    else:
        print('gpu num:', len(ctx))
    args.ctx_num = len(ctx)
    args.batch_size *= args.ctx_num
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    W = None
    i = 0
    while True:
        key = 'fc7_%d_weight' % i
        i += 1
        if key not in arg_params:
            break
        _W = arg_params[key].asnumpy()
        #_W = _W.reshape( (-1, 10, 512) )
        if W is None:
            W = _W
        else:
            W = np.concatenate((W, _W), axis=0)
    K = args.k
    W = sklearn.preprocessing.normalize(W)
    W = W.reshape((-1, K, 512))
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (args.ctx_num, 3, image_size[0],
                                      image_size[1]))])
    model.set_params(arg_params, aux_params)
    print('W:', W.shape)
    path_imgrec = os.path.join(args.data, 'train.rec')
    path_imgidx = os.path.join(args.data, 'train.idx')
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
    id_list = []
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    assert header.flag > 0
    print('header0 label', header.label)
    header0 = (int(header.label[0]), int(header.label[1]))
    #assert(header.flag==1)
    imgidx = range(1, int(header.label[0]))
    id2range = {}
    a, b = int(header.label[0]), int(header.label[1])
    seq_identity = range(a, b)
    print(len(seq_identity))
    image_count = 0
    pp = 0
    for wid, identity in enumerate(seq_identity):
        pp += 1
        s = imgrec.read_idx(identity)
        header, _ = mx.recordio.unpack(s)
        contents = []
        a, b = int(header.label[0]), int(header.label[1])
        _count = b - a
        id_list.append((wid, a, b, _count))
        image_count += _count
    pp = 0
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    ret = np.zeros((image_count, K + 1), dtype=np.float32)
    output_dir = args.output
    builder = SeqRecBuilder(output_dir)
    print(ret.shape)
    imid = 0
    da = datetime.datetime.now()
    label = 0
    num_images = 0
    cos_thresh = np.cos(np.pi * args.threshold / 180.0)
    for id_item in id_list:
        wid = id_item[0]
        pp += 1
        if pp % 40 == 0:
            db = datetime.datetime.now()
            print('processing id', pp, (db - da).total_seconds())
            da = db
        x, _, contents = get_embedding(args, imgrec, id_item[1], id_item[2],
                                       image_size, model)
        subcenters = W[wid]
        K_stat = np.zeros((K, ), dtype=np.int)
        for i in range(x.shape[0]):
            _x = x[i]
            sim = np.dot(subcenters, _x)  # len(sim)==K
            mc = np.argmax(sim)
            K_stat[mc] += 1
        dominant_index = np.argmax(K_stat)
        dominant_center = subcenters[dominant_index]
        sim = np.dot(x, dominant_center)
        idx = np.where(sim > cos_thresh)[0]
        num_drop = x.shape[0] - len(idx)
        if len(idx) == 0:
            continue
        #print("labelid %d dropped %d, from %d to %d"% (wid, num_drop, x.shape[0], len(idx)))
        num_images += len(idx)
        for _idx in idx:
            c = contents[_idx]
            builder.add(label, c, is_image=False)
        label += 1
    builder.close()

    print('total:', num_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # general
    parser.add_argument('--data',
                        default='/bigdata/faces_ms1m_full',
                        type=str,
                        help='')
    parser.add_argument('--output',
                        default='/bigdata/ms1m_full_k3drop075',
                        type=str,
                        help='')
    parser.add_argument(
        '--model',
        default=
        '../Evaluation/IJB/pretrained_models/r50-arcfacesc-msf-k3z/model,2',
        help='path to load model.')
    parser.add_argument('--batch-size', default=16, type=int, help='')
    parser.add_argument('--threshold', default=75, type=float, help='')
    parser.add_argument('--k', default=3, type=int, help='')
    args = parser.parse_args()
    main(args)
