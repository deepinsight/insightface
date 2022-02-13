from __future__ import print_function
import sys
import mxnet as mx
import numpy as np
import random
import datetime
import multiprocessing
import cv2
from mxnet.executor_manager import _split_input_slice

from rcnn.config import config
from rcnn.io.image import tensor_vstack
from rcnn.io.rpn import get_rpn_testbatch, get_rpn_batch, assign_anchor_fpn, get_crop_batch, AA


class CropLoader(mx.io.DataIter):
    def __init__(self,
                 feat_sym,
                 roidb,
                 batch_size=1,
                 shuffle=False,
                 ctx=None,
                 work_load_list=None,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: AnchorLoader
        """
        super(CropLoader, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        #self.feat_stride = feat_stride
        #self.anchor_scales = anchor_scales
        #self.anchor_ratios = anchor_ratios
        #self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping
        self.feat_stride = config.RPN_FEAT_STRIDE

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        #self.data_name = ['data']
        #self.label_name = []
        #self.label_name.append('label')
        #self.label_name.append('bbox_target')
        #self.label_name.append('bbox_weight')

        self.data_name = ['data']
        #self.label_name = ['label', 'bbox_target', 'bbox_weight']
        self.label_name = []
        prefixes = ['face']
        if config.HEAD_BOX:
            prefixes.append('head')
        names = []
        for prefix in prefixes:
            names += [
                prefix + '_label', prefix + '_bbox_target',
                prefix + '_bbox_weight'
            ]
            if prefix == 'face' and config.FACE_LANDMARK:
                names += [
                    prefix + '_landmark_target', prefix + '_landmark_weight'
                ]
        #names = ['label', 'bbox_weight']
        for stride in self.feat_stride:
            for n in names:
                k = "%s_stride%d" % (n, stride)
                self.label_name.append(k)
        if config.CASCADE > 0:
            self.label_name.append('gt_boxes')

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None
        # infer shape
        feat_shape_list = []
        _data_shape = [('data', (1, 3, max([v[1] for v in config.SCALES]),
                                 max([v[1] for v in config.SCALES])))]
        _data_shape = dict(_data_shape)
        for i in range(len(self.feat_stride)):
            _, feat_shape, _ = self.feat_sym[i].infer_shape(**_data_shape)
            feat_shape = [int(i) for i in feat_shape[0]]
            feat_shape_list.append(feat_shape)
        self.aa = AA(feat_shape_list)

        self._debug = False
        self._debug_id = 0
        self._times = [0.0, 0.0, 0.0, 0.0]

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data,
                                   label=self.label,
                                   pad=self.getpad(),
                                   index=self.getindex(),
                                   provide_data=self.provide_data,
                                   provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        dummy_boxes = np.zeros((0, 5))
        dummy_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]
        dummy_label = {'gt_boxes': dummy_boxes}
        dummy_blur = np.zeros((0, ))
        dummy_label['gt_blur'] = dummy_blur

        label_dict = {}
        if config.HEAD_BOX:
            head_label_dict = self.aa.assign_anchor_fpn(dummy_label,
                                                        dummy_info,
                                                        False,
                                                        prefix='head')
            label_dict.update(head_label_dict)

        if config.FACE_LANDMARK:
            dummy_landmarks = np.zeros((0, 5, 3))
            dummy_label['gt_landmarks'] = dummy_landmarks
        face_label_dict = self.aa.assign_anchor_fpn(dummy_label,
                                                    dummy_info,
                                                    config.FACE_LANDMARK,
                                                    prefix='face')
        label_dict.update(face_label_dict)
        if config.CASCADE > 0:
            label_dict['gt_boxes'] = np.zeros(
                (0, config.TRAIN.MAX_BBOX_PER_IMAGE, 5), dtype=np.float32)

        label_list = []
        for k in self.label_name:
            label_list.append(label_dict[k])
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:])))
                       for k, v in zip(self.label_name, label_list)]
        return max_data_shape, label_shape

    def get_batch(self):
        # slice roidb
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        assert cur_to == cur_from + self.batch_size
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        # get testing data for multigpu
        data_list = []
        label_list = []
        for islice in slices:
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            data, label = get_crop_batch(iroidb)
            data_list += data
            label_list += label
            #data_list.append(data)
            #label_list.append(label)

        # pad data first and then assign anchor (read label)
        #data_tensor = tensor_vstack([batch['data'] for batch in data_list])
        #for i_card in range(len(data_list)):
        #    data_list[i_card]['data'] = data_tensor[
        #                                i_card * config.TRAIN.BATCH_IMAGES:(1 + i_card) * config.TRAIN.BATCH_IMAGES]

        #iiddxx = 0
        select_stride = 0
        if config.RANDOM_FEAT_STRIDE:
            select_stride = random.choice(config.RPN_FEAT_STRIDE)

        for data, label in zip(data_list, label_list):
            data_shape = {k: v.shape for k, v in data.items()}
            del data_shape['im_info']
            feat_shape_list = []
            for s in range(len(self.feat_stride)):
                _, feat_shape, _ = self.feat_sym[s].infer_shape(**data_shape)
                feat_shape = [int(i) for i in feat_shape[0]]
                feat_shape_list.append(feat_shape)
            im_info = data['im_info']
            gt_boxes = label['gt_boxes']
            gt_label = {'gt_boxes': gt_boxes}
            if config.USE_BLUR:
                gt_blur = label['gt_blur']
                gt_label['gt_blur'] = gt_blur
            if self._debug:
                img = data['data'].copy()[0].transpose(
                    (1, 2, 0))[:, :, ::-1].copy()
                print('DEBUG SHAPE', data['data'].shape,
                      label['gt_boxes'].shape)

                box = label['gt_boxes'].copy()[0][0:4].astype(np.int)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
                              (0, 255, 0), 2)
                filename = './debugout/%d.png' % (self._debug_id)
                print('debug write', filename)
                cv2.imwrite(filename, img)
                self._debug_id += 1
                #print('DEBUG', img.shape, bbox.shape)
            label_dict = {}
            if config.HEAD_BOX:
                head_label_dict = self.aa.assign_anchor_fpn(
                    gt_label,
                    im_info,
                    False,
                    prefix='head',
                    select_stride=select_stride)
                label_dict.update(head_label_dict)
            if config.FACE_LANDMARK:
                gt_landmarks = label['gt_landmarks']
                gt_label['gt_landmarks'] = gt_landmarks
            #ta = datetime.datetime.now()
            #face_label_dict = assign_anchor_fpn(feat_shape_list, gt_label, im_info, config.FACE_LANDMARK, prefix='face', select_stride = select_stride)
            face_label_dict = self.aa.assign_anchor_fpn(
                gt_label,
                im_info,
                config.FACE_LANDMARK,
                prefix='face',
                select_stride=select_stride)
            #tb = datetime.datetime.now()
            #self._times[0] += (tb-ta).total_seconds()
            label_dict.update(face_label_dict)
            #for k in label_dict:
            #  print(k, label_dict[k].shape)

            if config.CASCADE > 0:
                pad_gt_boxes = np.empty(
                    (1, config.TRAIN.MAX_BBOX_PER_IMAGE, 5), dtype=np.float32)
                pad_gt_boxes.fill(-1)
                pad_gt_boxes[0, 0:gt_boxes.shape[0], :] = gt_boxes
                label_dict['gt_boxes'] = pad_gt_boxes
            #print('im_info', im_info.shape)
            #print(gt_boxes.shape)
            for k in self.label_name:
                label[k] = label_dict[k]

        all_data = dict()
        for key in self.data_name:
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in self.label_name:
            pad = 0 if key.startswith('bbox_') else -1
            #print('label vstack', key, pad, len(label_list), file=sys.stderr)
            all_label[key] = tensor_vstack(
                [batch[key] for batch in label_list], pad=pad)

        self.data = [mx.nd.array(all_data[key]) for key in self.data_name]
        self.label = [mx.nd.array(all_label[key]) for key in self.label_name]
        #for _label in self.label:
        #  print('LABEL SHAPE', _label.shape)
        #print(self._times)


class CropLoader2(mx.io.DataIter):
    def __init__(self,
                 feat_sym,
                 roidb,
                 batch_size=1,
                 shuffle=False,
                 ctx=None,
                 work_load_list=None,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: AnchorLoader
        """
        super(CropLoader2, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        #self.feat_stride = feat_stride
        #self.anchor_scales = anchor_scales
        #self.anchor_ratios = anchor_ratios
        #self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping
        self.feat_stride = config.RPN_FEAT_STRIDE

        # infer properties from roidb
        self.size = len(roidb)

        # decide data and label names
        #self.data_name = ['data']
        #self.label_name = []
        #self.label_name.append('label')
        #self.label_name.append('bbox_target')
        #self.label_name.append('bbox_weight')

        self.data_name = ['data']
        #self.label_name = ['label', 'bbox_target', 'bbox_weight']
        self.label_name = []
        prefixes = ['face']
        if config.HEAD_BOX:
            prefixes.append('head')
        names = []
        for prefix in prefixes:
            names += [
                prefix + '_label', prefix + '_bbox_target',
                prefix + '_bbox_weight'
            ]
            if prefix == 'face' and config.FACE_LANDMARK:
                names += [
                    prefix + '_landmark_target', prefix + '_landmark_weight'
                ]
        #names = ['label', 'bbox_weight']
        for stride in self.feat_stride:
            for n in names:
                k = "%s_stride%d" % (n, stride)
                self.label_name.append(k)
        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.q_in = [
            multiprocessing.Queue(1024) for i in range(config.NUM_CPU)
        ]
        #self.q_in = multiprocessing.Queue(1024)
        self.q_out = multiprocessing.Queue(1024)
        self.start()
        self.get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def reset(self):
        pass

    @staticmethod
    def input_worker(q_in, roidb, batch_size):
        index = np.arange(len(roidb))
        np.random.shuffle(index)
        cur_from = 0
        while True:
            cur_to = cur_from + batch_size
            if cur_to > len(roidb):
                np.random.shuffle(index)
                cur_from = 0
                continue
            _roidb = [roidb[index[i]] for i in range(cur_from, cur_to)]
            istart = index[cur_from]
            q_in[istart % len(q_in)].put(_roidb)
            cur_from = cur_to

    @staticmethod
    def gen_worker(q_in, q_out):
        while True:
            deq = q_in.get()
            if deq is None:
                break
            _roidb = deq
            data, label = get_crop_batch(_roidb)
            print('generated')
            q_out.put((data, label))

    def start(self):
        input_process = multiprocessing.Process(
            target=CropLoader2.input_worker,
            args=(self.q_in, self.roidb, self.batch_size))
        #gen_process = multiprocessing.Process(target=gen_worker, args=(q_in, q_out))
        gen_process = [multiprocessing.Process(target=CropLoader2.gen_worker, args=(self.q_in[i], self.q_out)) \
                  for i in range(config.NUM_CPU)]
        input_process.start()
        for p in gen_process:
            p.start()

    def next(self):
        self.get_batch()
        return mx.io.DataBatch(data=self.data,
                               label=self.label,
                               provide_data=self.provide_data,
                               provide_label=self.provide_label)

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        dummy_boxes = np.zeros((0, 5))
        dummy_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]
        dummy_label = {'gt_boxes': dummy_boxes}

        # infer shape
        feat_shape_list = []
        for i in range(len(self.feat_stride)):
            _, feat_shape, _ = self.feat_sym[i].infer_shape(**max_shapes)
            feat_shape = [int(i) for i in feat_shape[0]]
            feat_shape_list.append(feat_shape)

        label_dict = {}
        if config.HEAD_BOX:
            head_label_dict = assign_anchor_fpn(feat_shape_list,
                                                dummy_label,
                                                dummy_info,
                                                False,
                                                prefix='head')
            label_dict.update(head_label_dict)

        if config.FACE_LANDMARK:
            dummy_landmarks = np.zeros((0, 11))
            dummy_label['gt_landmarks'] = dummy_landmarks
        face_label_dict = assign_anchor_fpn(feat_shape_list,
                                            dummy_label,
                                            dummy_info,
                                            config.FACE_LANDMARK,
                                            prefix='face')
        label_dict.update(face_label_dict)

        label_list = []
        for k in self.label_name:
            label_list.append(label_dict[k])
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:])))
                       for k, v in zip(self.label_name, label_list)]
        return max_data_shape, label_shape

    def get_batch(self):
        deq = self.q_out.get()
        print('q_out got')
        data_list, label_list = deq

        for data, label in zip(data_list, label_list):
            data_shape = {k: v.shape for k, v in data.items()}
            del data_shape['im_info']
            feat_shape_list = []
            for s in range(len(self.feat_stride)):
                _, feat_shape, _ = self.feat_sym[s].infer_shape(**data_shape)
                feat_shape = [int(i) for i in feat_shape[0]]
                feat_shape_list.append(feat_shape)
            #for k in self.label_name:
            #  label[k] = [0 for i in range(config.TRAIN.BATCH_IMAGES)]
            im_info = data['im_info']
            gt_boxes = label['gt_boxes']
            gt_label = {'gt_boxes': gt_boxes}
            label_dict = {}
            head_label_dict = assign_anchor_fpn(feat_shape_list,
                                                gt_label,
                                                im_info,
                                                False,
                                                prefix='head')
            label_dict.update(head_label_dict)
            if config.FACE_LANDMARK:
                gt_landmarks = label['gt_landmarks']
                gt_label['gt_landmarks'] = gt_landmarks
            face_label_dict = assign_anchor_fpn(feat_shape_list,
                                                gt_label,
                                                im_info,
                                                config.FACE_LANDMARK,
                                                prefix='face')
            label_dict.update(face_label_dict)
            #print('im_info', im_info.shape)
            #print(gt_boxes.shape)
            for k in self.label_name:
                label[k] = label_dict[k]

        all_data = dict()
        for key in self.data_name:
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in self.label_name:
            pad = 0 if key.startswith('bbox_') else -1
            #print('label vstack', key, pad, len(label_list), file=sys.stderr)
            all_label[key] = tensor_vstack(
                [batch[key] for batch in label_list], pad=pad)
        self.data = [mx.nd.array(all_data[key]) for key in self.data_name]
        self.label = [mx.nd.array(all_label[key]) for key in self.label_name]
