from __future__ import print_function
try:
    import cPickle as pickle
except ImportError:
    import pickle
import cv2
import os
import numpy as np
import json
#from PIL import Image

from ..logger import logger
from .imdb import IMDB
from .ds_utils import unique_boxes, filter_small_boxes
from ..config import config


class retinaface(IMDB):
    def __init__(self, image_set, root_path, data_path):
        super(retinaface, self).__init__('retinaface', image_set, root_path,
                                         data_path)
        #assert image_set=='train'

        split = image_set
        self._split = image_set
        self._image_set = image_set

        self.root_path = root_path
        self.data_path = data_path

        self._dataset_path = self.data_path
        self._imgs_path = os.path.join(self._dataset_path, image_set, 'images')
        self._fp_bbox_map = {}
        label_file = os.path.join(self._dataset_path, image_set, 'label.txt')
        name = None
        for line in open(label_file, 'r'):
            line = line.strip()
            if line.startswith('#'):
                name = line[1:].strip()
                self._fp_bbox_map[name] = []
                continue
            assert name is not None
            assert name in self._fp_bbox_map
            self._fp_bbox_map[name].append(line)
        print('origin image size', len(self._fp_bbox_map))

        #self.num_images = len(self._image_paths)
        #self._image_index = range(len(self._image_paths))
        self.classes = ['bg', 'face']
        self.num_classes = len(self.classes)

    def gt_roidb(self):
        cache_file = os.path.join(
            self.cache_path,
            '{}_{}_gt_roidb.pkl'.format(self.name, self._split))
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            self.num_images = len(roidb)
            return roidb

        roidb = []
        max_num_boxes = 0
        nonattr_box_num = 0
        landmark_num = 0

        pp = 0
        for fp in self._fp_bbox_map:
            pp += 1
            if pp % 1000 == 0:
                print('loading', pp)
            if self._split == 'test':
                image_path = os.path.join(self._imgs_path, fp)
                roi = {'image': image_path}
                roidb.append(roi)
                continue
            boxes = np.zeros([len(self._fp_bbox_map[fp]), 4], np.float32)
            landmarks = np.zeros([len(self._fp_bbox_map[fp]), 5, 3], np.float32)
            blur = np.zeros((len(self._fp_bbox_map[fp]), ), np.float32)
            boxes_mask = []

            gt_classes = np.ones([len(self._fp_bbox_map[fp])], np.int32)
            overlaps = np.zeros([len(self._fp_bbox_map[fp]), 2], np.float32)

            imsize = cv2.imread(os.path.join(self._imgs_path,
                                             fp)).shape[0:2][::-1]
            ix = 0

            for aline in self._fp_bbox_map[fp]:
                #imsize = Image.open(os.path.join(self._imgs_path, fp)).size
                values = [float(x) for x in aline.strip().split()]
                bbox = [
                    values[0], values[1], values[0] + values[2],
                    values[1] + values[3]
                ]

                x1 = bbox[0]
                y1 = bbox[1]
                x2 = min(imsize[0], bbox[2])
                y2 = min(imsize[1], bbox[3])
                if x1 >= x2 or y1 >= y2:
                    continue

                if config.BBOX_MASK_THRESH > 0:
                    if (
                            x2 - x1
                    ) < config.BBOX_MASK_THRESH or y2 - y1 < config.BBOX_MASK_THRESH:
                        boxes_mask.append(np.array([x1, y1, x2, y2], np.float32))
                        continue
                if (
                        x2 - x1
                ) < config.TRAIN.MIN_BOX_SIZE or y2 - y1 < config.TRAIN.MIN_BOX_SIZE:
                    continue

                boxes[ix, :] = np.array([x1, y1, x2, y2], np.float32)
                if self._split == 'train':
                    landmark = np.array(values[4:19],
                                        dtype=np.float32).reshape((5, 3))
                    for li in range(5):
                        #print(landmark)
                        if landmark[li][0] == -1. and landmark[li][
                                1] == -1.:  #missing landmark
                            assert landmark[li][2] == -1
                        else:
                            assert landmark[li][2] >= 0
                            if li == 0:
                                landmark_num += 1
                            if landmark[li][2] == 0.0:  #visible
                                landmark[li][2] = 1.0
                            else:
                                landmark[li][2] = 0.0

                    landmarks[ix] = landmark

                    blur[ix] = values[19]
                    #print(aline, blur[ix])
                    if blur[ix] < 0.0:
                        blur[ix] = 0.3
                        nonattr_box_num += 1

                cls = int(1)
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
                ix += 1
            max_num_boxes = max(max_num_boxes, ix)
            #overlaps = scipy.sparse.csr_matrix(overlaps)
            if self._split == 'train' and ix == 0:
                continue
            boxes = boxes[:ix, :]
            landmarks = landmarks[:ix, :, :]
            blur = blur[:ix]
            gt_classes = gt_classes[:ix]
            overlaps = overlaps[:ix, :]
            image_path = os.path.join(self._imgs_path, fp)
            with open(image_path, 'rb') as fin:
                stream = fin.read()
            stream = np.fromstring(stream, dtype=np.uint8)

            roi = {
                'image': image_path,
                'stream': stream,
                'height': imsize[1],
                'width': imsize[0],
                'boxes': boxes,
                'landmarks': landmarks,
                'blur': blur,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'max_classes': overlaps.argmax(axis=1),
                'max_overlaps': overlaps.max(axis=1),
                'flipped': False,
            }
            if len(boxes_mask) > 0:
                boxes_mask = np.array(boxes_mask)
                roi['boxes_mask'] = boxes_mask
            roidb.append(roi)
        for roi in roidb:
            roi['max_num_boxes'] = max_num_boxes
        self.num_images = len(roidb)
        print('roidb size', len(roidb))
        print('non attr box num', nonattr_box_num)
        print('landmark num', landmark_num)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return roidb

    def write_detections(self, all_boxes, output_dir='./output/'):
        pass

    def evaluate_detections(self,
                            all_boxes,
                            output_dir='./output/',
                            method_name='insightdetection'):
        pass
