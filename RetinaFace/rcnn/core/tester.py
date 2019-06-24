from __future__ import print_function
try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import sys
import time
import mxnet as mx
import numpy as np
from builtins import range

from mxnet.module import Module
from .module import MutableModule
from rcnn.logger import logger
from rcnn.config import config
from rcnn.io import image
from rcnn.processing.bbox_transform import bbox_pred, clip_boxes
from rcnn.processing.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from rcnn.processing.bbox_transform import bbox_overlaps



def IOU(Reframe,GTframe):
  x1 = Reframe[0];
  y1 = Reframe[1];
  width1 = Reframe[2]-Reframe[0];
  height1 = Reframe[3]-Reframe[1];

  x2 = GTframe[0]
  y2 = GTframe[1]
  width2 = GTframe[2]-GTframe[0]
  height2 = GTframe[3]-GTframe[1]

  endx = max(x1+width1,x2+width2)
  startx = min(x1,x2)
  width = width1+width2-(endx-startx)

  endy = max(y1+height1,y2+height2)
  starty = min(y1,y2)
  height = height1+height2-(endy-starty)

  if width <=0 or height <= 0:
    ratio = 0
  else:
    Area = width*height
    Area1 = width1*height1
    Area2 = width2*height2
    ratio = Area*1./(Area1+Area2-Area)
  return ratio

class Predictor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        #self._mod = MutableModule(symbol, data_names, label_names,
        #                          context=context, max_data_shapes=max_data_shapes)
        self._mod = Module(symbol, data_names, label_names, context=context)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict(self, data_batch):
        self._mod.forward(data_batch)
        return dict(zip(self._mod.output_names, self._mod.get_outputs())) #TODO
        #return self._mod.get_outputs()


def im_proposal(predictor, data_batch, data_names, scale):
    data_dict = dict(zip(data_names, data_batch.data))
    output = predictor.predict(data_batch)

    # drop the batch index
    boxes = output['rois_output'].asnumpy()[:, 1:]
    scores = output['rois_score'].asnumpy()

    # transform to original scale
    boxes = boxes / scale

    return scores, boxes, data_dict

def _im_proposal(predictor, data_batch, data_names, scale):
    data_dict = dict(zip(data_names, data_batch.data))
    output = predictor.predict(data_batch)
    print('output', output)

    # drop the batch index
    boxes = output['rois_output'].asnumpy()[:, 1:]
    scores = output['rois_score'].asnumpy()

    # transform to original scale
    boxes = boxes / scale

    return scores, boxes, data_dict


def generate_proposals(predictor, test_data, imdb, vis=False, thresh=0.):
    """
    Generate detections results using RPN.
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffled
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: thresh for valid detections
    :return: list of detected boxes
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    i = 0
    t = time.time()
    imdb_boxes = list()
    original_boxes = list()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scale = im_info[0, 2]
        scores, boxes, data_dict = im_proposal(predictor, data_batch, data_names, scale)
        print(scores.shape, boxes.shape, file=sys.stderr)
        t2 = time.time() - t
        t = time.time()

        # assemble proposals
        dets = np.hstack((boxes, scores))
        original_boxes.append(dets)

        # filter proposals
        keep = np.where(dets[:, 4:] > thresh)[0]
        dets = dets[keep, :]
        imdb_boxes.append(dets)

        if vis:
            vis_all_detection(data_dict['data'].asnumpy(), [dets], ['obj'], scale)

        logger.info('generating %d/%d ' % (i + 1, imdb.num_images) +
                    'proposal %d ' % (dets.shape[0]) +
                    'data %.4fs net %.4fs' % (t1, t2))
        i += 1

    assert len(imdb_boxes) == imdb.num_images, 'calculations not complete'

    # save results
    rpn_folder = os.path.join(imdb.root_path, 'rpn_data')
    if not os.path.exists(rpn_folder):
        os.mkdir(rpn_folder)

    rpn_file = os.path.join(rpn_folder, imdb.name + '_rpn.pkl')
    with open(rpn_file, 'wb') as f:
        pickle.dump(imdb_boxes, f, pickle.HIGHEST_PROTOCOL)

    if thresh > 0:
        full_rpn_file = os.path.join(rpn_folder, imdb.name + '_full_rpn.pkl')
        with open(full_rpn_file, 'wb') as f:
            pickle.dump(original_boxes, f, pickle.HIGHEST_PROTOCOL)

    logger.info('wrote rpn proposals to %s' % rpn_file)
    return imdb_boxes

def test_proposals(predictor, test_data, imdb, roidb, vis=False):
    """
    Test detections results using RPN.
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffled
    :param imdb: image database
    :param roidb: roidb 
    :param vis: controls visualization
    :return: recall, mAP
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    #bbox_file = os.path.join(rpn_folder, imdb.name + '_bbox.txt')
    #bbox_f = open(bbox_file, 'w')

    i = 0
    t = time.time()
    output_folder = os.path.join(imdb.root_path, 'output')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    imdb_boxes = list()
    original_boxes = list()
    gt_overlaps = np.zeros(0)
    overall = [0.0, 0.0]
    gt_max = np.array( (0.0, 0.0) )
    num_pos = 0
    #apply scale, for SSH
    #_, roidb = image.get_image(roidb)
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        oscale = im_info[0, 2]
        #print('scale', scale, file=sys.stderr)
        scale = 1.0 #fix scale=1.0 for SSH face detector
        scores, boxes, data_dict = im_proposal(predictor, data_batch, data_names, scale)
        #print(scores.shape, boxes.shape, file=sys.stderr)
        t2 = time.time() - t
        t = time.time()

        # assemble proposals
        dets = np.hstack((boxes, scores))
        original_boxes.append(dets)

        # filter proposals
        keep = np.where(dets[:, 4:] > config.TEST.SCORE_THRESH)[0]
        dets = dets[keep, :]
        imdb_boxes.append(dets)


        logger.info('generating %d/%d ' % (i + 1, imdb.num_images) +
                    'proposal %d ' % (dets.shape[0]) +
                    'data %.4fs net %.4fs' % (t1, t2))

        #if dets.shape[0]==0:
        #  continue
        if vis:
            vis_all_detection(data_dict['data'].asnumpy(), [dets], ['obj'], scale)
        boxes = dets
        #max_gt_overlaps = roidb[i]['gt_overlaps'].max(axis=1)
        #gt_inds = np.where((roidb[i]['gt_classes'] > 0) & (max_gt_overlaps == 1))[0]
        #gt_boxes = roidb[i]['boxes'][gt_inds, :]
        gt_boxes = roidb[i]['boxes'].copy() * oscale # as roidb is the original one, need to scale GT for SSH
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
        num_pos += gt_boxes.shape[0]

        overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))
        #print(im_info, gt_boxes.shape, boxes.shape, overlaps.shape, file=sys.stderr)

        _gt_overlaps = np.zeros((gt_boxes.shape[0]))
        # choose whatever is smaller to iterate

        #for j in range(gt_boxes.shape[0]):
        #  print('gt %d,%d,%d,%d'% (gt_boxes[j][0], gt_boxes[j][1], gt_boxes[j][2]-gt_boxes[j][0], gt_boxes[j][3]-gt_boxes[j][1]), file=sys.stderr)
        #  gt_max = np.maximum( gt_max, np.array( (gt_boxes[j][2], gt_boxes[j][3]) ) )
        #print('gt max', gt_max, file=sys.stderr)
        #for j in range(boxes.shape[0]):
        #  print('anchor_box %.2f,%.2f,%.2f,%.2f'% (boxes[j][0], boxes[j][1], boxes[j][2]-boxes[j][0], boxes[j][3]-boxes[j][1]), file=sys.stderr)

        #rounds = min(boxes.shape[0], gt_boxes.shape[0])
        #for j in range(rounds):
        #    # find which proposal maximally covers each gt box
        #    argmax_overlaps = overlaps.argmax(axis=0)
        #    print(j, 'argmax_overlaps', argmax_overlaps, file=sys.stderr)
        #    # get the IoU amount of coverage for each gt box
        #    max_overlaps = overlaps.max(axis=0)
        #    print(j, 'max_overlaps', max_overlaps, file=sys.stderr)
        #    # find which gt box is covered by most IoU
        #    gt_ind = max_overlaps.argmax()
        #    gt_ovr = max_overlaps.max()
        #    assert (gt_ovr >= 0), '%s\n%s\n%s' % (boxes, gt_boxes, overlaps)
        #    # find the proposal box that covers the best covered gt box
        #    box_ind = argmax_overlaps[gt_ind]
        #    print('max box', gt_ind, box_ind, (boxes[box_ind][0], boxes[box_ind][1], boxes[box_ind][2]-boxes[box_ind][0], boxes[box_ind][3]-boxes[box_ind][1], boxes[box_ind][4]), file=sys.stderr)
        #    # record the IoU coverage of this gt box
        #    _gt_overlaps[j] = overlaps[box_ind, gt_ind]
        #    assert (_gt_overlaps[j] == gt_ovr)
        #    # mark the proposal box and the gt box as used
        #    overlaps[box_ind, :] = -1
        #    overlaps[:, gt_ind] = -1

        if boxes.shape[0]>0:
          _gt_overlaps = overlaps.max(axis=0)
          #print('max_overlaps', _gt_overlaps, file=sys.stderr)
          for j in range(len(_gt_overlaps)):
            if _gt_overlaps[j]>config.TEST.IOU_THRESH:
              continue
            print(j, 'failed', gt_boxes[j],  'max_overlap:', _gt_overlaps[j], file=sys.stderr)
            #_idx = np.where(overlaps[:,j]>0.4)[0]
            #print(j, _idx, file=sys.stderr)
            #print(overlaps[_idx,j], file=sys.stderr)
            #for __idx in _idx:
            #  print(gt_boxes[j], boxes[__idx], overlaps[__idx,j], IOU(gt_boxes[j], boxes[__idx,0:4]), file=sys.stderr)

          # append recorded IoU coverage level
          found = (_gt_overlaps > config.TEST.IOU_THRESH).sum()
          _recall = found / float(gt_boxes.shape[0])
          print('recall', _recall, gt_boxes.shape[0], boxes.shape[0], gt_areas, file=sys.stderr)
          overall[0]+=found
          overall[1]+=gt_boxes.shape[0]
          #gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))
          #_recall = (gt_overlaps >= threshold).sum() / float(num_pos)
          _recall = float(overall[0])/overall[1]
          print('recall_all', _recall, file=sys.stderr)


        boxes[:,0:4] /= oscale
        _vec = roidb[i]['image'].split('/')
        out_dir = os.path.join(output_folder, _vec[-2])
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_file = os.path.join(out_dir, _vec[-1].replace('jpg', 'txt'))
        with open(out_file, 'w') as f:
          name = '/'.join(roidb[i]['image'].split('/')[-2:])
          f.write("%s\n"%(name))
          f.write("%d\n"%(boxes.shape[0]))
          for b in range(boxes.shape[0]):
            box = boxes[b]
            f.write("%d %d %d %d %g \n"%(box[0], box[1], box[2]-box[0], box[3]-box[1], box[4]))
        i += 1

    #bbox_f.close()
    return
    gt_overlaps = np.sort(gt_overlaps)
    recalls = np.zeros_like(thresholds)

    # compute recall for each IoU threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
    ar = recalls.mean()

    # print results
    print('average recall for {}: {:.3f}'.format(area_name, ar))
    for threshold, recall in zip(thresholds, recalls):
        print('recall @{:.2f}: {:.3f}'.format(threshold, recall))




    assert len(imdb_boxes) == imdb.num_images, 'calculations not complete'

    # save results

    rpn_file = os.path.join(rpn_folder, imdb.name + '_rpn.pkl')
    with open(rpn_file, 'wb') as f:
        pickle.dump(imdb_boxes, f, pickle.HIGHEST_PROTOCOL)

    logger.info('wrote rpn proposals to %s' % rpn_file)
    return imdb_boxes

def im_detect(predictor, data_batch, data_names, scale):
    output = predictor.predict(data_batch)

    data_dict = dict(zip(data_names, data_batch.data))
    if config.TEST.HAS_RPN:
        rois = output['rois_output'].asnumpy()[:, 1:]
    else:
        rois = data_dict['rois'].asnumpy().reshape((-1, 5))[:, 1:]
    im_shape = data_dict['data'].shape

    # save output
    scores = output['cls_prob_reshape_output'].asnumpy()[0]
    bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]

    # post processing
    pred_boxes = bbox_pred(rois, bbox_deltas)
    pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])

    # we used scaled image & roi to train, so it is necessary to transform them back
    pred_boxes = pred_boxes / scale

    return scores, pred_boxes, data_dict


def pred_eval(predictor, test_data, imdb, vis=False, thresh=1e-3):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    nms = py_nms_wrapper(config.TEST.NMS)

    # limit detections to max_per_image over all classes
    max_per_image = -1

    num_images = imdb.num_images
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    i = 0
    t = time.time()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scale = im_info[0, 2]
        scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scale)

        t2 = time.time() - t
        t = time.time()

        for j in range(1, imdb.num_classes):
            indexes = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[indexes, j, np.newaxis]
            cls_boxes = boxes[indexes, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets)
            all_boxes[j][i] = cls_dets[keep, :]

        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        if vis:
            boxes_this_image = [[]] + [all_boxes[j][i] for j in range(1, imdb.num_classes)]
            vis_all_detection(data_dict['data'].asnumpy(), boxes_this_image, imdb.classes, scale)

        t3 = time.time() - t
        t = time.time()
        logger.info('testing %d/%d data %.4fs net %.4fs post %.4fs' % (i, imdb.num_images, t1, t2, t3))
        i += 1

    det_file = os.path.join(imdb.cache_path, imdb.name + '_detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, protocol=pickle.HIGHEST_PROTOCOL)

    imdb.evaluate_detections(all_boxes)


def vis_all_detection(im_array, detections, class_names, scale):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import matplotlib.pyplot as plt
    import random
    im = image.transform_inverse(im_array, config.PIXEL_MEANS)
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()


def draw_all_detection(im_array, detections, class_names, scale):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import cv2
    import random
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array, config.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.cv.CV_RGB2BGR)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im
