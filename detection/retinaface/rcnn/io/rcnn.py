"""
Fast R-CNN:
data =
    {'data': [num_images, c, h, w],
    'rois': [num_rois, 5]}
label =
    {'label': [num_rois],
    'bbox_target': [num_rois, 4 * num_classes],
    'bbox_weight': [num_rois, 4 * num_classes]}
roidb extended format [image_index]
    ['image', 'height', 'width', 'flipped',
     'boxes', 'gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'bbox_targets']
"""

import numpy as np
import numpy.random as npr

from ..config import config
from ..io.image import get_image, tensor_vstack
from ..processing.bbox_transform import bbox_overlaps, bbox_transform
from ..processing.bbox_regression import expand_bbox_regression_targets


def get_rcnn_testbatch(roidb):
    """
    return a dict of testbatch
    :param roidb: ['image', 'flipped'] + ['boxes']
    :return: data, label, im_info
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

    im_rois = roidb[0]['boxes']
    rois = im_rois
    batch_index = 0 * np.ones((rois.shape[0], 1))
    rois_array = np.hstack((batch_index, rois))[np.newaxis, :]

    data = {'data': im_array, 'rois': rois_array, 'im_info': im_info}
    label = {}

    return data, label


def get_rcnn_batch(roidb):
    """
    return a dict of multiple images
    :param roidb: a list of dict, whose length controls batch size
    ['images', 'flipped'] + ['gt_boxes', 'boxes', 'gt_overlap'] => ['bbox_targets']
    :return: data, label
    """
    num_images = len(roidb)
    imgs, roidb = get_image(roidb)
    im_array = tensor_vstack(imgs)

    assert config.TRAIN.BATCH_ROIS % config.TRAIN.BATCH_IMAGES == 0, \
        'BATCHIMAGES {} must divide BATCH_ROIS {}'.format(config.TRAIN.BATCH_IMAGES, config.TRAIN.BATCH_ROIS)
    rois_per_image = int(config.TRAIN.BATCH_ROIS / config.TRAIN.BATCH_IMAGES)
    fg_rois_per_image = int(round(config.TRAIN.FG_FRACTION * rois_per_image))

    rois_array = list()
    labels_array = list()
    bbox_targets_array = list()
    bbox_weights_array = list()

    for im_i in range(num_images):
        roi_rec = roidb[im_i]

        # infer num_classes from gt_overlaps
        num_classes = roi_rec['gt_overlaps'].shape[1]

        # label = class RoI has max overlap with
        rois = roi_rec['boxes']
        labels = roi_rec['max_classes']
        overlaps = roi_rec['max_overlaps']
        bbox_targets = roi_rec['bbox_targets']

        im_rois, labels, bbox_targets, bbox_weights = \
            sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes,
                        labels, overlaps, bbox_targets)

        # project im_rois
        # do not round roi
        rois = im_rois
        batch_index = im_i * np.ones((rois.shape[0], 1))
        rois_array_this_image = np.hstack((batch_index, rois))
        rois_array.append(rois_array_this_image)

        # add labels
        labels_array.append(labels)
        bbox_targets_array.append(bbox_targets)
        bbox_weights_array.append(bbox_weights)

    rois_array = np.array(rois_array)
    labels_array = np.array(labels_array)
    bbox_targets_array = np.array(bbox_targets_array)
    bbox_weights_array = np.array(bbox_weights_array)

    data = {'data': im_array, 'rois': rois_array}
    label = {
        'label': labels_array,
        'bbox_target': bbox_targets_array,
        'bbox_weight': bbox_weights_array
    }

    return data, label


def sample_rois(rois,
                fg_rois_per_image,
                rois_per_image,
                num_classes,
                labels=None,
                overlaps=None,
                bbox_targets=None,
                gt_boxes=None):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param rois: all_rois [n, 4]; e2e: [n, 5] with batch_index
    :param fg_rois_per_image: foreground roi number
    :param rois_per_image: total roi number
    :param num_classes: number of classes
    :param labels: maybe precomputed
    :param overlaps: maybe precomputed (max_overlaps)
    :param bbox_targets: maybe precomputed
    :param gt_boxes: optional for e2e [n, 5] (x1, y1, x2, y2, cls)
    :return: (labels, rois, bbox_targets, bbox_weights)
    """
    if labels is None:
        overlaps = bbox_overlaps(rois[:, 1:].astype(np.float),
                                 gt_boxes[:, :4].astype(np.float))
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 4]

    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes,
                                size=fg_rois_per_this_image,
                                replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < config.TRAIN.BG_THRESH_HI)
                          & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes,
                                size=bg_rois_per_this_image,
                                replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)
    neg_idx = np.where(overlaps < config.TRAIN.FG_THRESH)[0]
    neg_rois = rois[neg_idx]
    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(neg_rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(neg_rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, neg_idx[gap_indexes])

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]

    # load or compute bbox_target
    if bbox_targets is not None:
        bbox_target_data = bbox_targets[keep_indexes, :]
    else:
        targets = bbox_transform(rois[:, 1:],
                                 gt_boxes[gt_assignment[keep_indexes], :4])
        if config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(config.TRAIN.BBOX_MEANS)) /
                       np.array(config.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets(bbox_target_data, num_classes)

    return rois, labels, bbox_targets, bbox_weights


def get_fpn_rcnn_testbatch(roidb):
    """
    return a dict of testbatch
    :param roidb: ['image', 'flipped'] + ['boxes']
    :return: data, label, im_info
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

    im_rois = roidb[0]['boxes']
    rois = im_rois

    # assign rois
    rois_area = np.sqrt((rois[:, 2] - rois[:, 0]) * (rois[:, 3] - rois[:, 1]))
    area_threshold = {'P5': 448, 'P4': 224, 'P3': 112}
    rois_p5 = rois[area_threshold['P5'] <= rois_area]
    rois_p4 = rois[np.logical_and(area_threshold['P4'] <= rois_area,
                                  rois_area < area_threshold['P5'])]
    rois_p3 = rois[np.logical_and(area_threshold['P3'] <= rois_area,
                                  rois_area < area_threshold['P4'])]
    rois_p2 = rois[np.logical_and(0 < rois_area,
                                  rois_area < area_threshold['P3'])]

    # pad a virtual rois if on rois assigned
    if rois_p5.size == 0:
        rois_p5 = np.array([[12, 34, 56, 78]])
    if rois_p4.size == 0:
        rois_p4 = np.array([[12, 34, 56, 78]])
    if rois_p3.size == 0:
        rois_p3 = np.array([[12, 34, 56, 78]])
    if rois_p2.size == 0:
        rois_p2 = np.array([[12, 34, 56, 78]])

    p5_batch_index = 0 * np.ones((rois_p5.shape[0], 1))
    rois_p5_array = np.hstack((p5_batch_index, rois_p5))[np.newaxis, :]

    p4_batch_index = 0 * np.ones((rois_p4.shape[0], 1))
    rois_p4_array = np.hstack((p4_batch_index, rois_p4))[np.newaxis, :]

    p3_batch_index = 0 * np.ones((rois_p3.shape[0], 1))
    rois_p3_array = np.hstack((p3_batch_index, rois_p3))[np.newaxis, :]

    p2_batch_index = 0 * np.ones((rois_p2.shape[0], 1))
    rois_p2_array = np.hstack((p2_batch_index, rois_p2))[np.newaxis, :]

    data = {
        'data': im_array,
        'rois_stride32': rois_p5_array,
        'rois_stride16': rois_p4_array,
        'rois_stride8': rois_p3_array,
        'rois_stride4': rois_p2_array
    }
    label = {}

    return data, label, im_info


def get_fpn_maskrcnn_batch(roidb):
    """
    return a dictionary that contains raw data.
    """
    num_images = len(roidb)
    imgs, roidb = get_image(roidb, scale=config.TRAIN.SCALE)  #TODO
    #imgs, roidb = get_image(roidb)
    im_array = tensor_vstack(imgs)

    assert config.TRAIN.BATCH_ROIS % config.TRAIN.BATCH_IMAGES == 0, \
        'BATCHIMAGES {} must divide BATCH_ROIS {}'.format(config.TRAIN.BATCH_IMAGES, config.TRAIN.BATCH_ROIS)
    rois_per_image = config.TRAIN.BATCH_ROIS / config.TRAIN.BATCH_IMAGES
    fg_rois_per_image = np.round(config.TRAIN.FG_FRACTION *
                                 rois_per_image).astype(int)

    rois_on_imgs = dict()
    labels_on_imgs = dict()
    bbox_targets_on_imgs = dict()
    bbox_weights_on_imgs = dict()
    mask_targets_on_imgs = dict()
    mask_weights_on_imgs = dict()
    for s in config.RCNN_FEAT_STRIDE:
        rois_on_imgs.update({'stride%s' % s: list()})
        labels_on_imgs.update({'stride%s' % s: list()})
        bbox_targets_on_imgs.update({'stride%s' % s: list()})
        bbox_weights_on_imgs.update({'stride%s' % s: list()})
        mask_targets_on_imgs.update({'stride%s' % s: list()})
        mask_weights_on_imgs.update({'stride%s' % s: list()})

    # Sample rois
    level_related_data_on_imgs = {}
    for im_i in range(num_images):
        roi_rec = roidb[im_i]
        # infer num_classes from gt_overlaps
        num_classes = roi_rec['gt_overlaps'].shape[1]
        # label = class RoI has max overlap with
        rois = roi_rec['boxes']
        labels = roi_rec['max_classes']
        overlaps = roi_rec['max_overlaps']
        bbox_targets = roi_rec['bbox_targets']
        im_info = roi_rec['im_info']

        mask_targets = roi_rec['mask_targets']
        mask_labels = roi_rec['mask_labels']
        mask_inds = roi_rec['mask_inds']

        assign_levels = roi_rec['assign_levels']

        im_rois_on_levels, labels_on_levels, bbox_targets_on_levels, bbox_weights_on_levels, mask_targets_on_levels, mask_weights_on_levels = \
            sample_rois_fpn(rois, assign_levels, fg_rois_per_image, rois_per_image, num_classes,
                            labels, overlaps, bbox_targets, mask_targets=mask_targets, mask_labels=mask_labels, mask_inds=mask_inds, im_info=im_info)

        level_related_data_on_imgs.update({
            'img_%s' % im_i: {
                'rois_on_levels': im_rois_on_levels,
                'labels_on_levels': labels_on_levels,
                'bbox_targets_on_levels': bbox_targets_on_levels,
                'bbox_weights_on_levels': bbox_weights_on_levels,
                'mask_targets_on_levels': mask_targets_on_levels,
                'mask_weights_on_levels': mask_weights_on_levels,
            }
        })

    return im_array, level_related_data_on_imgs


def sample_rois(rois,
                fg_rois_per_image,
                rois_per_image,
                num_classes,
                labels=None,
                overlaps=None,
                bbox_targets=None,
                gt_boxes=None,
                mask_targets=None,
                mask_labels=None,
                mask_inds=None):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param rois: all_rois [n, 4]; e2e: [n, 5] with batch_index
    :param fg_rois_per_image: foreground roi number
    :param rois_per_image: total roi number
    :param num_classes: number of classes
    :param labels: maybe precomputed
    :param overlaps: maybe precomputed (max_overlaps)
    :param bbox_targets: maybe precomputed
    :param gt_boxes: optional for e2e [n, 5] (x1, y1, x2, y2, cls)
    :return: (rois, labels, bbox_targets, bbox_weights)
    """
    if labels is None:
        if len(gt_boxes) == 0:
            gt_boxes = np.zeros((1, 5))
            gt_assignment = np.zeros((len(rois), ), dtype=np.int32)
            overlaps = np.zeros((len(rois), ))
            labels = np.zeros((len(rois), ))
        else:
            overlaps = bbox_overlaps(rois[:, 1:].astype(np.float),
                                     gt_boxes[:, :4].astype(np.float))
            gt_assignment = overlaps.argmax(axis=1)
            overlaps = overlaps.max(axis=1)
            labels = gt_boxes[gt_assignment, 4]

    num_rois = rois.shape[0]
    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes,
                                size=fg_rois_per_this_image,
                                replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < config.TRAIN.BG_THRESH_HI)
                          & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes,
                                size=bg_rois_per_this_image,
                                replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    neg_idx = np.where(overlaps < config.TRAIN.FG_THRESH)[0]
    neg_rois = rois[neg_idx]

    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(neg_rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(neg_rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, neg_idx[gap_indexes])

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]
    if mask_targets is not None:
        assert mask_labels is not None
        assert mask_inds is not None

        def _mask_umap(mask_targets, mask_labels, mask_inds):
            _mask_targets = np.zeros((num_rois, num_classes, 28, 28),
                                     dtype=np.int8)
            _mask_weights = np.zeros((num_rois, num_classes, 28, 28),
                                     dtype=np.int8)
            _mask_targets[mask_inds, mask_labels] = mask_targets
            _mask_weights[mask_inds, mask_labels] = 1
            _mask_weights[:, 0] = 0  # set background mask weight to zeros
            return _mask_targets, _mask_weights  # [num_rois, num_classes, 28, 28]

        mask_targets, mask_weights = _mask_umap(mask_targets, mask_labels,
                                                mask_inds)
        mask_targets = mask_targets[keep_indexes]
        mask_weights = mask_weights[keep_indexes]

    # load or compute bbox_target
    if bbox_targets is not None:
        bbox_target_data = bbox_targets[keep_indexes, :]
    else:
        targets = bbox_transform(rois[:, 1:],
                                 gt_boxes[gt_assignment[keep_indexes], :4])
        if config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(config.TRAIN.BBOX_MEANS)) /
                       np.array(config.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets(bbox_target_data, num_classes)

    if mask_targets is not None:
        return rois, labels, bbox_targets, bbox_weights, mask_targets, mask_weights
    else:
        return rois, labels, bbox_targets, bbox_weights


def sample_rois_fpn(rois,
                    assign_levels,
                    fg_rois_per_image,
                    rois_per_image,
                    num_classes,
                    labels=None,
                    overlaps=None,
                    bbox_targets=None,
                    mask_targets=None,
                    mask_labels=None,
                    mask_inds=None,
                    gt_boxes=None,
                    im_info=None):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param rois: all_rois [n, 4]; e2e: [n, 5] with batch_index
    :param assign_levels: [n]
    :param fg_rois_per_image: foreground roi number
    :param rois_per_image: total roi number
    :param num_classes: number of classes
    :param labels: maybe precomputed
    :param overlaps: maybe precomputed (max_overlaps)
    :param bbox_targets: maybe precomputed
    :param gt_boxes: optional for e2e [n, 5] (x1, y1, x2, y2, cls)
    :return: (rois, labels, bbox_targets, bbox_weights)
    """
    DEBUG = False
    if labels is None:
        if len(gt_boxes) == 0:
            gt_boxes = np.zeros((1, 5))
            gt_assignment = np.zeros((len(rois), ), dtype=np.int32)
            overlaps = np.zeros((len(rois), ))
            labels = np.zeros((len(rois), ))
        else:
            overlaps = bbox_overlaps(rois[:, 1:].astype(np.float),
                                     gt_boxes[:, :4].astype(np.float))
            gt_assignment = overlaps.argmax(axis=1)
            overlaps = overlaps.max(axis=1)
            labels = gt_boxes[gt_assignment, 4]

    num_rois = rois.shape[0]
    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)

    if DEBUG:
        print 'fg total num:', len(fg_indexes)

    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes,
                                size=fg_rois_per_this_image,
                                replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < config.TRAIN.BG_THRESH_HI)
                          & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
    if DEBUG:
        print 'bg total num:', len(bg_indexes)
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes,
                                size=bg_rois_per_this_image,
                                replace=False)
    if DEBUG:
        print 'fg num:', len(fg_indexes)
        print 'bg num:', len(bg_indexes)

    # bg rois statistics
    if DEBUG:
        bg_assign = assign_levels[bg_indexes]
        bg_rois_on_levels = dict()
        for i, s in enumerate(config.RCNN_FEAT_STRIDE):
            bg_rois_on_levels.update(
                {'stride%s' % s: len(np.where(bg_assign == s)[0])})
        print bg_rois_on_levels

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    neg_idx = np.where(overlaps < config.TRAIN.FG_THRESH)[0]
    neg_rois = rois[neg_idx]

    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(neg_rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(neg_rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, neg_idx[gap_indexes])

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]
    assign_levels = assign_levels[keep_indexes]

    if mask_targets is not None:
        assert mask_labels is not None
        assert mask_inds is not None

        def _mask_umap(mask_targets, mask_labels, mask_inds):
            _mask_targets = np.zeros((num_rois, num_classes, 28, 28),
                                     dtype=np.int8)
            _mask_weights = np.zeros((num_rois, num_classes, 1, 1),
                                     dtype=np.int8)
            _mask_targets[mask_inds, mask_labels] = mask_targets
            _mask_weights[mask_inds, mask_labels] = 1
            return _mask_targets, _mask_weights  # [num_rois, num_classes, 28, 28]

        mask_targets, mask_weights = _mask_umap(mask_targets, mask_labels,
                                                mask_inds)
        mask_targets = mask_targets[keep_indexes]
        mask_weights = mask_weights[keep_indexes]

    # load or compute bbox_target
    if bbox_targets is not None:
        bbox_target_data = bbox_targets[keep_indexes, :]
    else:
        targets = bbox_transform(rois[:, 1:],
                                 gt_boxes[gt_assignment[keep_indexes], :4])
        if config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(config.TRAIN.BBOX_MEANS)) /
                       np.array(config.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets(bbox_target_data, num_classes)

    # Assign to levels
    rois_on_levels = dict()
    labels_on_levels = dict()
    bbox_targets_on_levels = dict()
    bbox_weights_on_levels = dict()
    if mask_targets is not None:
        mask_targets_on_levels = dict()
        mask_weights_on_levels = dict()
    for i, s in enumerate(config.RCNN_FEAT_STRIDE):
        index = np.where(assign_levels == s)
        _rois = rois[index]
        _labels = labels[index]
        _bbox_targets = bbox_targets[index]
        _bbox_weights = bbox_weights[index]
        if mask_targets is not None:
            _mask_targets = mask_targets[index]
            _mask_weights = mask_weights[index]

        rois_on_levels.update({'stride%s' % s: _rois})
        labels_on_levels.update({'stride%s' % s: _labels})
        bbox_targets_on_levels.update({'stride%s' % s: _bbox_targets})
        bbox_weights_on_levels.update({'stride%s' % s: _bbox_weights})
        if mask_targets is not None:
            mask_targets_on_levels.update({'stride%s' % s: _mask_targets})
            mask_weights_on_levels.update({'stride%s' % s: _mask_weights})

    if mask_targets is not None:
        return rois_on_levels, labels_on_levels, bbox_targets_on_levels, bbox_weights_on_levels, mask_targets_on_levels, mask_weights_on_levels
    else:
        return rois_on_levels, labels_on_levels, bbox_targets_on_levels, bbox_weights_on_levels


def get_rois(rois,
             rois_per_image,
             num_classes,
             labels=None,
             overlaps=None,
             bbox_targets=None,
             gt_boxes=None):
    """
    get top N ROIs, used in online hard example mining
    :param rois: all_rois [n, 4]; e2e: [n, 5] with batch_index
    :param rois_per_image: total roi number
    :param num_classes: number of classes
    :param labels: maybe precomputed
    :param overlaps: maybe precomputed (max_overlaps)
    :param bbox_targets: maybe precomputed
    :param gt_boxes: optional for e2e [n, 5] (x1, y1, x2, y2, cls)
    :return: (rois, labels, bbox_targets, bbox_weights)
    """
    if labels is None:
        if len(gt_boxes) == 0:
            gt_boxes = np.array([[1, 1, 1, 1, 0]])
        overlaps = bbox_overlaps(rois[:, 1:].astype(np.float),
                                 gt_boxes[:, :4].astype(np.float))
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 4]

    # select indices
    keep_indexes = np.arange(rois.shape[0])
    if keep_indexes.shape[0] > rois_per_image:
        keep_indexes = npr.choice(keep_indexes,
                                  size=rois_per_image,
                                  replace=False)

    # if not enough, pad until rois_per_image is satisfied
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(rois_per_image - keep_indexes.shape[0], len(rois))
        gap_indexes = npr.choice(range(len(rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, gap_indexes)

    # suppress any bg defined by overlap
    bg_indexes = np.where((overlaps < config.TRAIN.BG_THRESH_HI)
                          & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
    labels[bg_indexes] = 0

    labels = labels[keep_indexes]
    rois = rois[keep_indexes]

    # load or compute bbox_target
    if bbox_targets is not None:
        bbox_target_data = bbox_targets[keep_indexes, :]
    else:
        targets = bbox_transform(rois[:, 1:],
                                 gt_boxes[gt_assignment[keep_indexes], :4])
        if config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(config.TRAIN.BBOX_MEANS)) /
                       np.array(config.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets(bbox_target_data, num_classes)

    return rois, labels, bbox_targets, bbox_weights
