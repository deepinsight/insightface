from __future__ import division
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
import cv2

__all__ = [
    'FaceDetector', 'retinaface_r50_v1', 'retinaface_mnet025_v1',
    'retinaface_mnet025_v2', 'get_retinaface'
]


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def anchors_plane(height, width, stride, base_anchors):
    """
    Parameters
    ----------
    height: height of plane
    width:  width of plane
    stride: stride ot the original image
    anchors_base: (A, 4) a base set of anchors
    Returns
    -------
    all_anchors: (height, width, A, 4) ndarray of anchors spreading over the plane
    """
    A = base_anchors.shape[0]
    all_anchors = np.zeros((height, width, A, 4), dtype=np.float32)
    for iw in range(width):
        sw = iw * stride
        for ih in range(height):
            sh = ih * stride
            for k in range(A):
                all_anchors[ih, iw, k, 0] = base_anchors[k, 0] + sw
                all_anchors[ih, iw, k, 1] = base_anchors[k, 1] + sh
                all_anchors[ih, iw, k, 2] = base_anchors[k, 2] + sw
                all_anchors[ih, iw, k, 3] = base_anchors[k, 3] + sh
    return all_anchors


def generate_anchors(base_size=16,
                     ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6),
                     stride=16):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([
        _scale_enum(ratio_anchors[i, :], scales)
        for i in range(ratio_anchors.shape[0])
    ])
    return anchors


def generate_anchors_fpn(cfg):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    RPN_FEAT_STRIDE = []
    for k in cfg:
        RPN_FEAT_STRIDE.append(int(k))
    RPN_FEAT_STRIDE = sorted(RPN_FEAT_STRIDE, reverse=True)
    anchors = []
    for k in RPN_FEAT_STRIDE:
        v = cfg[str(k)]
        bs = v['BASE_SIZE']
        __ratios = np.array(v['RATIOS'])
        __scales = np.array(v['SCALES'])
        stride = int(k)
        #print('anchors_fpn', bs, __ratios, __scales, file=sys.stderr)
        r = generate_anchors(bs, __ratios, __scales, stride)
        #print('anchors_fpn', r.shape, file=sys.stderr)
        anchors.append(r)

    return anchors


def clip_pad(tensor, pad_shape):
    """
    Clip boxes of the pad area.
    :param tensor: [n, c, H, W]
    :param pad_shape: [h, w]
    :return: [n, c, h, w]
    """
    H, W = tensor.shape[2:]
    h, w = pad_shape

    if h < H or w < W:
        tensor = tensor[:, :, :h, :w].copy()

    return tensor


def bbox_pred(boxes, box_deltas):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0:1]
    dy = box_deltas[:, 1:2]
    dw = box_deltas[:, 2:3]
    dh = box_deltas[:, 3:4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    if box_deltas.shape[1] > 4:
        pred_boxes[:, 4:] = box_deltas[:, 4:]

    return pred_boxes


def landmark_pred(boxes, landmark_deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, landmark_deltas.shape[1]))
    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
    pred = landmark_deltas.copy()
    for i in range(5):
        pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
        pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y
    return pred


class FaceDetector:
    def __init__(self, param_file, rac):
        self.param_file = param_file
        self.rac = rac
        self.default_image_size = (480, 640)

    def prepare(self, ctx_id, nms=0.4, fix_image_size=None):
        pos = self.param_file.rfind('-')
        prefix = self.param_file[0:pos]
        pos2 = self.param_file.rfind('.')
        epoch = int(self.param_file[pos + 1:pos2])
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        if ctx_id >= 0:
            ctx = mx.gpu(ctx_id)
        else:
            ctx = mx.cpu()
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        if fix_image_size is not None:
            data_shape = (1, 3) + fix_image_size
        else:
            data_shape = (1, 3) + self.default_image_size
        model.bind(data_shapes=[('data', data_shape)])
        model.set_params(arg_params, aux_params)
        #warmup
        data = mx.nd.zeros(shape=data_shape)
        db = mx.io.DataBatch(data=(data, ))
        model.forward(db, is_train=False)
        out = model.get_outputs()[0].asnumpy()
        self.model = model
        self.nms_threshold = nms

        self.landmark_std = 1.0
        _ratio = (1., )
        fmc = 3
        if self.rac == 'net3':
            _ratio = (1., )
        elif self.rac == 'net3l':
            _ratio = (1., )
            self.landmark_std = 0.2
        elif network == 'net5':  #retinaface
            fmc = 5
        else:
            assert False, 'rac setting error %s' % self.rac

        if fmc == 3:
            self._feat_stride_fpn = [32, 16, 8]
            self.anchor_cfg = {
                '32': {
                    'SCALES': (32, 16),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
                '16': {
                    'SCALES': (8, 4),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
                '8': {
                    'SCALES': (2, 1),
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                },
            }
        elif fmc == 5:
            self._feat_stride_fpn = [64, 32, 16, 8, 4]
            self.anchor_cfg = {}
            _ass = 2.0**(1.0 / 3)
            _basescale = 1.0
            for _stride in [4, 8, 16, 32, 64]:
                key = str(_stride)
                value = {
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'ALLOWED_BORDER': 9999
                }
                scales = []
                for _ in range(3):
                    scales.append(_basescale)
                    _basescale *= _ass
                value['SCALES'] = tuple(scales)
                self.anchor_cfg[key] = value

        print(self._feat_stride_fpn, self.anchor_cfg)
        self.use_landmarks = False
        if len(sym) // len(self._feat_stride_fpn) == 3:
            self.use_landmarks = True
        print('use_landmarks', self.use_landmarks)
        self.fpn_keys = []

        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s' % s)

        self._anchors_fpn = dict(
            zip(self.fpn_keys, generate_anchors_fpn(cfg=self.anchor_cfg)))
        for k in self._anchors_fpn:
            v = self._anchors_fpn[k].astype(np.float32)
            self._anchors_fpn[k] = v
        self.anchor_plane_cache = {}

        self._num_anchors = dict(
            zip(self.fpn_keys,
                [anchors.shape[0] for anchors in self._anchors_fpn.values()]))

    def detect(self, img, threshold=0.5, scale=1.0):
        proposals_list = []
        scores_list = []
        landmarks_list = []
        if scale == 1.0:
            im = img
        else:
            im = cv2.resize(img,
                            None,
                            None,
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_LINEAR)
        im_info = [im.shape[0], im.shape[1]]
        im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
        for i in range(3):
            im_tensor[0, i, :, :] = im[:, :, 2 - i]
        data = nd.array(im_tensor)
        db = mx.io.DataBatch(data=(data, ),
                             provide_data=[('data', data.shape)])
        self.model.forward(db, is_train=False)
        net_out = self.model.get_outputs()
        for _idx, s in enumerate(self._feat_stride_fpn):
            _key = 'stride%s' % s
            stride = int(s)
            if self.use_landmarks:
                idx = _idx * 3
            else:
                idx = _idx * 2
            scores = net_out[idx].asnumpy()
            scores = scores[:, self._num_anchors['stride%s' % s]:, :, :]
            idx += 1
            bbox_deltas = net_out[idx].asnumpy()

            height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]
            A = self._num_anchors['stride%s' % s]
            K = height * width
            key = (height, width, stride)
            if key in self.anchor_plane_cache:
                anchors = self.anchor_plane_cache[key]
            else:
                anchors_fpn = self._anchors_fpn['stride%s' % s]
                anchors = anchors_plane(height, width, stride, anchors_fpn)
                anchors = anchors.reshape((K * A, 4))
                if len(self.anchor_plane_cache) < 100:
                    self.anchor_plane_cache[key] = anchors

            scores = clip_pad(scores, (height, width))
            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

            bbox_deltas = clip_pad(bbox_deltas, (height, width))
            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
            bbox_pred_len = bbox_deltas.shape[3] // A
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))

            proposals = bbox_pred(anchors, bbox_deltas)
            #proposals = clip_boxes(proposals, im_info[:2])

            scores_ravel = scores.ravel()
            order = np.where(scores_ravel >= threshold)[0]
            proposals = proposals[order, :]
            scores = scores[order]

            proposals[:, 0:4] /= scale

            proposals_list.append(proposals)
            scores_list.append(scores)

            if self.use_landmarks:
                idx += 1
                landmark_deltas = net_out[idx].asnumpy()
                landmark_deltas = clip_pad(landmark_deltas, (height, width))
                landmark_pred_len = landmark_deltas.shape[1] // A
                landmark_deltas = landmark_deltas.transpose(
                    (0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len // 5))
                landmark_deltas *= self.landmark_std
                #print(landmark_deltas.shape, landmark_deltas)
                landmarks = landmark_pred(anchors, landmark_deltas)
                landmarks = landmarks[order, :]

                landmarks[:, :, 0:2] /= scale
                landmarks_list.append(landmarks)

        proposals = np.vstack(proposals_list)
        landmarks = None
        if proposals.shape[0] == 0:
            if self.use_landmarks:
                landmarks = np.zeros((0, 5, 2))
            return np.zeros((0, 5)), landmarks
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        proposals = proposals[order, :]
        scores = scores[order]
        if self.use_landmarks:
            landmarks = np.vstack(landmarks_list)
            landmarks = landmarks[order].astype(np.float32, copy=False)

        pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32,
                                                                copy=False)
        keep = self.nms(pre_det)
        det = np.hstack((pre_det, proposals[:, 4:]))
        det = det[keep, :]
        if self.use_landmarks:
            landmarks = landmarks[keep]

        return det, landmarks

    def nms(self, dets):
        thresh = self.nms_threshold
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


def get_retinaface(name, rac='net3', root='~/.insightface/models', **kwargs):
    from .model_store import get_model_file
    _file = get_model_file("retinaface_%s" % name, root=root)
    return FaceDetector(_file, rac)


def retinaface_r50_v1(**kwargs):
    return get_retinaface("r50_v1", rac='net3', **kwargs)


def retinaface_mnet025_v1(**kwargs):
    return get_retinaface("mnet025_v1", rac='net3', **kwargs)


def retinaface_mnet025_v2(**kwargs):
    return get_retinaface("mnet025_v2", rac='net3l', **kwargs)
