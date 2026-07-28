from __future__ import print_function
import sys
import os
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import cv2
from rcnn.logger import logger
from rcnn.processing.bbox_transform import clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper, cpu_nms_wrapper


class RetinaFace:
    def __init__(self,
                 prefix,
                 epoch,
                 ctx_id=0,
                 network='net3',
                 nms=0.4,
                 nocrop=False,
                 decay4=0.5,
                 vote=False):
        self.ctx_id = ctx_id
        self.network = network
        self.decay4 = decay4
        self.nms_threshold = nms
        self.vote = vote
        self.nocrop = nocrop
        self.fpn_keys = []
        self.anchor_cfg = None
        
        # Initialize preprocessing parameters
        pixel_means = [0.0, 0.0, 0.0]
        pixel_stds = [1.0, 1.0, 1.0]
        pixel_scale = 1.0
        self.preprocess = False
        _ratio = (1.,)
        fmc = 3
        
        # Network-specific configurations
        if network == 'ssh' or network == 'vgg':
            pixel_means = [103.939, 116.779, 123.68]
            self.preprocess = True
        elif network == 'net3':
            _ratio = (1.,)
        elif network == 'net3a':
            _ratio = (1., 1.5)
        elif network == 'net6':
            fmc = 6
        elif network == 'net5':
            fmc = 5
        elif network == 'net5a':
            fmc = 5
            _ratio = (1., 1.5)
        elif network == 'net4':
            fmc = 4
        elif network == 'net4a':
            fmc = 4
            _ratio = (1., 1.5)
        elif network in ['x5', 'x3', 'x3a']:
            fmc = 5 if network == 'x5' else 3
            pixel_means = [103.52, 116.28, 123.675]
            pixel_stds = [57.375, 57.12, 58.395]
            if network == 'x3a':
                _ratio = (1., 1.5)
        else:
            assert False, 'network setting error %s' % network

        # Configure FPN strides and anchors
        self._configure_fpn(fmc, _ratio)
        
        print(self._feat_stride_fpn, self.anchor_cfg)

        # Generate anchors
        self.fpn_keys = ['stride%s' % s for s in self._feat_stride_fpn]
        dense_anchor = False
        self._anchors_fpn = dict(
            zip(self.fpn_keys,
                generate_anchors_fpn(dense_anchor=dense_anchor, cfg=self.anchor_cfg)))
        
        # Convert to float32 and cache num_anchors
        for k in self._anchors_fpn:
            self._anchors_fpn[k] = self._anchors_fpn[k].astype(np.float32)
        
        self._num_anchors = {k: anchors.shape[0] 
                            for k, anchors in zip(self.fpn_keys, self._anchors_fpn.values())}
        
        # Load model
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        
        # Setup context and NMS
        if self.ctx_id >= 0:
            self.ctx = mx.gpu(self.ctx_id)
            self.nms = gpu_nms_wrapper(self.nms_threshold, self.ctx_id)
        else:
            self.ctx = mx.cpu()
            self.nms = cpu_nms_wrapper(self.nms_threshold)
        
        # Cache preprocessing parameters as numpy arrays
        self.pixel_means = np.array(pixel_means, dtype=np.float32)
        self.pixel_stds = np.array(pixel_stds, dtype=np.float32)
        self.pixel_scale = float(pixel_scale)
        
        # Precompute reversed pixel parameters for optimization
        self.pixel_means_reversed = self.pixel_means[::-1]
        self.pixel_stds_reversed = self.pixel_stds[::-1]
        
        print('means', self.pixel_means)
        
        # Detect landmarks and cascade usage
        self.use_landmarks = len(sym) // len(self._feat_stride_fpn) >= 3
        print('use_landmarks', self.use_landmarks)
        
        self.cascade = 1 if float(len(sym)) // len(self._feat_stride_fpn) > 3.0 else 0
        print('cascade', self.cascade)
        
        # Bbox and landmark standards
        self.bbox_stds = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.landmark_std = 1.0
        
        print('sym size:', len(sym))
        
        # Initialize model
        image_size = (640, 640)
        self.model = mx.mod.Module(symbol=sym, context=self.ctx, label_names=None)
        self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))],
                       for_training=False)
        self.model.set_params(arg_params, aux_params)

    def _configure_fpn(self, fmc, _ratio):
        """Configure FPN strides and anchor configurations"""
        if fmc == 3:
            self._feat_stride_fpn = [32, 16, 8]
            self.anchor_cfg = {
                '32': {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '16': {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '8': {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
            }
        elif fmc == 4:
            self._feat_stride_fpn = [32, 16, 8, 4]
            self.anchor_cfg = {
                '32': {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '16': {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '8': {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '4': {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
            }
        elif fmc == 6:
            self._feat_stride_fpn = [128, 64, 32, 16, 8, 4]
            self.anchor_cfg = {
                '128': {'SCALES': (32,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '64': {'SCALES': (16,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '32': {'SCALES': (8,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '16': {'SCALES': (4,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '8': {'SCALES': (2,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '4': {'SCALES': (1,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
            }
        elif fmc == 5:
            self._feat_stride_fpn = [64, 32, 16, 8, 4]
            self.anchor_cfg = {}
            _ass = 2.0**(1.0 / 3)
            _basescale = 1.0
            for _stride in [4, 8, 16, 32, 64]:
                scales = [_basescale * (_ass ** i) for i in range(3)]
                self.anchor_cfg[str(_stride)] = {
                    'BASE_SIZE': 16,
                    'RATIOS': _ratio,
                    'SCALES': tuple(scales),
                    'ALLOWED_BORDER': 9999
                }
                _basescale = scales[-1] * _ass

    def get_input(self, img):
        """Optimized preprocessing - single pass through channels"""
        im = img.astype(np.float32)
        im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
        
        # Vectorized channel processing
        for i in range(3):
            im_tensor[0, i] = (im[:, :, 2 - i] / self.pixel_scale - 
                              self.pixel_means_reversed[i]) / self.pixel_stds_reversed[i]
        
        return nd.array(im_tensor)

    def detect(self, img, threshold=0.5, scales=[1.0], do_flip=False):
        """Optimized detection with reduced redundancy"""
        proposals_list = []
        scores_list = []
        landmarks_list = []
        strides_list = []
        
        flips = [0, 1] if do_flip else [0]
        imgs = img if isinstance(img, list) else [img]
        
        # Cache frequently used values
        use_landmarks = self.use_landmarks
        num_anchors = self._num_anchors
        bbox_stds = self.bbox_stds
        
        for img in imgs:
            for im_scale in scales:
                for flip in flips:
                    # Resize image
                    if im_scale != 1.0:
                        im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale,
                                      interpolation=cv2.INTER_LINEAR)
                    else:
                        im = img.copy()
                    
                    if flip:
                        im = im[:, ::-1, :]
                    
                    # Handle nocrop padding
                    if self.nocrop:
                        h = ((im.shape[0] + 31) // 32) * 32
                        w = ((im.shape[1] + 31) // 32) * 32
                        if h != im.shape[0] or w != im.shape[1]:
                            _im = np.zeros((h, w, 3), dtype=np.float32)
                            _im[:im.shape[0], :im.shape[1], :] = im
                            im = _im
                    else:
                        im = im.astype(np.float32)
                    
                    # Preprocess image (optimized)
                    im_info = [im.shape[0], im.shape[1]]
                    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
                    
                    # Vectorized preprocessing
                    for i in range(3):
                        im_tensor[0, i] = (im[:, :, 2 - i] / self.pixel_scale - 
                                          self.pixel_means_reversed[i]) / self.pixel_stds_reversed[i]
                    
                    data = nd.array(im_tensor)
                    db = mx.io.DataBatch(data=(data,), provide_data=[('data', data.shape)])
                    
                    # Forward pass
                    self.model.forward(db, is_train=False)
                    net_out = self.model.get_outputs()
                    
                    sym_idx = 0
                    
                    for _idx, s in enumerate(self._feat_stride_fpn):
                        _key = 'stride%s' % s
                        stride = int(s)
                        
                        # Get scores and bbox deltas
                        scores = net_out[sym_idx].asnumpy()
                        scores = scores[:, num_anchors[_key]:, :, :]
                        bbox_deltas = net_out[sym_idx + 1].asnumpy()
                        
                        height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]
                        A = num_anchors[_key]
                        K = height * width
                        
                        # Generate anchors
                        anchors_fpn = self._anchors_fpn[_key]
                        anchors = anchors_plane(height, width, stride, anchors_fpn).reshape((K * A, 4))
                        
                        # Process scores
                        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
                        
                        # Process bbox deltas
                        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
                        bbox_pred_len = bbox_deltas.shape[3] // A
                        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
                        
                        # Apply bbox standards (vectorized)
                        bbox_deltas[:, 0::4] *= bbox_stds[0]
                        bbox_deltas[:, 1::4] *= bbox_stds[1]
                        bbox_deltas[:, 2::4] *= bbox_stds[2]
                        bbox_deltas[:, 3::4] *= bbox_stds[3]
                        
                        proposals = self.bbox_pred(anchors, bbox_deltas)
                        
                        # Handle cascade
                        if self.cascade:
                            cascade_sym_num = 0
                            cls_cascade = False
                            bbox_cascade = False
                            __idx = [2, 3] if not use_landmarks else [3, 4]
                            
                            for diff_idx in __idx:
                                if sym_idx + diff_idx >= len(net_out):
                                    break
                                body = net_out[sym_idx + diff_idx].asnumpy()
                                
                                if body.shape[1] // A == 2:  # cls branch
                                    if not cls_cascade and not bbox_cascade:
                                        cascade_scores = body[:, num_anchors[_key]:, :, :]
                                        scores = cascade_scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
                                        cascade_sym_num += 1
                                        cls_cascade = True
                                    else:
                                        break
                                        
                                elif body.shape[1] // A == 4:  # bbox branch
                                    cascade_deltas = body.transpose((0, 2, 3, 1)).reshape((-1, bbox_pred_len))
                                    cascade_deltas[:, 0::4] *= bbox_stds[0]
                                    cascade_deltas[:, 1::4] *= bbox_stds[1]
                                    cascade_deltas[:, 2::4] *= bbox_stds[2]
                                    cascade_deltas[:, 3::4] *= bbox_stds[3]
                                    proposals = self.bbox_pred(proposals, cascade_deltas)
                                    cascade_sym_num += 1
                                    bbox_cascade = True
                        
                        proposals = clip_boxes(proposals, im_info[:2])
                        
                        # Apply decay for stride 4
                        if stride == 4 and self.decay4 < 1.0:
                            scores *= self.decay4
                        
                        # Filter by threshold
                        scores_ravel = scores.ravel()
                        order = np.where(scores_ravel >= threshold)[0]
                        proposals = proposals[order]
                        scores = scores[order]
                        
                        # Handle flipping
                        if flip:
                            proposals[:, [0, 2]] = im.shape[1] - proposals[:, [2, 0]] - 1
                        
                        proposals[:, :4] /= im_scale
                        
                        proposals_list.append(proposals)
                        scores_list.append(scores)
                        
                        if self.nms_threshold < 0.0:
                            strides_list.append(np.full(scores.shape, stride, dtype=np.float32))
                        
                        # Handle landmarks
                        if not self.vote and use_landmarks:
                            landmark_deltas = net_out[sym_idx + 2].asnumpy()
                            landmark_pred_len = landmark_deltas.shape[1] // A
                            landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape(
                                (-1, 5, landmark_pred_len // 5))
                            landmark_deltas *= self.landmark_std
                            
                            landmarks = self.landmark_pred(anchors, landmark_deltas)
                            landmarks = landmarks[order]
                            
                            if flip:
                                landmarks[:, :, 0] = im.shape[1] - landmarks[:, :, 0] - 1
                                landmarks = landmarks[:, [1, 0, 2, 4, 3], :]
                            
                            landmarks[:, :, :2] /= im_scale
                            landmarks_list.append(landmarks)
                        
                        # Update symbol index
                        sym_idx += 3 if use_landmarks else 2
                        if self.cascade:
                            sym_idx += cascade_sym_num
        
        # Combine all proposals
        if not proposals_list:
            landmarks = np.zeros((0, 5, 2)) if use_landmarks else None
            if self.nms_threshold < 0.0:
                return np.zeros((0, 6)), landmarks
            else:
                return np.zeros((0, 5)), landmarks
        
        proposals = np.vstack(proposals_list)
        scores = np.vstack(scores_list)
        
        # Sort by score
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        proposals = proposals[order]
        scores = scores[order]
        
        if self.nms_threshold < 0.0:
            strides = np.vstack(strides_list)[order]
        
        if not self.vote and use_landmarks:
            landmarks = np.vstack(landmarks_list)[order].astype(np.float32, copy=False)
        else:
            landmarks = None
        
        # Apply NMS
        if self.nms_threshold > 0.0:
            pre_det = np.hstack((proposals[:, :4], scores)).astype(np.float32, copy=False)
            
            if not self.vote:
                keep = self.nms(pre_det)
                det = np.hstack((pre_det, proposals[:, 4:]))[keep]
                if use_landmarks:
                    landmarks = landmarks[keep]
            else:
                det = np.hstack((pre_det, proposals[:, 4:]))
                det = self.bbox_vote(det)
        elif self.nms_threshold < 0.0:
            det = np.hstack((proposals[:, :4], scores, strides)).astype(np.float32, copy=False)
        else:
            det = np.hstack((proposals[:, :4], scores)).astype(np.float32, copy=False)
        
        return det, landmarks

    def detect_center(self, img, threshold=0.5, scales=[1.0], do_flip=False):
        """Detect face closest to center"""
        det, landmarks = self.detect(img, threshold, scales, do_flip)
        if det.shape[0] == 0:
            return None, None
        
        if det.shape[0] > 1:
            img_size = np.asarray(img.shape[:2])
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img_size / 2
            
            # Vectorized center distance calculation
            box_centers = np.column_stack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(box_centers ** 2, axis=1)
            bindex = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
        else:
            bindex = 0
        
        return det[bindex], landmarks[bindex]

    @staticmethod
    def check_large_pose(landmark, bbox):
        """Check for large pose angles"""
        assert landmark.shape == (5, 2)
        assert len(bbox) == 4

        def get_theta(base, x, y):
            vx = x - base
            vy = y - base
            vx[1] *= -1
            vy[1] *= -1
            tx = np.arctan2(vx[1], vx[0])
            ty = np.arctan2(vy[1], vy[0])
            d = np.degrees(ty - tx)
            
            if d < -180.0:
                d += 360.
            elif d > 180.0:
                d -= 360.0
            return d

        landmark = landmark.astype(np.float32)

        theta1 = get_theta(landmark[0], landmark[3], landmark[2])
        theta2 = get_theta(landmark[1], landmark[2], landmark[4])
        theta3 = get_theta(landmark[0], landmark[2], landmark[1])
        theta4 = get_theta(landmark[1], landmark[0], landmark[2])
        theta5 = get_theta(landmark[3], landmark[4], landmark[2])
        theta6 = get_theta(landmark[4], landmark[2], landmark[3])
        theta7 = get_theta(landmark[3], landmark[2], landmark[0])
        theta8 = get_theta(landmark[4], landmark[1], landmark[2])

        left_score = 10.0 if theta1 <= 0.0 else (theta2 / theta1 if theta2 > 0.0 else 0.0)
        right_score = 10.0 if theta2 <= 0.0 else (theta1 / theta2 if theta1 > 0.0 else 0.0)
        up_score = 10.0 if (theta3 <= 10.0 or theta4 <= 10.0) else max(theta1 / theta3, theta2 / theta4)
        down_score = 10.0 if (theta5 <= 10.0 or theta6 <= 10.0) else max(theta7 / theta5, theta8 / theta6)
        
        mleft = (landmark[0, 0] + landmark[3, 0]) / 2
        mright = (landmark[1, 0] + landmark[4, 0]) / 2
        box_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        ret = 0
        if left_score >= 3.0:
            ret = 1
        elif left_score >= 2.0 and mright <= box_center[0]:
            ret = 1
        elif right_score >= 3.0:
            ret = 2
        elif right_score >= 2.0 and mleft >= box_center[0]:
            ret = 2
        elif up_score >= 2.0:
            ret = 3
        elif down_score >= 5.0:
            ret = 4
        
        return ret, left_score, right_score, up_score, down_score

    @staticmethod
    def bbox_pred(boxes, box_deltas):
        """Optimized bbox prediction with vectorized operations"""
        if boxes.shape[0] == 0:
            return np.zeros((0, box_deltas.shape[1]), dtype=np.float32)

        boxes = boxes.astype(np.float32, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

        # Vectorized delta application
        pred_ctr_x = box_deltas[:, 0:1] * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = box_deltas[:, 1:2] * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(box_deltas[:, 2:3]) * widths[:, np.newaxis]
        pred_h = np.exp(box_deltas[:, 3:4]) * heights[:, np.newaxis]

        pred_boxes = np.zeros(box_deltas.shape, dtype=np.float32)
        pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
        pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
        pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
        pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

        if box_deltas.shape[1] > 4:
            pred_boxes[:, 4:] = box_deltas[:, 4:]

        return pred_boxes

    @staticmethod
    def landmark_pred(boxes, landmark_deltas):
        """Optimized landmark prediction"""
        if boxes.shape[0] == 0:
            return np.zeros((0, landmark_deltas.shape[1], landmark_deltas.shape[2]), dtype=np.float32)
        
        boxes = boxes.astype(np.float32, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
        
        pred = landmark_deltas.copy()
        pred[:, :, 0] = landmark_deltas[:, :, 0] * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred[:, :, 1] = landmark_deltas[:, :, 1] * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        
        return pred

    def bbox_vote(self, det):
        """Optimized bbox voting with vectorized operations"""
        if det.shape[0] == 0:
            return np.zeros((0, 5), dtype=np.float32)
        
        dets = []
        
        while det.shape[0] > 0 and len(dets) < 750:
            # Vectorized IOU calculation
            area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
            xx1 = np.maximum(det[0, 0], det[:, 0])
            yy1 = np.maximum(det[0, 1], det[:, 1])
            xx2 = np.minimum(det[0, 2], det[:, 2])
            yy2 = np.minimum(det[0, 3], det[:, 3])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            o = inter / (area[0] + area - inter)

            # NMS merge
            merge_index = np.where(o >= self.nms_threshold)[0]
            det_accu = det[merge_index]
            det = np.delete(det, merge_index, 0)
            
            if merge_index.shape[0] <= 1:
                if det.shape[0] == 0:
                    dets.append(det_accu)
                continue
            
            # Weighted average
            det_accu[:, :4] *= det_accu[:, 4:5]
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5), dtype=np.float32)
            det_accu_sum[0, :4] = np.sum(det_accu[:, :4], axis=0) / np.sum(det_accu[:, 4])
            det_accu_sum[0, 4] = max_score
            dets.append(det_accu_sum)
        
        return np.vstack(dets[:750]) if dets else np.zeros((0, 5), dtype=np.float32)
