# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-09-18
# @Function      : 

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime
import os
import os.path as osp
import cv2

__all__ = ['RetinaFace']


def softmax(z: np.ndarray) -> np.ndarray:
    """Apply softmax activation along axis 1."""
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div


def distance2bbox(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Decode distance prediction to bounding box.
    
    Args:
        points: Anchor points with shape (n, 2), [x, y].
        distance: Distance from anchor to 4 boundaries (left, top, right, bottom).
        max_shape: Optional image shape for clipping.
    
    Returns:
        Decoded bboxes with shape (n, 4).
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Decode distance prediction to keypoints.
    
    Args:
        points: Anchor points with shape (n, 2).
        distance: Distance predictions for keypoints.
        max_shape: Optional image shape for clipping.
    
    Returns:
        Decoded keypoints.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class RetinaFace:
    """ONNX-based RetinaFace face detection model.
    
    This class provides face detection with optional keypoint detection.
    
    Attributes:
        model_file: Path to the ONNX model file.
        session: ONNX Runtime inference session.
        taskname: Task identifier ('detection').
        input_size: Expected input size (width, height) or None for dynamic.
        input_mean: Mean value for input normalization.
        input_std: Std value for input normalization.
        nms_thresh: NMS IoU threshold.
        det_thresh: Detection confidence threshold.
    """
    
    def __init__(
        self,
        model_file: Optional[str] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> None:
        """Initialize the RetinaFace model.
        
        Args:
            model_file: Path to the ONNX model file.
            session: Pre-existing ONNX Runtime session (optional).
        """
        import onnxruntime
        self.model_file = model_file
        self.session = session
        self.taskname = 'detection'
        
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        
        self.center_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self._init_vars()

    def _init_vars(self) -> None:
        """Initialize model variables from session."""
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        
        if isinstance(input_shape[2], str):
            self.input_size: Optional[Tuple[int, int]] = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        
        input_name = input_cfg.name
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = [o.name for o in outputs]
        self.input_name = input_name
        self.output_names = output_names
        self.input_mean = 127.5
        self.input_std = 128.0
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def prepare(self, ctx_id: int, **kwargs: Any) -> None:
        """Prepare the model for inference.
        
        Args:
            ctx_id: Context ID for GPU device. Use -1 for CPU.
            **kwargs: Additional arguments:
                - nms_thresh: NMS IoU threshold
                - det_thresh: Detection confidence threshold
                - input_size: Input size override
        """
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])
        
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        
        det_thresh = kwargs.get('det_thresh', None)
        if det_thresh is not None:
            self.det_thresh = det_thresh
        
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print('warning: det_size is already set in detection model, ignore')
            else:
                self.input_size = input_size

    def forward(
        self,
        img: np.ndarray,
        threshold: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Run forward pass on image.
        
        Args:
            img: Input image (BGR, uint8, 0-255).
            threshold: Detection threshold.
        
        Returns:
            Tuple of (scores_list, bboxes_list, kpss_list).
        """
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx + fmc * 2] * stride
            
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    ).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        
        return scores_list, bboxes_list, kpss_list

    def detect(
        self,
        img: np.ndarray,
        input_size: Optional[Tuple[int, int]] = None,
        max_num: int = 0,
        metric: str = 'default',
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Detect faces in an image.
        
        Args:
            img: Input image as numpy array.
                - Format: BGR (OpenCV default)
                - Dtype: uint8
                - Range: 0-255
                - Shape: (H, W, 3)
            input_size: Override input size for detection.
            max_num: Maximum number of faces to return. 0 means no limit.
            metric: Selection metric when max_num > 0.
                - 'default': Prefer faces closer to image center
                - 'max': Select largest faces
        
        Returns:
            Tuple of (bboxes, keypoints):
                - bboxes: np.ndarray with shape (N, 5), each row is [x1, y1, x2, y2, score]
                - keypoints: np.ndarray with shape (N, 5, 2) or None
        """
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
        
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        else:
            kpss = None
        
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0
            
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        
        return det, kpss

    def nms(self, dets: np.ndarray) -> List[int]:
        """Non-maximum suppression.
        
        Args:
            dets: Detection boxes with shape (N, 5), each row is [x1, y1, x2, y2, score].
        
        Returns:
            List of indices to keep.
        """
        thresh = self.nms_thresh
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


def get_retinaface(
    name: str,
    download: bool = False,
    root: str = '~/.insightface/models',
    **kwargs: Any,
) -> RetinaFace:
    """Get RetinaFace model by name.
    
    Args:
        name: Model name or path.
        download: Whether to download if not found.
        root: Root directory for model storage.
        **kwargs: Additional arguments.
    
    Returns:
        RetinaFace model instance.
    """
    if not download:
        assert os.path.exists(name)
        return RetinaFace(name)
    else:
        from .model_store import get_model_file
        _file = get_model_file("retinaface_%s" % name, root=root)
        return RetinaFace(_file)
