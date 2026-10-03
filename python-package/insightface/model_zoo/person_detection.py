"""Explicit PicoDet/PP-YOLOE pedestrian contracts; one Python NMS per image.

The package declares basic input rules. Both supported exports decode
boxes into original-image coordinates before Python NMS.
"""
from __future__ import annotations

import cv2
import numpy as np

from .person_package import (EMBEDDED_PREPROCESSING, create_session, model_fingerprint,
                             resolve_person_descriptor, validate_image)

NMS = {"score_threshold": .5, "iou_threshold": .6, "max_detections": 100}


def nms_xyxy(boxes, scores, iou_threshold, max_detections):
    """Stable continuous-coordinate greedy NMS; no inclusive-pixel +1."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    area = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    while order.size and len(keep) < max_detections:
        selected = int(order[0])
        keep.append(selected)
        remaining = order[1:]
        lt = np.maximum(boxes[selected, :2], boxes[remaining, :2])
        rb = np.minimum(boxes[selected, 2:], boxes[remaining, 2:])
        overlap = np.prod(np.maximum(rb - lt, 0), axis=1)
        union = area[selected] + area[remaining] - overlap
        iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
        order = remaining[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


class PersonDetection:
    def __init__(self, descriptor, providers, threads=4):
        self.descriptor = descriptor = resolve_person_descriptor("person_detection", descriptor)
        self.model_id = model_fingerprint(descriptor)
        self.session, self.diagnostics = create_session(descriptor, providers, threads)
        self.diagnostics.update(input_size=list(descriptor["input_size"]),
                                nms=dict(NMS))

    def _preprocess(self, bgr):
        validate_image(bgr)
        height, width = self.descriptor["input_size"]
        sy, sx = height / float(bgr.shape[0]), width / float(bgr.shape[1])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Match the supplied, Paddle-validated fx/fy resize path exactly.
        value = cv2.resize(rgb, None, fx=sx, fy=sy, interpolation=cv2.INTER_CUBIC).astype(np.float32)
        preprocessing = self.descriptor["preprocessing"]
        if preprocessing != EMBEDDED_PREPROCESSING:
            value *= preprocessing["scale"]
            value -= np.asarray(preprocessing["mean"])[None, None, :]
            value /= np.asarray(preprocessing["std"])[None, None, :]
        else:
            value = value.astype(self.descriptor["input_dtype"], copy=False)
        return {"image": np.ascontiguousarray(value.transpose(2, 0, 1)[None]),
                "scale_factor": np.asarray([[sy, sx]], np.float32)}

    def detect(self, bgr):
        return self.detect_candidates(bgr, NMS["score_threshold"])

    def detect_candidates(self, bgr, min_score):
        """One inference/NMS pass with an explicit candidate score threshold."""
        if not np.isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("min_score must be between zero and one")
        feed = self._preprocess(bgr)
        boxes, scores = self.session.run(["nms_pre_boxes", "nms_pre_scores"], feed)
        if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[2] != 4 or scores.shape != (1, 1, boxes.shape[1]):
            raise RuntimeError("unexpected person detector candidate output shape")
        boxes, scores = boxes[0], scores[0, 0]
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise RuntimeError("nonfinite or invalid person detection output")
        accepted = (scores >= min_score) & np.all(boxes[:, 2:] > boxes[:, :2], axis=1)
        boxes, scores = boxes[accepted], scores[accepted]
        keep = nms_xyxy(boxes, scores, NMS["iou_threshold"], NMS["max_detections"])
        boxes, scores = boxes[keep].copy(), scores[keep]
        # Coordinates have already been restored to original pixels by ONNX.
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, bgr.shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, bgr.shape[0])
        valid = np.all(boxes[:, 2:] > boxes[:, :2], axis=1)
        return np.column_stack((boxes[valid], scores[valid])).astype(np.float32, copy=False)
