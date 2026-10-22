"""Explicit ONNX face/non-face verifier adapter for manifest packages."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from .package_manifest import (
    EMBEDDED_PREPROCESSING,
    normalize_preprocessing,
)


class FaceVerifier:
    """Score centered, zero-padded face proposal crops in one batch."""

    taskname = "verification"

    def __init__(
        self,
        model_file=None,
        session=None,
        *,
        expansion=1.3,
        preprocessing=None,
    ):
        if model_file is None:
            raise ValueError("face verifier model_file is required")
        if session is None:
            raise ValueError("face verifier requires an explicit inference session")
        self.model_file = str(model_file)
        self.session = session
        self.preprocessing = normalize_preprocessing(
            preprocessing,
            "face verifier preprocessing",
        )
        self.crop_expansion = float(expansion)
        if not math.isfinite(self.crop_expansion) or self.crop_expansion <= 0.0:
            raise ValueError("face verifier expansion must be finite and positive")

        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError(
                f"face verifier requires one input, received {len(inputs)}"
            )
        input_meta = inputs[0]
        self.input_name = str(input_meta.name)
        self.input_shape = list(input_meta.shape)
        self.input_type = str(input_meta.type)
        if (
            self.input_type not in {"tensor(uint8)", "tensor(float)"}
            or len(self.input_shape) != 4
            or self.input_shape[1] != 3
            or not isinstance(self.input_shape[2], int)
            or not isinstance(self.input_shape[3], int)
            or self.input_shape[2] != self.input_shape[3]
            or isinstance(self.input_shape[0], int)
        ):
            raise RuntimeError(
                "face verifier input must be dynamic-batch uint8 or float32 "
                "NCHW with a fixed square size: "
                f"type={self.input_type}, shape={self.input_shape}"
            )
        self.input_dtype = (
            np.uint8
            if self.input_type == "tensor(uint8)"
            else np.float32
        )
        if (
            isinstance(self.preprocessing, Mapping)
            and self.input_dtype is not np.float32
        ):
            raise RuntimeError(
                "face verifier mean/std preprocessing requires a float32 input "
                "(tensor(float))"
            )
        self.input_size = int(self.input_shape[2])
        if self.input_size <= 0:
            raise RuntimeError(
                f"face verifier input size must be positive: {self.input_shape}"
            )
        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            raise RuntimeError(
                f"face verifier requires one output, received {len(outputs)}"
            )
        output_meta = outputs[0]
        self.output_name = str(output_meta.name)
        self.output_shape = list(output_meta.shape)
        if (
            str(output_meta.type) != "tensor(float)"
            or len(self.output_shape) != 1
            or isinstance(self.output_shape[0], int)
        ):
            raise RuntimeError(
                "face verifier output must be one float probability per sample: "
                f"type={output_meta.type}, shape={self.output_shape}"
            )

    def prepare(self, ctx_id, **kwargs):
        del kwargs
        if ctx_id < 0:
            self.session.set_providers(["CPUExecutionProvider"])

    @staticmethod
    def _crop(frame: np.ndarray, box: Sequence[float], size: int, expansion: float) -> np.ndarray:
        if size <= 0:
            raise ValueError("face verifier crop size must be positive")
        if not math.isfinite(expansion) or expansion <= 0.0:
            raise ValueError("face verifier crop expansion must be finite and positive")
        target = np.asarray(box, dtype=np.float64).reshape(4)
        if not np.all(np.isfinite(target)):
            raise ValueError("face verifier box must be finite")
        width = float(target[2] - target[0])
        height = float(target[3] - target[1])
        if width <= 0.0 or height <= 0.0:
            raise ValueError("face verifier box must have positive area")
        center = (target[:2] + target[2:]) * 0.5
        side = max(2.0, max(width, height) * expansion)
        left = float(center[0] - side * 0.5)
        top = float(center[1] - side * 0.5)
        scale = size / side
        matrix = np.asarray(
            [
                [scale, 0.0, -left * scale],
                [0.0, scale, -top * scale],
            ],
            dtype=np.float32,
        )
        interpolation = cv2.INTER_AREA if side >= size else cv2.INTER_LINEAR
        return cv2.warpAffine(
            frame,
            matrix,
            (size, size),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def verify(self, frame, boxes):
        if len(boxes) == 0:
            return []
        crops = [
            self._crop(
                np.asarray(frame),
                box,
                self.input_size,
                self.crop_expansion,
            )
            for box in boxes
        ]
        batch = self.preprocess(crops)
        scores = np.asarray(
            self.session.run(
                [self.output_name],
                {self.input_name: batch},
            )[0],
            dtype=np.float64,
        ).reshape(len(boxes))
        return [{"face_probability": float(score)} for score in scores]

    def preprocess(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Convert BGR crops to the declared RGB NCHW input contract."""

        if len(crops) == 0:
            return np.empty(
                (0, 3, self.input_size, self.input_size),
                dtype=self.input_dtype,
            )
        expected = (self.input_size, self.input_size, 3)
        tensors = []
        for crop in crops:
            value = np.asarray(crop)
            if value.shape != expected:
                raise ValueError(
                    f"face verifier crop must have shape {expected}, "
                    f"received {value.shape}"
                )
            tensors.append(
                np.ascontiguousarray(value[:, :, ::-1].transpose(2, 0, 1))
            )
        batch = np.ascontiguousarray(np.stack(tensors), dtype=np.uint8)
        if self.preprocessing == EMBEDDED_PREPROCESSING:
            return np.ascontiguousarray(batch, dtype=self.input_dtype)
        mean = float(self.preprocessing["mean"])
        std = float(self.preprocessing["std"])
        return np.ascontiguousarray(
            (batch.astype(np.float32) - mean) / std,
            dtype=np.float32,
        )

    score_boxes = verify

    def get(self, image, face):
        values = self.verify(image, [face.bbox])
        score = float(values[0]["face_probability"])
        face.verification_score = score
        return score


__all__ = ["FaceVerifier"]
