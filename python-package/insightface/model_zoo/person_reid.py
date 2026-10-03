"""Person ReID using explicit image size and RGB normalization rules."""
from __future__ import annotations

import cv2
import numpy as np

from .person_package import (EMBEDDED_PREPROCESSING, create_session, model_fingerprint,
                             resolve_person_descriptor, validate_image)


class PersonReID:
    def __init__(self, descriptor, providers, threads=4):
        self.descriptor = descriptor = resolve_person_descriptor("person_reid", descriptor)
        self.model_id = model_fingerprint(descriptor)
        preprocessing = descriptor["preprocessing"]
        if preprocessing != EMBEDDED_PREPROCESSING:
            self._mean = np.asarray(preprocessing["mean"], np.float32)[None, :, None, None]
            # Keep the declared std precision until converting its reciprocal to FP32.
            self._inverse_std = np.asarray([1.0 / value for value in preprocessing["std"]], np.float32)[None, :, None, None]
            self._scale = preprocessing["scale"]
        self.session, self.diagnostics = create_session(descriptor, providers, threads)

    def _preprocess(self, crop):
        validate_image(crop)
        value = cv2.resize(crop, tuple(reversed(self.descriptor["input_size"])), interpolation=cv2.INTER_LINEAR)
        value = value.transpose(2, 0, 1)[None].astype(np.float32)
        value = value[:, [2, 1, 0]]
        if self.descriptor["preprocessing"] == EMBEDDED_PREPROCESSING:
            return np.ascontiguousarray(value, dtype=self.descriptor["input_dtype"])
        if self._scale != 1:
            value *= self._scale
        return np.ascontiguousarray((value - self._mean) * self._inverse_std)

    def embed(self, crop):
        value = self.session.run(["reid_embedding"], {"data": self._preprocess(crop)})[0]
        if value.shape != (1, 256) or not np.isfinite(value).all():
            raise RuntimeError("invalid Intel 0265 embedding shape or nonfinite values")
        feature = value[0].astype(np.float32, copy=True)
        norm = float(np.linalg.norm(feature))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError("zero or invalid Intel 0265 embedding")
        return feature / norm
