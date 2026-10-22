from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_DIR = Path(__file__).resolve().parents[3]
PYTHON_PACKAGE = REPOSITORY_DIR / "python-package"
if str(PYTHON_PACKAGE) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE))

from insightface.model_zoo.scrfd import SCRFD  # noqa: E402


def test_multires_candidates_are_merged_before_one_global_nms() -> None:
    detector = object.__new__(SCRFD)
    detector.use_kps = True
    nms_inputs: list[np.ndarray] = []

    def candidates(
        _image: np.ndarray,
        size: tuple[int, int],
        _threshold: float,
    ):
        offset = 0.0 if size == (96, 96) else 1.0
        score = 0.90 if size == (96, 96) else 0.80
        boxes = np.asarray(
            [[10.0 + offset, 10.0 + offset, 50.0 + offset, 50.0 + offset, score]],
            dtype=np.float32,
        )
        landmarks = np.full((1, 5, 2), offset, dtype=np.float32)
        return boxes, landmarks

    def global_nms(merged: np.ndarray) -> list[int]:
        nms_inputs.append(merged.copy())
        return [0]

    detector._detect_candidates = candidates
    detector.nms = global_nms

    boxes, landmarks = detector.detect(
        np.zeros((128, 128, 3), dtype=np.uint8),
        input_size=[(96, 96), (512, 512)],
    )

    assert len(nms_inputs) == 1
    assert nms_inputs[0].shape == (2, 5)
    assert nms_inputs[0][:, 4].tolist() == pytest.approx([0.90, 0.80])
    assert boxes.shape == (1, 5)
    assert landmarks.shape == (1, 5, 2)
