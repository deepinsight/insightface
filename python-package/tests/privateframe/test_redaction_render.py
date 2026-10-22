from __future__ import annotations

from copy import deepcopy

import cv2
import numpy as np
import pytest

from insightface.app.privateframe.artifact_render import _blur, _gaussian_blur
from insightface.app.privateframe.config import validate_redaction


def _redaction(*, algorithm: str = "exact", max_side: int = 64) -> dict:
    return {
        "method": "gaussian",
        "box_scale": 1.0,
        "gaussian": {
            "algorithm": algorithm,
            "max_side": max_side,
            "kernel_ratio": 1.0,
            "min_kernel": 5,
            "sigma": 0.0,
        },
        "mosaic": {
            "block_size_ratio": 0.12,
            "min_block_size": 8,
        },
        "feather": {"enabled": False},
    }


def test_exact_gaussian_matches_the_original_full_resolution_operation():
    frame = np.arange(40 * 48 * 3, dtype=np.uint8).reshape(40, 48, 3)
    expected = frame.copy()
    region = expected[5:30, 7:37]
    region[:] = cv2.GaussianBlur(region, (31, 31), sigmaX=0.0, sigmaY=0.0)

    settings = {"redaction": _redaction(algorithm="exact", max_side=1)}
    _blur(frame, [7.0, 5.0, 37.0, 30.0], settings)

    np.testing.assert_array_equal(frame, expected)


def test_missing_algorithm_keeps_old_results_on_exact_path(monkeypatch):
    calls = []
    original = cv2.GaussianBlur

    def capture(region, kernel, *, sigmaX, sigmaY):
        calls.append((region.shape, kernel, sigmaX, sigmaY))
        return original(region, kernel, sigmaX=sigmaX, sigmaY=sigmaY)

    monkeypatch.setattr(cv2, "GaussianBlur", capture)
    region = np.zeros((160, 120, 3), dtype=np.uint8)
    _gaussian_blur(
        region,
        {"kernel_ratio": 1.0, "min_kernel": 121, "sigma": 0.0},
    )

    assert calls == [((160, 120, 3), (161, 161), 0.0, 0.0)]


def test_pyramid_gaussian_scales_image_kernel_and_explicit_sigma(monkeypatch):
    calls = []
    original = cv2.GaussianBlur

    def capture(region, kernel, *, sigmaX, sigmaY):
        calls.append((region.shape, kernel, sigmaX, sigmaY))
        return original(region, kernel, sigmaX=sigmaX, sigmaY=sigmaY)

    monkeypatch.setattr(cv2, "GaussianBlur", capture)
    region = np.arange(200 * 100 * 3, dtype=np.uint8).reshape(200, 100, 3)
    result = _gaussian_blur(
        region,
        {
            "algorithm": "pyramid",
            "max_side": 50,
            "kernel_ratio": 1.0,
            "min_kernel": 121,
            "sigma": 20.0,
        },
    )

    assert result.shape == region.shape
    assert result.dtype == region.dtype
    assert calls == [((50, 25, 3), (51, 51), 5.0, 5.0)]


def test_pyramid_blur_changes_only_the_requested_roi():
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(180, 220, 3), dtype=np.uint8)
    original = frame.copy()
    settings = {"redaction": _redaction(algorithm="pyramid", max_side=32)}

    _blur(frame, [40.0, 30.0, 180.0, 160.0], settings)

    np.testing.assert_array_equal(frame[:30], original[:30])
    np.testing.assert_array_equal(frame[160:], original[160:])
    np.testing.assert_array_equal(frame[30:160, :40], original[30:160, :40])
    np.testing.assert_array_equal(frame[30:160, 180:], original[30:160, 180:])
    assert not np.array_equal(frame[30:160, 40:180], original[30:160, 40:180])


@pytest.mark.parametrize("algorithm", ["other", "", "Pyramid"])
def test_redaction_rejects_unknown_gaussian_algorithm(algorithm):
    settings = _redaction()
    settings["gaussian"]["algorithm"] = algorithm

    with pytest.raises(ValueError, match="algorithm must be exact or pyramid"):
        validate_redaction(settings)


@pytest.mark.parametrize("max_side", [0, -1, 1.5, True, "96"])
def test_redaction_rejects_invalid_pyramid_max_side(max_side):
    settings = _redaction()
    settings["gaussian"]["max_side"] = max_side

    with pytest.raises(ValueError, match="max_side must be a positive integer"):
        validate_redaction(settings)


def test_exact_ignores_pyramid_max_side():
    settings = _redaction(algorithm="exact", max_side=1)
    before = deepcopy(settings)

    validate_redaction(settings)

    assert settings == before
