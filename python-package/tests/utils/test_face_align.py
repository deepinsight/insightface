"""Tests for landmark-based face alignment."""
import warnings

import numpy as np
import pytest
from skimage.transform import SimilarityTransform

from insightface.utils import face_align

LMK = np.array(
    [[30, 50], [70, 48], [50, 70], [35, 90], [68, 89]], dtype=np.float32)

# Reference values recorded before the from_estimate migration.
EXPECTED_112 = np.array([
    [0.94786581, -0.03185195, 10.27467753],
    [0.03185195, 0.94786581, 4.50718242],
])
EXPECTED_128 = np.array([
    [0.94786581, -0.03185195, 18.27467753],
    [0.03185195, 0.94786581, 4.50718242],
])

requires_from_estimate = pytest.mark.skipif(
    not hasattr(SimilarityTransform, 'from_estimate'),
    reason='scikit-image < 0.26 has no from_estimate; the fallback keeps '
           'the previous non-raising behaviour')


@pytest.mark.parametrize('size, expected',
                         [(112, EXPECTED_112), (128, EXPECTED_128)])
def test_estimate_norm_matches_reference(size, expected):
    M = face_align.estimate_norm(LMK, image_size=size)
    np.testing.assert_allclose(M, expected, atol=1e-6)


def test_estimate_norm_emits_no_future_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        face_align.estimate_norm(LMK)


@requires_from_estimate
def test_estimate_norm_raises_on_degenerate_landmarks():
    degenerate = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(RuntimeError):
        face_align.estimate_norm(degenerate)
