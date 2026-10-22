import numpy as np

from insightface.gui.core.quality import blur_score, score_face


def test_quality_range_and_blur():
    image = np.full((120, 120, 3), 128, dtype=np.uint8)
    image[40:80, 40:80] = 255
    assert blur_score(image) >= 0
    score, flags = score_face(image, [30, 30, 90, 90], det_score=0.9)
    assert 0.0 <= score <= 1.0
    assert isinstance(flags, list)
