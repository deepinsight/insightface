from __future__ import annotations

import pytest

from scripts.smoke_test import (
    inference_report,
    multipart_image,
    redacted_report,
    require_single_detected_face,
)


def test_multipart_image_uses_a_fixed_filename_and_preserves_bytes(tmp_path) -> None:
    image = tmp_path / 'unsafe"\r\nname.jpg'
    image.write_bytes(b"\xff\xd8release-smoke\xff\xd9")

    body, content_type = multipart_image(image, fields={"max_faces": "2"})

    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="max_faces"' in body
    assert b"\r\n\r\n2\r\n" in body
    assert b'filename="release-smoke-image"' in body
    assert image.name.encode() not in body
    assert image.read_bytes() in body


def test_inference_report_requires_a_normalized_finite_embedding() -> None:
    report = inference_report(
        {
            "faces": [
                {
                    "bbox": {"pixels": {"x": 1, "y": 2, "width": 3, "height": 4}},
                    "detection_score": 0.9,
                    "embedding": [0.6, 0.8],
                }
            ]
        }
    )

    assert report["embedding_dimension"] == 2
    assert report["embedding_norm"] == pytest.approx(1.0)
    assert report["face_count"] == 1

    with pytest.raises(SystemExit, match="not normalized"):
        inference_report(
            {
                "faces": [
                    {
                        "bbox": {},
                        "detection_score": 0.9,
                        "embedding": [1.0, 1.0],
                    }
                ]
            }
        )


def test_release_image_must_detect_exactly_one_face() -> None:
    require_single_detected_face({"faces": [{"bbox": {}}]})

    with pytest.raises(SystemExit, match="found 0"):
        require_single_detected_face({"faces": []})

    with pytest.raises(SystemExit, match="found 2"):
        require_single_detected_face({"faces": [{"bbox": {}}, {"bbox": {}}]})

    with pytest.raises(SystemExit, match="exactly one face"):
        inference_report(
            {
                "faces": [
                    {
                        "bbox": {},
                        "detection_score": 0.9,
                        "embedding": [0.6, 0.8],
                    },
                    {
                        "bbox": {},
                        "detection_score": 0.8,
                        "embedding": [0.6, 0.8],
                    },
                ]
            }
        )


def test_redacted_report_removes_biometric_material() -> None:
    report = {
        "health": {"status": "ready"},
        "inference": {
            "face_count": 1,
            "bbox": {"pixels": {"x": 1, "y": 2, "width": 3, "height": 4}},
            "detection_score": 0.9,
            "embedding_dimension": 2,
            "embedding_norm": 1.0,
            "embedding_sha256": "stable-feature-fingerprint",
            "embedding": [0.6, 0.8],
        },
    }

    redacted = redacted_report(report)

    assert redacted == {
        "health": {"status": "ready"},
        "inference": {
            "face_count": 1,
            "embedding_dimension": 2,
            "embedding_norm": 1.0,
        },
    }
