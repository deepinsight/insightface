from pathlib import Path

import numpy as np

from insightface.gui.core.evaluation import (
    MULTI_FACE_REQUIRE_ONE,
    MULTI_FACE_SKIP,
    MULTI_FACE_USE_CENTERED_LARGEST,
    MULTI_FACE_USE_LARGEST,
    _select_face_from_faces,
    _tar_far_from_scores,
    run_identity_identification_evaluation,
    run_identity_verification_evaluation,
    validate_enterprise_dataset,
)
from insightface.gui.core.models import FaceRecord


class FakeEngine:
    model_name = "fake"
    requested_providers = ["CPU"]

    def _record(self, source_path=None, bbox=None):
        path = str(source_path or "")
        if "Alice" in path:
            embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif "Bob" in path:
            embedding = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            embedding = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return FaceRecord(
            bbox=bbox or [0, 0, 10, 10],
            kps=None,
            det_score=1.0,
            embedding=embedding,
            normed_embedding=embedding,
            source_path=path,
        )

    def detect_faces(self, image, source_path=None):
        del image
        path = str(source_path or "")
        if "noface" in path:
            return []
        if "multi" in path:
            return [
                self._record(source_path, [0, 0, 10, 10]),
                self._record(source_path, [0, 0, 20, 20]),
            ]
        return [self._record(source_path)]

    def detect_best_face(self, image, source_path=None):
        faces = self.detect_faces(image, source_path=source_path)
        return max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])) if faces else None


def _write_image(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(128, 128, 128)).save(path)


def test_identity_verification_auto_split_generates_probe_vs_gallery_pairs(tmp_path):
    root = tmp_path / "dataset"
    for rel in [
        "0001__Alice/gallery.jpg",
        "0001__Alice/probe.jpg",
        "0002__Bob/gallery.jpg",
        "0002__Bob/probe.jpg",
    ]:
        _write_image(root / rel)

    result = run_identity_verification_evaluation(root, FakeEngine(), auto_split=True)

    assert result.scenario.startswith("Enterprise 1:1")
    assert result.metrics["total_pairs"] == 4
    assert result.metrics["positive_pairs"] == 2
    assert result.metrics["negative_pairs"] == 2
    assert result.metrics["best_accuracy"] == 1.0
    assert "TAR@FAR=1e-6" not in result.metrics
    assert "Threshold@FAR=1e-3" not in result.metrics


def test_identity_identification_structured_dataset_reports_top1_and_far(tmp_path):
    root = tmp_path / "dataset_1n"
    for rel in [
        "gallery/0001__Alice/enroll_001.jpg",
        "gallery/0002__Bob/enroll_001.jpg",
        "probe/0001__Alice/test_001.jpg",
        "probe/0002__Bob/test_001.jpg",
        "unknown/unknown_001.jpg",
    ]:
        _write_image(root / rel)

    result = run_identity_identification_evaluation(root, FakeEngine(), auto_split=False)

    assert result.scenario.startswith("Enterprise 1:N")
    assert result.metrics["gallery_identities"] == 2
    assert result.metrics["Top1"] == 1.0
    assert "TAR@FAR=1e-5" not in result.metrics
    assert "Threshold@FAR=1e-2" not in result.metrics


def test_tar_far_skips_targets_with_zero_false_accept_budget():
    metrics = _tar_far_from_scores(
        positive_scores=[0.95, 0.9],
        positive_total=2,
        negative_scores=[0.1] * 99 + [0.2],
        far_targets=[1e-3, 1e-2],
    )

    assert "TAR@FAR=1e-3" not in metrics
    assert "False accept budget@1e-3" not in metrics
    assert metrics["False accept budget@1e-2"] == 1
    assert "TAR@FAR=1e-2" in metrics


def test_enterprise_validation_requires_one_face_by_default(tmp_path):
    root = tmp_path / "dataset"
    for rel in [
        "0001__Alice/gallery.jpg",
        "0001__Alice/multi_probe.jpg",
        "0002__Bob/gallery.jpg",
        "0002__Bob/probe.jpg",
    ]:
        _write_image(root / rel)

    result = validate_enterprise_dataset(
        root,
        "1:1 Verification",
        True,
        FakeEngine(),
        multi_face_policy=MULTI_FACE_REQUIRE_ONE,
    )

    assert not result.ok
    assert any("multiple faces" in row.get("error", "") for row in result.errors)
    assert result.summary["images_checked"] == 2
    assert result.summary["validation_scope"] == "all images"


def test_enterprise_validation_can_use_largest_face(tmp_path):
    root = tmp_path / "dataset"
    for rel in [
        "0001__Alice/gallery.jpg",
        "0001__Alice/multi_probe.jpg",
        "0002__Bob/gallery.jpg",
        "0002__Bob/probe.jpg",
    ]:
        _write_image(root / rel)

    result = validate_enterprise_dataset(
        root,
        "1:1 Verification",
        True,
        FakeEngine(),
        multi_face_policy=MULTI_FACE_USE_LARGEST,
    )

    assert result.ok
    assert result.summary["comparison_pairs"] == 4
    assert result.summary["images_checked"] == 2
    assert result.summary["validation_scope"] == "gallery or representative images only"
    assert result.warnings == []


def test_centered_largest_face_policy_uses_area_center_penalty():
    far_large = FaceRecord(
        bbox=[0, 0, 60, 60],
        kps=None,
        det_score=1.0,
        embedding=np.array([1.0], dtype=np.float32),
        normed_embedding=np.array([1.0], dtype=np.float32),
    )
    centered_medium = FaceRecord(
        bbox=[40, 40, 90, 90],
        kps=None,
        det_score=1.0,
        embedding=np.array([2.0], dtype=np.float32),
        normed_embedding=np.array([2.0], dtype=np.float32),
    )

    selected = _select_face_from_faces(
        [far_large, centered_medium],
        (128, 128, 3),
        Path("multi.jpg"),
        MULTI_FACE_USE_CENTERED_LARGEST,
        "test",
    )

    assert selected is centered_medium


def test_enterprise_validation_can_use_centered_largest_face(tmp_path):
    root = tmp_path / "dataset"
    for rel in [
        "0001__Alice/gallery.jpg",
        "0001__Alice/multi_probe.jpg",
        "0002__Bob/gallery.jpg",
        "0002__Bob/probe.jpg",
    ]:
        _write_image(root / rel)

    result = validate_enterprise_dataset(
        root,
        "1:1 Verification",
        True,
        FakeEngine(),
        multi_face_policy=MULTI_FACE_USE_CENTERED_LARGEST,
    )

    assert result.ok
    assert result.summary["comparison_pairs"] == 4
    assert result.summary["images_checked"] == 2
    assert result.summary["validation_scope"] == "gallery or representative images only"
    assert result.warnings == []


def test_enterprise_validation_skip_multi_face_gallery_blocks_structured_1n(tmp_path):
    root = tmp_path / "dataset_1n"
    for rel in [
        "gallery/0001__Alice/multi_enroll.jpg",
        "probe/0001__Alice/test_001.jpg",
    ]:
        _write_image(root / rel)

    result = validate_enterprise_dataset(
        root,
        "1:N Identification",
        False,
        FakeEngine(),
        multi_face_policy=MULTI_FACE_SKIP,
    )

    assert not result.ok
    assert any("gallery" in row.get("role", "") and row.get("error") for row in result.errors)


def test_enterprise_validation_fast_policy_fails_on_no_face_gallery(tmp_path):
    root = tmp_path / "dataset"
    for rel in [
        "0001__Alice/gallery_noface.jpg",
        "0001__Alice/probe.jpg",
        "0002__Bob/gallery.jpg",
        "0002__Bob/probe.jpg",
    ]:
        _write_image(root / rel)

    result = validate_enterprise_dataset(
        root,
        "1:1 Verification",
        True,
        FakeEngine(),
        multi_face_policy=MULTI_FACE_USE_CENTERED_LARGEST,
    )

    assert not result.ok
    assert result.summary["images_checked"] == 1
    assert any("no face detected" in row.get("error", "") for row in result.errors)


def test_enterprise_validation_fast_policy_does_not_inspect_probe_faces(tmp_path):
    root = tmp_path / "dataset"
    for rel in [
        "0001__Alice/gallery.jpg",
        "0001__Alice/probe_noface.jpg",
        "0002__Bob/gallery.jpg",
        "0002__Bob/probe.jpg",
    ]:
        _write_image(root / rel)

    result = validate_enterprise_dataset(
        root,
        "1:1 Verification",
        True,
        FakeEngine(),
        multi_face_policy=MULTI_FACE_USE_LARGEST,
    )

    assert result.ok
    assert result.summary["images_checked"] == 2
    assert result.summary["comparison_pairs"] == 4
