from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from insightface.app.privateframe import recognition


class Recognizer:
    input_size = (112, 112)

    def __init__(self, *outputs: Any) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    def get_feat(self, _image: np.ndarray) -> np.ndarray:
        output = self.outputs[self.calls]
        self.calls += 1
        if isinstance(output, BaseException):
            raise output
        return np.asarray([output], dtype=np.float32)


def photos(tmp_path: Path, *names: str) -> Path:
    root = tmp_path / "references"
    root.mkdir()
    for index, name in enumerate(names):
        (root / name).write_bytes(f"reference-{index}".encode())
    return root


def face(box: tuple[float, ...] = (20, 20, 100, 100)) -> dict[str, Any]:
    return {"box": box, "confidence": 0.99, "landmarks": recognition.ARC_FACE_TEMPLATE_112.copy()}


@pytest.fixture
def geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recognition, "arcface_align_112", lambda *_: np.zeros((112, 112, 3), dtype=np.uint8))
    monkeypatch.setattr(recognition, "recognition_candidate_quality", lambda *_args, **_kwargs: (1.0, True, {}))


def load_image(_path: Path) -> np.ndarray:
    return np.zeros((160, 160, 3), dtype=np.uint8)


def test_scan_only_direct_supported_reference_photos(tmp_path: Path) -> None:
    root = photos(tmp_path, "one.JPG", "two.jpeg", "three.png", "four.webp", ".hidden.jpg", "notes.txt")
    nested = root / "somebody"
    nested.mkdir()
    (nested / "nested.jpg").write_bytes(b"nested")
    (root / "linked.jpg").symlink_to(root / "one.JPG")
    scanned = recognition.scan_gallery(root)
    assert [value.relative_file_name for value in scanned.images] == ["four.webp", "one.JPG", "three.png", "two.jpeg"]


def test_largest_face_selected_before_validation_and_logged(
    tmp_path: Path, geometry: None, caplog: pytest.LogCaptureFixture,
) -> None:
    root = photos(tmp_path, "group.jpg")
    small, large = face((5, 5, 35, 35)), face((10, 10, 140, 140))
    with caplog.at_level(logging.INFO, logger=recognition.__name__):
        result = recognition.build_gallery(root, lambda _: [small, large], Recognizer([1, 0]), image_loader=load_image)
    assert result.references[0].selected_box == (10.0, 10.0, 140.0, 140.0)
    assert result.references[0].detected_face_count == 2
    assert "group.jpg: detected 2 faces; using only the largest face" in caplog.text
    assert "read 1, used 1, skipped 0" in caplog.text


@pytest.mark.parametrize("invalid", ["landmarks", "quality"])
def test_invalid_largest_face_skips_photo_without_smaller_fallback(
    tmp_path: Path, geometry: None, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, invalid: str,
) -> None:
    root = photos(tmp_path, "group.jpg", "valid.jpg")
    small, large = face((5, 5, 35, 35)), face((10, 10, 140, 140))
    if invalid == "landmarks":
        large.pop("landmarks")
    else:
        monkeypatch.setattr(recognition, "recognition_candidate_quality", lambda _crop, box, *_args: (1.0, box[2] < 140, {}))
    count = 0

    def detector(_image: Any) -> list[dict[str, Any]]:
        nonlocal count
        count += 1
        return [small, large] if count == 1 else [face()]

    recognizer = Recognizer([1, 0])
    result = recognition.build_gallery(root, detector, recognizer, image_loader=load_image)
    assert [ref.file_name for ref in result.references] == ["valid.jpg"]
    assert result.rejections[0].file_name == "group.jpg"
    assert result.rejections[0].reason == ("missing_landmarks" if invalid == "landmarks" else "low_quality")
    assert recognizer.calls == 1
    assert "group.jpg was not used" in caplog.text


def test_same_person_photos_retained_and_different_people_not_averaged(tmp_path: Path, geometry: None) -> None:
    root = photos(tmp_path, "one.jpg", "other.jpg", "same.jpg")
    result = recognition.build_gallery(
        root, lambda _: [face()], Recognizer([1, 0], [0, 1], [0.99, 0.01]), image_loader=load_image,
    )
    assert len(result.references) == len(result.prototypes) == 3
    np.testing.assert_allclose(result.prototypes["one.jpg"], [1, 0])
    np.testing.assert_allclose(result.prototypes["other.jpg"], [0, 1])
    assert recognition.decide_track_identity([sample(0, [1, 0])], result.prototypes, 0.7).status is recognition.IdentityStatus.CONFIRMED
    assert recognition.decide_track_identity([sample(0, [0, 1])], result.prototypes, 0.7).status is recognition.IdentityStatus.CONFIRMED


def test_duplicate_photo_content_skipped_with_reason(tmp_path: Path, geometry: None, caplog: pytest.LogCaptureFixture) -> None:
    root = photos(tmp_path, "a.jpg", "copy.jpg")
    (root / "copy.jpg").write_bytes((root / "a.jpg").read_bytes())
    recognizer = Recognizer([1, 0])
    result = recognition.build_gallery(root, lambda _: [face()], recognizer, image_loader=load_image)
    assert recognizer.calls == 1
    assert result.rejections[0].reason == "duplicate_content"
    assert "copy.jpg was not used: duplicate_content" in caplog.text


@pytest.mark.parametrize("mode", ["blur_only", "exempt"])
def test_no_usable_reference_faces_fails_for_both_selective_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    root = photos(tmp_path, "empty.jpg")
    monkeypatch.setattr(recognition.cv2, "imread", lambda *_: load_image(root))
    with pytest.raises(ValueError, match="No usable faces were found"):
        recognition.create_recognition_engine(
            {"mode": mode, "reference_dir": str(root), "similarity_threshold": 0.4},
            recognizer=Recognizer(), gallery_detector=lambda _: [],
        )


def test_empty_folder_is_not_treated_as_named_person_gallery(tmp_path: Path) -> None:
    root = photos(tmp_path)
    (root / "person").mkdir()
    (root / "person" / "face.jpg").write_bytes(b"not read")
    with pytest.raises(ValueError, match="No usable faces"):
        recognition.build_gallery(root, lambda _: [face()], Recognizer([1, 0]), image_loader=load_image)


@pytest.mark.parametrize("stage", ["detector", "recognizer"])
def test_reference_model_inference_failure_aborts_with_filename(
    tmp_path: Path, geometry: None, stage: str,
) -> None:
    root = photos(tmp_path, "photo.jpg")

    def broken_detector(_image: Any) -> Any:
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match=f"photo.jpg: .*{stage} inference failed") as caught:
        recognition.build_gallery(
            root, broken_detector if stage == "detector" else lambda _: [face()],
            Recognizer(RuntimeError("provider failed")), image_loader=load_image,
        )
    assert isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.parametrize("error", [MemoryError(), KeyboardInterrupt(), SystemExit(2)])
def test_reference_inference_fatal_errors_propagate(tmp_path: Path, geometry: None, error: BaseException) -> None:
    root = photos(tmp_path, "photo.jpg")
    with pytest.raises(type(error)):
        recognition.build_gallery(root, lambda _: [face()], Recognizer(error), image_loader=load_image)


def sample(index: int, vector: Any) -> recognition.EmbeddingSample:
    return recognition.EmbeddingSample(index, np.asarray(vector, dtype=np.float32))


def test_duplicate_and_same_person_references_do_not_compete() -> None:
    references = {"a.jpg": np.array([1.0, 0.0]), "b.jpg": np.array([1.0, 0.0]), "c.jpg": np.array([0.99, 0.01])}
    result = recognition.decide_track_identity([sample(0, [1, 0])], references, 0.7)
    assert result.status is recognition.IdentityStatus.CONFIRMED
    assert result.matched_reference_files == ("a.jpg",)
    assert result.similarity == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("vector", "expected"),
    [([0, 1], recognition.IdentityStatus.CONFLICT), ([0.5, 0.866], recognition.IdentityStatus.UNKNOWN)],
)
def test_one_unmatched_sample_vetoes_reference_membership(vector: Any, expected: Any) -> None:
    result = recognition.decide_track_identity(
        [sample(0, [1, 0]), sample(30, [1, 0]), sample(60, vector)],
        {"a.jpg": np.array([1.0, 0.0])}, 0.7,
    )
    assert result.status is expected
    assert result.support == 2
    assert result.frame_indices == (0, 30, 60)


def test_every_reference_person_belongs_to_selected_set() -> None:
    result = recognition.decide_track_identity(
        [sample(0, [1, 0]), sample(30, [0, 1]), sample(60, [1, 0])],
        {"a.jpg": np.array([1.0, 0.0]), "b.jpg": np.array([0.0, 1.0])}, 0.7,
    )
    assert result.status is recognition.IdentityStatus.CONFIRMED
    assert result.matched_reference_files == ("a.jpg", "b.jpg")


def test_unmatched_video_is_normal_unknown() -> None:
    result = recognition.decide_track_identity([sample(0, [0, 1])], {"a.jpg": np.array([1.0, 0.0])}, 0.7)
    assert result.status is recognition.IdentityStatus.UNKNOWN
    assert result.matched_reference_files == ()
    assert result.reason == "below_similarity_threshold"


def test_single_frame_requires_stricter_threshold() -> None:
    result = recognition.decide_track_identity([sample(0, [0.72, np.sqrt(1 - 0.72 ** 2)])], {"a.jpg": np.array([1.0, 0.0])}, 0.7)
    assert result.status is recognition.IdentityStatus.UNKNOWN


@pytest.mark.parametrize(
    ("mode", "unknown", "confirmed_blur", "unknown_blur"),
    [
        ("all", "auto", True, True), ("all", "keep", True, True),
        ("blur_only", "auto", True, False), ("exempt", "auto", False, True),
        ("blur_only", "blur", True, True), ("exempt", "keep", False, False),
    ],
)
def test_policy_defaults_and_explicit_unknown_action(
    mode: str, unknown: str, confirmed_blur: bool, unknown_blur: bool,
) -> None:
    confirmed = recognition.IdentityDecision(recognition.IdentityStatus.CONFIRMED, matched_reference_files=("a.jpg",))
    assert recognition.apply_identity_policy(mode, confirmed, unknown).should_blur is confirmed_blur
    for status in (recognition.IdentityStatus.UNKNOWN, recognition.IdentityStatus.CONFLICT):
        assert recognition.apply_identity_policy(mode, recognition.IdentityDecision(status), unknown).should_blur is unknown_blur


@pytest.mark.parametrize("record", [None, {}, {"status": "bad"}, {"status": "CONFIRMED", "matched_reference_files": []}])
def test_missing_or_malformed_artifact_is_an_error(record: Any) -> None:
    with pytest.raises(ValueError):
        recognition.apply_identity_policy("blur_only", record)


def test_all_mode_does_not_load_references_or_recognizer() -> None:
    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("all mode touched reference model state")
    engine = recognition.create_recognition_engine(
        {"mode": "all", "reference_dir": "/does/not/exist", "profile": "invalid"},
        recognizer=None, gallery_detector=None, gallery_builder=unexpected,
    )
    assert not engine.enabled
    assert engine.unknown_action == "blur"


def test_skipped_photos_report_reasons_and_photo_totals(
    tmp_path: Path, geometry: None, caplog: pytest.LogCaptureFixture,
) -> None:
    root = photos(tmp_path, "a-unreadable.jpg", "b-empty.jpg", "c-valid.jpg")
    calls = 0

    def detector(_image: Any) -> list[Any]:
        nonlocal calls
        calls += 1
        return [] if calls == 1 else [face()]

    with caplog.at_level(logging.INFO, logger=recognition.__name__):
        result = recognition.build_gallery(
            root, detector, Recognizer([1, 0]),
            image_loader=lambda path: None if path.name == "a-unreadable.jpg" else load_image(path),
        )
    assert [item.reason for item in result.rejections] == ["unreadable_image", "no_face"]
    assert "a-unreadable.jpg was not used: unreadable_image" in caplog.text
    assert "b-empty.jpg was not used: no_face" in caplog.text
    assert "read 3, used 1, skipped 2" in caplog.text


def test_inconsistent_model_embedding_widths_fail_before_track_analysis(tmp_path: Path, geometry: None) -> None:
    root = photos(tmp_path, "a.jpg", "b.jpg")
    with pytest.raises(ValueError, match="same width"):
        recognition.build_gallery(root, lambda _: [face()], Recognizer([1, 0], [1, 0, 0]), image_loader=load_image)


@pytest.mark.parametrize(
    ("indices", "threshold", "reason", "status"),
    [
        ([0], 0.85, "insufficient_selected_frames+single_frame_offset", recognition.IdentityStatus.UNKNOWN),
        ([0, 1, 2], 0.80, "insufficient_temporal_separation", recognition.IdentityStatus.UNKNOWN),
        ([0, 30, 60], 0.70, "independent_temporal_evidence", recognition.IdentityStatus.CONFIRMED),
    ],
)
def test_engine_preserves_temporal_evidence_thresholds(
    indices: list[int], threshold: float, reason: str, status: Any,
) -> None:
    vector = [0.75, np.sqrt(1 - 0.75 ** 2)]
    engine = recognition.RecognitionEngine(
        enabled=True, mode="blur_only", unknown_action="keep",
        profile=recognition.RECOGNITION_PROFILES["balanced"], similarity_threshold=0.7,
        recognizer=Recognizer(*([vector] * len(indices))),
        gallery=type("References", (), {"prototypes": {"a.jpg": np.array([1.0, 0.0])}})(),
    )
    result = engine.identify_track(
        [recognition.RecognitionCandidate(i, 1.0, np.zeros((112, 112, 3), dtype=np.uint8)) for i in indices],
        frames_per_second=30.0,
    )
    assert result.status is status
    assert result.selected_frame_count == len(indices)
    assert result.effective_similarity_threshold == pytest.approx(threshold)
    assert result.threshold_reason == reason


@pytest.mark.parametrize("value", [[], {}, None, True, 1, "invalid"])
def test_invalid_unknown_action_has_clear_validation_error(value: Any) -> None:
    with pytest.raises(ValueError, match="recognition unknown_action must be one of"):
        recognition.resolve_unknown_action("blur_only", value)


@pytest.mark.parametrize("files", [["../face.jpg"], ["face\\photo.jpg"], ["bad\x00.jpg"], [42], ["café.jpg", "cafe\u0301.jpg"]])
def test_artifact_rejects_invalid_or_unicode_duplicate_filenames(files: Any) -> None:
    with pytest.raises(ValueError, match="filenames"):
        recognition.apply_identity_policy("blur_only", {"status": "CONFIRMED", "matched_reference_files": files})


def test_artifact_normalizes_reference_filenames_to_nfc() -> None:
    record = {"status": "CONFIRMED", "matched_reference_files": ["cafe\u0301.jpg"]}
    decision = recognition._identity_from_artifact(record)
    assert decision.matched_reference_files == ("café.jpg",)
    assert recognition.apply_identity_policy("exempt", decision).should_blur is False
