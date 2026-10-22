from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from insightface.app.privateframe.streaming import StreamingEngine


class _Reviewer:
    def __init__(self) -> None:
        self.local_frames: list[int] = []
        self.verifier_boxes: list[list[list[float]]] = []
        self.local_result: dict[str, object] | None = None

    def local_match(
        self,
        _frame: np.ndarray,
        box: list[float],
        *,
        candidate_selection: str,
    ) -> dict[str, object]:
        assert candidate_selection == "confidence"
        self.local_frames.append(int(box[0]))
        if self.local_result is not None:
            return dict(self.local_result)
        return {
            "local_match_count": 1,
            "local_confidence": 0.9,
            "local_box": box,
            "local_landmarks": None,
        }

    def verify(
        self,
        _frame: np.ndarray,
        boxes: list[list[float]],
    ) -> list[dict[str, float]]:
        self.verifier_boxes.append(boxes)
        return [{"face_probability": 0.8}]


def _engine(*, stride: int) -> StreamingEngine:
    engine = StreamingEngine.__new__(StreamingEngine)
    engine.fast_review_mode = stride > 1
    engine.local_review_stride = stride if engine.fast_review_mode else 1
    engine.local_review_phase = (
        engine.local_review_stride // 2 if engine.fast_review_mode else 0
    )
    engine.reviewer = _Reviewer()
    engine._verifier_scores = {}
    engine.local_review_attempts = 0
    engine.local_review_sampled_out = 0
    engine.local_review_forced = 0
    engine.verifier_review_calls = 0
    engine.verifier_review_cache_hits = 0
    return engine


def _item(frame_idx: int, *, source: str = "kalman_optical_flow") -> dict[str, object]:
    return {
        "frame_idx": frame_idx,
        "track_id": "t00001",
        "source": source,
        # Encoding the frame in x1 lets the fake reviewer report which frames
        # actually reached Local SCRFD without needing another test hook.
        "box": [float(frame_idx), 1.0, float(frame_idx + 4), 5.0],
    }


def test_stride_four_uses_complementary_local_phase_and_verifies_every_frame() -> None:
    engine = _engine(stride=4)
    frame = np.zeros((12, 12, 3), dtype=np.uint8)

    reviews = [engine._measure_review(frame, _item(index)) for index in range(8)]

    assert engine.reviewer.local_frames == [2, 6]
    assert engine.local_review_attempts == 2
    assert engine.local_review_sampled_out == 6
    assert engine.verifier_review_calls == 8
    assert len(engine.reviewer.verifier_boxes) == 8
    assert [value["local_match_count"] for value in reviews] == [
        -1,
        -1,
        1,
        -1,
        -1,
        -1,
        1,
        -1,
    ]
    assert {value["verifier_face_probability"] for value in reviews} == {0.8}


@pytest.mark.parametrize("stride", [8, 16])
def test_large_strides_keep_the_complementary_local_review_phase(
    stride: int,
) -> None:
    engine = _engine(stride=stride)
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    reviews = [
        engine._measure_review(frame, _item(index))
        for index in range(stride * 2)
    ]

    assert engine.local_review_phase == stride // 2
    assert engine.reviewer.local_frames == [stride // 2, stride + stride // 2]
    assert engine.local_review_attempts == 2
    assert engine.local_review_sampled_out == stride * 2 - 2
    assert engine.verifier_review_calls == stride * 2
    assert sum(value["local_match_count"] == 1 for value in reviews) == 2


def test_fast_review_can_force_local_without_skipping_verifier() -> None:
    engine = _engine(stride=4)
    frame = np.zeros((12, 12, 3), dtype=np.uint8)

    review = engine._measure_review(
        frame,
        _item(3),
        force_local=True,
        local_review_reason="long_gap_anchor",
    )

    assert review["local_match_count"] == 1
    assert review["local_review_reason"] == "long_gap_anchor"
    assert review["verifier_face_probability"] == 0.8
    assert engine.local_review_attempts == 1
    assert engine.local_review_forced == 1
    assert engine.verifier_review_calls == 1


def test_forced_detector_upgrade_reuses_verifier_and_replaces_skip_audit() -> None:
    engine = _engine(stride=4)
    engine.evidence = []
    engine.recognition_engine = SimpleNamespace(enabled=False)
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    item = {
        "detection_id": "d00001",
        "frame_idx": 0,
        "source": "detector",
        "box": [1.0, 1.0, 5.0, 5.0],
    }
    item["_review_measurement"] = engine._measure_review(frame, item)
    item["track_id"] = "t00001"

    review = engine._review(
        frame,
        item,
        force_local=True,
        local_review_reason="new_track_anchor",
    )

    assert review["local_match_count"] == 1
    assert review["verifier_face_probability"] == 0.8
    assert engine.local_review_attempts == 1
    assert engine.local_review_sampled_out == 0
    assert engine.verifier_review_calls == 1
    assert engine.verifier_review_cache_hits == 1
    assert len(engine.reviewer.verifier_boxes) == 1


def test_derived_stride_uses_the_same_sampled_review_policy() -> None:
    engine = _engine(stride=3)
    frame = np.zeros((12, 12, 3), dtype=np.uint8)

    tracking = engine._measure_review(frame, _item(1))
    detector = engine._measure_review(frame, _item(2, source="detector"))

    assert engine.reviewer.local_frames == [1]
    assert tracking["verifier_face_probability"] == 0.8
    assert detector["verifier_face_probability"] == 0.8
    assert engine.verifier_review_calls == 2


def test_recognition_landmarks_are_local_first_then_global_detector_fallback() -> None:
    item = {
        "source": "detector",
        "detector_box": [1.0, 2.0, 5.0, 6.0],
        "detector_landmarks": np.arange(10, dtype=np.float32).reshape(5, 2),
        "confidence": 0.85,
        "scan_angle_degrees": 0,
    }
    local_review = {
        "local_box": [1.5, 2.5, 5.5, 6.5],
        "local_landmarks": np.ones((5, 2), dtype=np.float32),
        "local_confidence": 0.9,
        "local_angle": 0,
    }

    proposals = StreamingEngine._ordered_landmark_proposals(item, local_review)

    assert [proposal.source for proposal in proposals] == [
        "local_scrfd",
        "global_scrfd",
    ]
    assert proposals[0].require_reference_agreement is True
    assert proposals[1].require_reference_agreement is False


def test_verifier_cache_key_survives_detector_assignment_to_track() -> None:
    engine = _engine(stride=4)
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    item = {
        "detection_id": "d00001",
        "frame_idx": 2,
    }
    box = [1.0, 1.0, 5.0, 5.0]

    first = engine._verify_once(frame, item, box)
    item["track_id"] = "t00001"
    second = engine._verify_once(frame, item, box)

    assert first == second == 0.8
    assert engine.verifier_review_calls == 1
    assert engine.verifier_review_cache_hits == 1
    assert len(engine.reviewer.verifier_boxes) == 1


def test_anchor_recovery_translates_landmarks_and_repairs_verifier_pair() -> None:
    engine = _engine(stride=4)
    engine.config = {
        "revalidation": {
            "match_max_center_distance": 1.0,
            "geometry_refinement": {
                "enabled": True,
                "min_local_confidence": 0.25,
                "max_area_ratio": 2.5,
                "max_center_distance": 0.5,
                "anchor_recovery": {
                    "enabled": True,
                    "candidate_selection": "confidence",
                },
            },
        },
    }
    raw_landmarks = np.asarray(
        [[12.0, 18.0], [28.0, 18.0], [20.0, 24.0], [14.0, 30.0], [26.0, 30.0]],
        dtype=np.float64,
    )
    engine.reviewer.local_result = {
        "local_match_count": 1,
        "local_confidence": 0.9,
        "local_box": [10.0, 15.0, 30.0, 35.0],
        "local_landmarks": raw_landmarks.tolist(),
    }
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    item = {
        "frame_idx": 3,
        "track_id": "t00001",
        "source": "kalman_optical_flow",
        "box": [100.0, 100.0, 120.0, 120.0],
    }
    review = {
        "local_match_count": -1,
        "local_confidence": None,
        "local_box": None,
        "local_landmarks": None,
        "verifier_face_probability": 0.1,
    }
    origin = (10, 5)
    anchor = np.asarray([20.0, 20.0, 40.0, 40.0], dtype=np.float64)

    recovered = engine._recover_tracking_geometry(
        frame,
        item,
        review,
        anchor,
        origin,
    )

    np.testing.assert_allclose(recovered, anchor)
    np.testing.assert_allclose(review["local_box"], anchor)
    np.testing.assert_allclose(
        review["local_landmarks"],
        raw_landmarks + np.asarray(origin),
    )
    assert review["verifier_face_probability"] == 0.8
    assert engine.reviewer.verifier_boxes == [[[10.0, 15.0, 30.0, 35.0]]]
