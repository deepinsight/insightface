from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from insightface.app.privateframe import recognition, streaming


class _Recognizer:
    input_size = (112, 112)

    def __init__(self, *outputs: Any) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    def get_feat(self, _aligned: np.ndarray) -> np.ndarray:
        output = self.outputs[self.calls]
        self.calls += 1
        if isinstance(output, BaseException):
            raise output
        return np.asarray(output)


def _candidate(frame_index: int = 0) -> recognition.RecognitionCandidate:
    return recognition.RecognitionCandidate(
        frame_index=frame_index,
        quality=1.0,
        aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
    )


def _sourced_candidate(
    frame_index: int,
    quality: float,
    landmark_source: str,
) -> recognition.RecognitionCandidate:
    return recognition.RecognitionCandidate(
        frame_index=frame_index,
        quality=quality,
        aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
        landmark_source=landmark_source,
    )


def test_recognition_candidate_defaults_to_local_scrfd_landmarks() -> None:
    assert _candidate().landmark_source == "local_scrfd"


def test_recognition_candidate_rejects_unknown_landmark_source() -> None:
    with pytest.raises(ValueError, match="landmark_source"):
        _sourced_candidate(0, 1.0, "optical_flow")


def test_temporal_selection_prefers_local_and_falls_back_to_global() -> None:
    candidates = [
        _sourced_candidate(0, 0.99, "global_scrfd"),
        _sourced_candidate(2, 0.40, "local_scrfd"),
        _sourced_candidate(8, 0.60, "global_scrfd"),
        _sourced_candidate(9, 0.90, "global_scrfd"),
    ]

    selected = recognition.select_temporally_distributed(candidates, 2)

    assert [value.frame_index for value in selected] == [2, 9]
    assert [value.landmark_source for value in selected] == [
        "local_scrfd",
        "global_scrfd",
    ]
    assert [
        (value.frame_index, value.landmark_source)
        for value in recognition.select_temporally_distributed(
            list(reversed(candidates)),
            2,
        )
    ] == [
        (2, "local_scrfd"),
        (9, "global_scrfd"),
    ]


def test_temporal_selection_uses_only_one_proposal_from_the_same_frame() -> None:
    candidates = [
        _sourced_candidate(5, 1.00, "global_scrfd"),
        _sourced_candidate(5, 0.25, "local_scrfd"),
        _sourced_candidate(12, 0.50, "global_scrfd"),
    ]

    selected = recognition.select_temporally_distributed(candidates, 3)

    assert [(value.frame_index, value.landmark_source) for value in selected] == [
        (5, "local_scrfd"),
        (12, "global_scrfd"),
    ]


def _recognition_engine(recognizer: Any) -> recognition.RecognitionEngine:
    return recognition.RecognitionEngine(
        enabled=True,
        mode="exempt",
        profile=recognition.RECOGNITION_PROFILES["fast"],
        unknown_action="blur",
        similarity_threshold=0.5,
        recognizer=recognizer,
        gallery=SimpleNamespace(prototypes={"photo.jpg": np.asarray([0.6, 0.8])}),
    )


@pytest.mark.parametrize(
    ("output", "error_type"),
    [
        (RuntimeError("onnx failure details"), RuntimeError),
        (np.asarray([[0.0, 0.0]], dtype=np.float32), ValueError),
        (np.asarray([3.0, 4.0], dtype=np.float32), RuntimeError),
    ],
)
def test_track_embedding_failure_aborts_instead_of_becoming_unknown(
    output: Any, error_type: type[Exception],
) -> None:
    engine = _recognition_engine(_Recognizer(output))
    with pytest.raises(error_type):
        engine.identify_track([_candidate()], frames_per_second=30.0)


def test_track_decision_failure_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _recognition_engine(_Recognizer(np.asarray([[3.0, 4.0]])))

    def fail_decision(*_args: Any, **_kwargs: Any) -> Any:
        raise LookupError("corrupt reference embedding")

    monkeypatch.setattr(recognition, "decide_track_identity", fail_decision)
    with pytest.raises(LookupError, match="corrupt reference"):
        engine.identify_track([_candidate()], frames_per_second=30.0)


@pytest.mark.parametrize(
    "fatal_error", [KeyboardInterrupt(), SystemExit(2), MemoryError()],
    ids=["keyboard-interrupt", "system-exit", "memory-error"],
)
def test_track_does_not_swallow_fatal_errors(fatal_error: BaseException) -> None:
    engine = _recognition_engine(_Recognizer(fatal_error))
    with pytest.raises(type(fatal_error)):
        engine.identify_track([_candidate()], frames_per_second=30.0)


class _StreamingRecognitionEngine:
    enabled = True
    mode = "blur_only"
    unknown_action = "keep"
    profile = recognition.RECOGNITION_PROFILES["fast"]
    similarity_threshold = 0.5

    def __init__(self) -> None:
        self.calls = 0

    @property
    def max_frames_per_track(self) -> int:
        return self.profile.max_frames_per_track

    def identify_track(self, _candidates: Any, *, frames_per_second: float) -> Any:
        assert frames_per_second == 30.0
        self.calls += 1
        raise RuntimeError("unexpected custom engine failure")


def test_streaming_does_not_convert_inference_failure_to_unknown() -> None:
    recognizer = _StreamingRecognitionEngine()
    engine = object.__new__(streaming.StreamingEngine)
    engine.recognition_engine = recognizer
    engine.fps = 30.0
    engine.tracks = [{"accepted": True, "track_id": "track-1", "detections": [], "accepted_intervals": []}]
    engine.recognition_candidates = {}
    engine.recognition_candidate_overflow_tracks = set()
    engine.recognition_candidates_prepared = 0
    engine.recognition_candidate_errors = 0
    engine.recognition_candidate_rejections = 0
    engine.recognition_setup_seconds = 0.0
    engine.recognition_candidate_prepare_seconds = 0.0
    with pytest.raises(RuntimeError, match="unexpected custom engine failure"):
        engine._finalize_recognition_impl({})
    assert recognizer.calls == 1
