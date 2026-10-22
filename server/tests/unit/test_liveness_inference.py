from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from insightface.addons.liveness import DESTINATION_LANDMARKS, OUT_OF_BOUNDS_REASON
from insightface_server.addons import LivenessUnavailable
from insightface_server.config import DetectionProfile
from insightface_server.inference.concurrency import InferenceConcurrencyLimiter
from insightface_server.inference.onnx_engine import OnnxInsightFaceEngine


@pytest.fixture
def engine():
    calls = []

    class Detector:
        center_cache = {}
        static_input_size = None

        def detect(self, *args, **kwargs):
            return np.array(
                [[0, 0, 90, 90, 0.99], [100, 10, 180, 90, 0.99], [200, 10, 250, 60, 0.99]]
            ), np.stack([DESTINATION_LANDMARKS + [i * 100, 0] for i in range(3)])

    class Recognizer:
        def get_feat(self, crop):
            calls.append("recognition")
            return np.array([[1.0, 0.0]])

    class Liveness:
        def predict(self, image, landmarks):
            index = int(landmarks[0, 0] // 100)
            calls.append(f"liveness-{index}")
            return [
                {"status": "ok", "is_live": False, "live_score": 0.1},
                {
                    "status": "input_rejected", "is_live": None, "live_score": None,
                    "reason": OUT_OF_BOUNDS_REASON,
                },
                {"status": "ok", "is_live": True, "live_score": 0.99},
            ][index]

    value = OnnxInsightFaceEngine.__new__(OnnxInsightFaceEngine)
    value._concurrency = InferenceConcurrencyLimiter(2)
    value._lifecycle_lock = threading.RLock()
    value._started, value._closed = True, False
    value._detector, value._recognizer = Detector(), Recognizer()
    value._liveness, value._liveness_mode = Liveness(), "normal"
    value._default_detection_profile = DetectionProfile(input_sizes=((96, 96),))
    value.bundle = SimpleNamespace(recognizer=SimpleNamespace(input_size=(112, 112)))
    return value, calls


@pytest.mark.parametrize("mode,count", [("normal", 1), ("observe", 3)])
def test_liveness_runs_before_recognition_and_rejected_faces_are_retained(engine, mode, count):
    value, calls = engine
    value._liveness_mode = mode
    faces = value.analyze(np.zeros((100, 300, 3), np.uint8))
    assert len(faces) == 3
    assert calls.count("recognition") == count
    assert sum(face.embedding is not None for face in faces) == count
    assert all(set(faces[index].liveness) == {"status", "is_live", "live_score"} for index in (0, 2))
    assert faces[1].liveness == {
        "status": "input_rejected", "is_live": None, "live_score": None,
        "reason": OUT_OF_BOUNDS_REASON,
    }
    assert calls[0] == "liveness-0"


def test_selected_fake_is_not_replaced_by_a_live_background_face(engine):
    value, calls = engine
    faces = value.analyze(np.zeros((100, 300, 3), np.uint8), single_face=True)
    assert len(faces) == 1
    assert faces[0].liveness["is_live"] is False
    assert faces[0].embedding is None
    assert calls == ["liveness-0"]


def test_reference_side_can_skip_liveness_and_detection_never_extracts_features(engine):
    value, calls = engine
    faces = value.analyze(np.zeros((100, 300, 3), np.uint8), single_face=True, apply_liveness=False)
    assert calls == ["recognition"]
    assert faces[0].liveness is None
    calls.clear()
    faces = value.analyze(np.zeros((100, 300, 3), np.uint8), require_embeddings=False)
    assert calls == ["liveness-0", "liveness-1", "liveness-2"]
    assert all(face.embedding is None for face in faces)


@pytest.mark.parametrize("mode", ["normal", "observe"])
@pytest.mark.parametrize("failure", [ValueError("Invalid landmarks"), RuntimeError("Alignment failed")])
def test_internal_liveness_fault_aborts_before_recognition(engine, monkeypatch, mode, failure):
    value, calls = engine
    value._liveness_mode = mode

    def fail(*args, **kwargs):
        calls.append("liveness-failed")
        raise failure

    monkeypatch.setattr(value._liveness, "predict", fail)
    with pytest.raises(LivenessUnavailable, match="Liveness inference failed") as caught:
        value.analyze(np.zeros((100, 300, 3), np.uint8))
    assert caught.value.__cause__ is failure
    assert calls == ["liveness-failed"]
