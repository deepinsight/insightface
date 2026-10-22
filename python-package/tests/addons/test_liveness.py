from types import SimpleNamespace

import numpy as np
import pytest
from insightface.addons.liveness import (
    DESTINATION_LANDMARKS,
    OUT_OF_BOUNDS_REASON,
    Liveness,
)


class Session:
    def __init__(self, score=0.9):
        self.output = np.asarray([score])
        self.feeds = []
        self.providers = None
        self.inputs = [
            SimpleNamespace(name="input", type="tensor(float)", shape=["N", 3, 80, 80])
        ]
        self.outputs = [
            SimpleNamespace(name="live_score", type="tensor(float)", shape=["N"])
        ]

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, names, feed):
        assert names == ["live_score"]
        self.feeds.append(feed["input"])
        return [self.output]

    def set_providers(self, providers):
        self.providers = providers


def image():
    return np.broadcast_to(np.array([10, 50, 240], dtype=np.uint8), (80, 80, 3)).copy()


@pytest.mark.parametrize(
    "score,expected", [(0.0, False), (0.79999, False), (0.8, True), (1.0, True)]
)
def test_score_contract_and_rgb_division_without_second_softmax(score, expected):
    session = Session(score)
    result = Liveness("unused.onnx", session=session).predict(
        image(), DESTINATION_LANDMARKS
    )
    assert result == {"status": "ok", "is_live": expected, "live_score": score}
    tensor = session.feeds[0]
    assert tensor.shape == (1, 3, 80, 80)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    np.testing.assert_allclose(tensor[0, :, 40, 40], np.array([240, 50, 10]) / 255)


@pytest.mark.parametrize("shift,accepted", [(23, True), (25, False), (80, False)])
def test_gate_measures_missing_aligned_area_before_inference(shift, accepted):
    session = Session()
    landmarks = DESTINATION_LANDMARKS + [shift, 0]
    result = Liveness("unused.onnx", session=session).predict(image(), landmarks)
    assert len(session.feeds) == int(accepted)
    if accepted:
        assert result["status"] == "ok"
        # Accepted out-of-image pixels use replicated image values, not zeros.
        np.testing.assert_allclose(
            session.feeds[0][0, :, 40, -1], np.array([240, 50, 10]) / 255
        )
    else:
        assert result == {
            "status": "input_rejected",
            "is_live": None,
            "live_score": None,
            "reason": OUT_OF_BOUNDS_REASON,
        }
        assert result["reason"] == (
            "Insufficient image area around the face for liveness detection. "
            "Move the face toward the center, step back from the camera, "
            "or use a less tightly cropped image."
        )


@pytest.mark.parametrize(
    "landmarks",
    [
        None,
        [],
        "not landmarks",
        np.zeros((4, 2)),
        np.full((5, 2), np.nan),
        np.full((5, 2), np.inf),
    ],
)
def test_invalid_landmarks_raise_without_running_model(landmarks):
    session = Session()
    with pytest.raises(ValueError, match="landmarks"):
        Liveness("unused.onnx", session=session).predict(image(), landmarks)
    assert session.feeds == []


def test_degenerate_landmarks_raise_without_running_model():
    session = Session()
    with pytest.raises(RuntimeError, match="alignment failed"):
        Liveness("unused.onnx", session=session).predict(image(), np.zeros((5, 2)))
    assert session.feeds == []


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        np.zeros((0, 80, 3), np.uint8),
        np.zeros((80, 80), np.uint8),
        np.zeros((80, 80, 3), np.float32),
    ],
)
def test_invalid_images_raise_instead_of_becoming_fake(invalid):
    with pytest.raises(ValueError, match="BGR"):
        Liveness("unused.onnx", session=Session()).predict(
            invalid, DESTINATION_LANDMARKS
        )


@pytest.mark.parametrize("score", [np.nan, np.inf, -0.1, 1.01])
def test_invalid_model_probabilities_raise(score):
    with pytest.raises(RuntimeError, match="invalid probability"):
        Liveness("unused.onnx", session=Session(score)).predict(
            image(), DESTINATION_LANDMARKS
        )


def test_incompatible_model_and_runtime_shape_raise():
    session = Session()
    session.outputs[0].shape = ["N", 3]
    with pytest.raises(RuntimeError, match="live probability"):
        Liveness("unused.onnx", session=session)
    session.outputs[0].shape = ["N"]
    model = Liveness("unused.onnx", session=session)
    session.output = np.array([[0.9]])
    with pytest.raises(RuntimeError, match="score shape"):
        model.predict(image(), DESTINATION_LANDMARKS)


def test_inference_exception_propagates():
    session = Session()

    def fail(*args):
        raise RuntimeError("provider failed")

    session.run = fail
    with pytest.raises(RuntimeError, match="provider failed"):
        Liveness("unused.onnx", session=session).predict(image(), DESTINATION_LANDMARKS)


def test_custom_threshold_and_cpu_preparation():
    session = Session(0.85)
    model = Liveness("unused.onnx", session=session, threshold=0.9)
    model.prepare(-1)
    assert session.providers == ["CPUExecutionProvider"]
    assert model.predict(image(), DESTINATION_LANDMARKS)["is_live"] is False
