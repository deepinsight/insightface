# -*- coding: utf-8 -*-
"""Pickle round-trip must preserve caller-supplied session options.

Regression tests for https://github.com/deepinsight/insightface/issues/2899:
PickableInferenceSession used to drop sess_options on unpickling, so a
restored session came back with the default thread pool sizing and
oversubscribed the CPU in quota-limited containers.
"""

import pickle

import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import pytest

from insightface.model_zoo.model_zoo import PickableInferenceSession


def _identity_model(path):
    """Write a tiny Identity ONNX model - no download needed."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "identity",
        [x],
        [y],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def identity_model(tmp_path):
    return _identity_model(tmp_path / "identity.onnx")


def _cpu_providers():
    return ["CPUExecutionProvider"]


def test_session_options_survive_pickling(identity_model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    session = PickableInferenceSession(
        identity_model,
        sess_options=options,
        providers=_cpu_providers(),
    )
    assert session.get_session_options().intra_op_num_threads == 1

    restored = pickle.loads(pickle.dumps(session))
    assert restored.get_session_options().intra_op_num_threads == 1
    assert restored.get_session_options().inter_op_num_threads == 1


def test_default_options_round_trip_unchanged(identity_model):
    """A session built without explicit options still pickles fine."""
    session = PickableInferenceSession(identity_model, providers=_cpu_providers())
    restored = pickle.loads(pickle.dumps(session))
    assert restored is not None


def test_state_without_session_options_key_is_supported(identity_model):
    """States written before the fix (no 'session_options') still unpickle."""
    session = PickableInferenceSession(identity_model, providers=_cpu_providers())
    state = session.__getstate__()
    state.pop("session_options")

    restored = PickableInferenceSession.__new__(PickableInferenceSession)
    restored.__setstate__(state)
    assert restored.model_path == session.model_path
