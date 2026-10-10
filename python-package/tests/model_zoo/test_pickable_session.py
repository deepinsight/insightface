"""Tests for inference session options across pickling."""
import pickle

import onnx
import onnxruntime
from onnx import TensorProto, helper

from insightface.model_zoo.model_zoo import PickableInferenceSession


def _identity_model(path):
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])], "identity", [inp], [out]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return str(path)


def test_pickle_preserves_scalar_session_options(tmp_path):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = PickableInferenceSession(
        _identity_model(tmp_path / "identity.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    restored = pickle.loads(pickle.dumps(session)).get_session_options()

    assert restored.intra_op_num_threads == 1
    assert restored.inter_op_num_threads == 1


def test_setstate_accepts_state_without_session_options(tmp_path):
    session = PickableInferenceSession.__new__(PickableInferenceSession)
    session.__setstate__(
        {
            "model_path": _identity_model(tmp_path / "identity.onnx"),
            "providers": ["CPUExecutionProvider"],
            "provider_options": [{}],
        }
    )

    assert session.get_providers() == ["CPUExecutionProvider"]
