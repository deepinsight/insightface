import numpy as np
import pytest

from insightface.model_zoo.face_verifier import FaceVerifier


class _Meta:
    def __init__(self, name, shape, tensor_type):
        self.name = name
        self.shape = shape
        self.type = tensor_type


class _Session:
    def __init__(self, input_type="tensor(uint8)"):
        self.batch = None
        self.providers = None
        self.input_type = input_type

    def get_inputs(self):
        return [_Meta("input", ["batch", 3, 96, 96], self.input_type)]

    def get_outputs(self):
        return [_Meta("score", ["batch"], "tensor(float)")]

    def run(self, _outputs, inputs):
        self.batch = inputs["input"]
        return [np.asarray([0.75], dtype=np.float32)]

    def set_providers(self, providers):
        self.providers = providers


def test_face_verifier_has_explicit_task_and_uint8_batch_contract(tmp_path):
    model_file = tmp_path / "verifier.onnx"
    model_file.write_bytes(b"fake")
    session = _Session()
    verifier = FaceVerifier(
        model_file=model_file,
        session=session,
        expansion=1.3,
        preprocessing="embedded",
    )

    result = verifier.verify(
        np.full((40, 40, 3), 127, dtype=np.uint8),
        [[-5, -5, 15, 15]],
    )

    assert verifier.taskname == "verification"
    assert verifier.preprocessing == "embedded"
    assert not hasattr(verifier, "input_mean")
    assert not hasattr(verifier, "input_std")
    assert session.batch.shape == (1, 3, 96, 96)
    assert session.batch.dtype == np.uint8
    assert result == [{"face_probability": 0.75}]


def _channel_test_crop():
    crop = np.empty((96, 96, 3), dtype=np.uint8)
    crop[:, :, 0] = 10
    crop[:, :, 1] = 20
    crop[:, :, 2] = 30
    return crop


def test_embedded_preprocessing_accepts_float_input_and_preserves_raw_values(
    tmp_path,
):
    model_file = tmp_path / "verifier.onnx"
    model_file.write_bytes(b"fake")
    verifier = FaceVerifier(
        model_file=model_file,
        session=_Session("tensor(float)"),
        preprocessing="embedded",
    )

    batch = verifier.preprocess([_channel_test_crop()])

    assert batch.dtype == np.float32
    assert batch.flags.c_contiguous
    assert np.array_equal(batch[0, :, 0, 0], [30.0, 20.0, 10.0])


def test_mean_std_preprocessing_accepts_float_input_and_normalizes(tmp_path):
    model_file = tmp_path / "verifier.onnx"
    model_file.write_bytes(b"fake")
    verifier = FaceVerifier(
        model_file=model_file,
        session=_Session("tensor(float)"),
        preprocessing={"mean": 10.0, "std": 10.0},
    )

    batch = verifier.preprocess([_channel_test_crop()])

    assert batch.dtype == np.float32
    assert batch.flags.c_contiguous
    assert np.array_equal(batch[0, :, 0, 0], [2.0, 1.0, 0.0])


def test_mean_std_preprocessing_rejects_uint8_model_input(tmp_path):
    model_file = tmp_path / "verifier.onnx"
    model_file.write_bytes(b"fake")

    with pytest.raises(RuntimeError, match=r"mean/std.*input"):
        FaceVerifier(
            model_file=model_file,
            session=_Session("tensor(uint8)"),
            preprocessing={"mean": 0.0, "std": 1.0},
        )


def test_embedded_preprocessing_rejects_unsupported_model_input_type(tmp_path):
    model_file = tmp_path / "verifier.onnx"
    model_file.write_bytes(b"fake")

    with pytest.raises(RuntimeError, match="uint8 or float"):
        FaceVerifier(
            model_file=model_file,
            session=_Session("tensor(int64)"),
            preprocessing="embedded",
        )


def test_face_verifier_prepare_can_force_cpu(tmp_path):
    model_file = tmp_path / "verifier.onnx"
    model_file.write_bytes(b"fake")
    session = _Session()
    verifier = FaceVerifier(
        model_file=model_file,
        session=session,
        preprocessing="embedded",
    )

    verifier.prepare(-1)

    assert session.providers == ["CPUExecutionProvider"]


def test_face_verifier_revalidates_mutable_crop_expansion(tmp_path):
    model_file = tmp_path / "verifier.onnx"
    model_file.write_bytes(b"fake")
    verifier = FaceVerifier(
        model_file=model_file,
        session=_Session(),
        preprocessing="embedded",
    )
    verifier.crop_expansion = -1.0

    with pytest.raises(ValueError, match="expansion must be finite and positive"):
        verifier.verify(
            np.zeros((40, 40, 3), dtype=np.uint8),
            [[5, 5, 20, 20]],
        )


def test_face_verifier_preprocess_rejects_wrong_crop_shape(tmp_path):
    model_file = tmp_path / "verifier.onnx"
    model_file.write_bytes(b"fake")
    verifier = FaceVerifier(
        model_file=model_file,
        session=_Session(),
        preprocessing="embedded",
    )

    with pytest.raises(ValueError, match="must have shape"):
        verifier.preprocess([np.zeros((95, 96, 3), dtype=np.uint8)])


@pytest.mark.parametrize(
    ("preprocessing", "error_type"),
    [
        (None, TypeError),
        (1, TypeError),
        ("external", ValueError),
        ({"mean": 0.0}, ValueError),
        ({"mean": 0.0, "std": 0.0}, ValueError),
        ({"mean": True, "std": 1.0}, TypeError),
    ],
)
def test_face_verifier_rejects_invalid_preprocessing(
    tmp_path,
    preprocessing,
    error_type,
):
    model_file = tmp_path / "verifier.onnx"
    model_file.write_bytes(b"fake")

    with pytest.raises(error_type, match="preprocessing"):
        FaceVerifier(
            model_file=model_file,
            session=_Session("tensor(float)"),
            preprocessing=preprocessing,
        )
