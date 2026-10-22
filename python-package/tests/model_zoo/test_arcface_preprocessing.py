import numpy as np

from insightface.model_zoo.arcface_onnx import ArcFaceONNX


class _Session:
    def __init__(self):
        self.batch = None

    def run(self, _outputs, inputs):
        self.batch = inputs["input"]
        return [np.zeros((len(self.batch), 2), dtype=np.float32)]


def _recognizer(input_dtype, mean, std):
    model = ArcFaceONNX.__new__(ArcFaceONNX)
    model.input_size = (2, 2)
    model.input_mean = mean
    model.input_std = std
    model.input_dtype = input_dtype
    model.input_name = "input"
    model.output_names = ["output"]
    model.session = _Session()
    return model


def test_arcface_embedded_preprocessing_sends_uint8_rgb_blob():
    model = _recognizer(np.uint8, 0.0, 1.0)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[:, :, 0] = 10
    image[:, :, 1] = 20
    image[:, :, 2] = 30

    model.get_feat(image)

    assert model.session.batch.dtype == np.uint8
    assert model.session.batch.flags.c_contiguous
    assert np.array_equal(model.session.batch[0, :, 0, 0], [30, 20, 10])


def test_arcface_mean_std_preprocessing_sends_float32_blob():
    model = _recognizer(np.float32, 10.0, 10.0)
    image = np.full((2, 2, 3), 30, dtype=np.uint8)

    model.get_feat(image)

    assert model.session.batch.dtype == np.float32
    assert np.allclose(model.session.batch, 2.0)


def test_arcface_forward_coerces_embedded_batch_to_uint8():
    model = _recognizer(np.uint8, 0.0, 1.0)

    model.forward(np.full((1, 3, 2, 2), 40, dtype=np.float32))

    assert model.session.batch.dtype == np.uint8
    assert np.all(model.session.batch == 40)
