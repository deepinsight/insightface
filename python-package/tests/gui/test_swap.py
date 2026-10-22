import numpy as np

from insightface.gui.core.swap import GFPGANRestorer


class _FakeSession:
    def __init__(self):
        self.input_shape = None

    def run(self, output_names, inputs):
        del output_names
        tensor = next(iter(inputs.values()))
        self.input_shape = tensor.shape
        return [np.zeros((1, 3, 512, 512), dtype=np.float32)]


def test_gfpgan_restorer_uses_512_input_and_restores_original_shape():
    restorer = GFPGANRestorer()
    restorer.session = _FakeSession()
    restorer.input_name = "input"
    restorer.output_name = "output"
    image = np.full((96, 128, 3), 127, dtype=np.uint8)

    output = restorer.restore(image)

    assert restorer.session.input_shape == (1, 3, 512, 512)
    assert output.shape == image.shape
