from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from insightface.model_zoo.retinaface import RetinaFace
from insightface.model_zoo.scrfd import DEFAULT_DET_SIZES, SCRFD
from insightface.gui.core.face_engine import FaceEngine


def test_scrfd_accepts_multi_det_size_config():
    assert SCRFD._normalize_input_sizes([(128, 128), (640, 640)]) == [(128, 128), (640, 640)]
    assert SCRFD._normalize_input_sizes((320, 320)) == [(320, 320)]
    assert SCRFD._normalize_input_sizes((0, 0)) == DEFAULT_DET_SIZES


def test_scrfd_default_dynamic_det_size_is_auto():
    detector = SCRFD.__new__(SCRFD)
    detector.input_sizes = []
    detector.input_size = None

    assert detector._resolve_input_sizes(None) == DEFAULT_DET_SIZES


def test_retinaface_accepts_multi_det_size_config():
    assert RetinaFace._normalize_input_sizes([(128, 128), (640, 640)]) == [(128, 128), (640, 640)]
    assert RetinaFace._normalize_input_sizes([1024, 1024]) == [(1024, 1024)]


def test_model_zoo_routes_detection_models_to_scrfd(monkeypatch):
    class FakeInput:
        name = "input"
        shape = [1, 3, "height", "width"]

    class FakeOutput:
        name = "output"
        shape = [1, 10]

    class FakeSession:
        _providers = ["CPUExecutionProvider"]
        _provider_options = [{}]

        def __init__(self, *args, **kwargs):
            pass

        def get_inputs(self):
            return [FakeInput()]

        def get_outputs(self):
            return [FakeOutput() for _ in range(9)]

    monkeypatch.setattr(model_zoo, "PickableInferenceSession", FakeSession)

    model = model_zoo.ModelRouter("fake_detection.onnx").get_model()

    assert isinstance(model, SCRFD)


def test_gui_auto_detection_size_passes_multi_scale_to_detector():
    engine = FaceEngine(det_size=(0, 0))

    assert engine._detector_input_size() == [(128, 128), (640, 640)]


def test_face_analysis_prepare_defaults_to_auto_det_size():
    class FakeDetectionModel:
        taskname = "detection"

        def __init__(self):
            self.ctx_id = None
            self.kwargs = None

        def prepare(self, ctx_id, **kwargs):
            self.ctx_id = ctx_id
            self.kwargs = kwargs

    detector = FakeDetectionModel()
    app = FaceAnalysis.__new__(FaceAnalysis)
    app.models = {"detection": detector}
    app.addons = {}

    app.prepare()

    assert detector.ctx_id == 0
    assert app.det_size == DEFAULT_DET_SIZES
    assert detector.kwargs["input_size"] == DEFAULT_DET_SIZES
