import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from insightface.app import face_analysis
from insightface.app.face_analysis import FaceAnalysis


_MODEL_FILES = {
    "detection": "detector.onnx",
    "verification": "verifier.onnx",
    "recognition": "recognizer.onnx",
}


class _Model:
    def __init__(self, task):
        self.task = task
        self.model_file = _MODEL_FILES[task]
        self.input_shape = [1, 3, 112, 112]
        self.input_mean = 0.0
        self.input_std = 1.0
        self.taskname = task


def test_constructor_preserves_positional_arguments_and_adds_keyword_only_addons():
    assert list(inspect.signature(FaceAnalysis).parameters) == [
        "name",
        "root",
        "allowed_modules",
        "addons",
        "liveness_mode",
        "liveness_threshold",
        "kwargs",
    ]
    for name in ("addons", "liveness_mode", "liveness_threshold"):
        assert inspect.signature(FaceAnalysis).parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(FaceAnalysis).parameters["liveness_mode"].default == "normal"


def test_manifest_coreml_enables_managed_cache_without_public_api_change(
    manifest_package_factory,
    monkeypatch,
    tmp_path,
):
    package, _manifest = manifest_package_factory()
    calls = []

    def get_model(_name, **kwargs):
        calls.append(kwargs)
        return _Model(kwargs["model_task"])

    monkeypatch.setattr(face_analysis.model_zoo, "get_model", get_model)

    analysis = FaceAnalysis(
        package,
        allowed_modules=("detection", "verification"),
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        _coreml_cache_root=tmp_path,
    )

    assert set(analysis.models) == {"detection", "verification"}
    assert [call["model_task"] for call in calls] == [
        "detection",
        "verification",
    ]
    assert all("_coreml_managed_cache" not in call for call in calls)
    assert all(call["_coreml_cache_root"] == tmp_path for call in calls)
    assert all(call["_coreml_detector_input_size"] == (640, 640) for call in calls)
    assert list(inspect.signature(FaceAnalysis).parameters) == [
        "name",
        "root",
        "allowed_modules",
        "addons",
        "liveness_mode",
        "liveness_threshold",
        "kwargs",
    ]


def test_manifest_cpu_does_not_enable_coreml_cache(
    manifest_package_factory,
    monkeypatch,
    tmp_path,
):
    package, _manifest = manifest_package_factory()
    cache_root = tmp_path / "coreml-cache"
    calls = []

    def get_model(_name, **kwargs):
        calls.append(kwargs)
        return _Model(kwargs["model_task"])

    monkeypatch.setattr(face_analysis.model_zoo, "get_model", get_model)

    FaceAnalysis(
        package,
        allowed_modules=("detection",),
        providers=["CPUExecutionProvider"],
        _coreml_cache_root=cache_root,
    )

    assert len(calls) == 1
    assert "_coreml_managed_cache" not in calls[0]
    assert not cache_root.exists()


def test_face_analysis_selects_default_providers_once(monkeypatch, tmp_path):
    directory = tmp_path / "legacy"
    directory.mkdir()
    (directory / "detector.onnx").write_bytes(b"fake")
    detector = _Model("detection")
    selected = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    selections = []
    calls = []
    monkeypatch.setattr(
        face_analysis,
        "get_default_providers",
        lambda: selections.append(True) or selected,
    )
    monkeypatch.setattr(
        face_analysis.model_zoo,
        "get_model",
        lambda _path, **kwargs: calls.append(kwargs) or detector,
    )

    FaceAnalysis(directory)

    assert selections == [True]
    assert calls[0]["providers"] is selected
    assert "_coreml_managed_cache" not in calls[0]
    assert calls[0]["_coreml_detector_input_size"] == (640, 640)


def test_face_analysis_preserves_explicit_providers(monkeypatch, tmp_path):
    directory = tmp_path / "legacy"
    directory.mkdir()
    (directory / "detector.onnx").write_bytes(b"fake")
    detector = _Model("detection")
    explicit = ["CPUExecutionProvider"]
    calls = []
    monkeypatch.setattr(
        face_analysis,
        "get_default_providers",
        lambda: (_ for _ in ()).throw(AssertionError("auto selection ran")),
    )
    monkeypatch.setattr(
        face_analysis.model_zoo,
        "get_model",
        lambda _path, **kwargs: calls.append(kwargs) or detector,
    )

    FaceAnalysis(directory, providers=explicit)

    assert calls[0]["providers"] is explicit
    assert "_coreml_managed_cache" not in calls[0]
    assert "_coreml_detector_input_size" not in calls[0]


def test_coreml_provider_tuple_and_dynamic_opt_out_preserve_public_api(
    manifest_package_factory,
    monkeypatch,
):
    package, _manifest = manifest_package_factory()
    calls = []

    def get_model(_name, **kwargs):
        calls.append(kwargs)
        return _Model(kwargs["model_task"])

    monkeypatch.setattr(face_analysis.model_zoo, "get_model", get_model)
    providers = [
        ("CoreMLExecutionProvider", {"MLComputeUnits": "ALL"}),
        "CPUExecutionProvider",
    ]

    FaceAnalysis(
        package,
        allowed_modules=("detection",),
        providers=providers,
        static_shape_sessions=False,
    )

    assert calls[0]["providers"] is providers
    assert "_coreml_managed_cache" not in calls[0]
    assert "_coreml_detector_input_size" not in calls[0]
    assert calls[0]["static_shape_sessions"] is False
    assert list(inspect.signature(FaceAnalysis).parameters) == [
        "name",
        "root",
        "allowed_modules",
        "addons",
        "liveness_mode",
        "liveness_threshold",
        "kwargs",
    ]


def test_existing_model_directory_avoids_ensure_available(tmp_path, monkeypatch):
    directory = tmp_path / "legacy"
    directory.mkdir()
    (directory / "detector.onnx").write_bytes(b"fake")
    detector = _Model("detection")
    monkeypatch.setattr(
        face_analysis,
        "ensure_available",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("existing directory triggered a download")
        ),
    )
    monkeypatch.setattr(
        face_analysis.model_zoo,
        "get_model",
        lambda *_args, **_kwargs: detector,
    )

    analysis = FaceAnalysis(directory)

    assert analysis.model_dir == str(directory.resolve())
    assert analysis.models == {"detection": detector}
    assert analysis.det_model is detector


def test_manifest_default_eagerly_loads_every_described_module(
    manifest_package_factory,
    monkeypatch,
):
    package, _manifest = manifest_package_factory()
    calls = []

    def get_model(_name, **kwargs):
        task = kwargs["model_task"]
        calls.append(task)
        return _Model(task)

    monkeypatch.setattr(face_analysis.model_zoo, "get_model", get_model)

    analysis = FaceAnalysis(package)

    assert calls == ["detection", "verification", "recognition"]
    assert set(analysis.models) == {
        "detection",
        "verification",
        "recognition",
    }


@pytest.mark.parametrize(
    ("allowed_modules", "expected_tasks"),
    [
        (("detection",), ("detection",)),
        (
            ("detection", "verification"),
            ("detection", "verification"),
        ),
        (("detection", "recognition"), ("detection", "recognition")),
        (
            ("detection", "verification", "recognition"),
            ("detection", "verification", "recognition"),
        ),
    ],
)
def test_allowed_modules_is_the_only_manifest_task_loading_control(
    manifest_package_factory,
    monkeypatch,
    allowed_modules,
    expected_tasks,
):
    package, _manifest = manifest_package_factory()
    calls = []

    def get_model(_name, **kwargs):
        task = kwargs["model_task"]
        calls.append(task)
        return _Model(task)

    monkeypatch.setattr(face_analysis.model_zoo, "get_model", get_model)

    analysis = FaceAnalysis(
        package,
        allowed_modules=allowed_modules,
    )

    assert calls == list(expected_tasks)
    assert set(analysis.models) == set(expected_tasks)


def test_invalid_manifest_fails_before_any_model_is_built(
    manifest_package_factory,
    monkeypatch,
):
    package, manifest = manifest_package_factory()
    manifest["tasks"]["detection"].pop("file")
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        face_analysis.model_zoo,
        "get_model",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(ValueError, match=r"tasks\.detection\.file is required"):
        FaceAnalysis(package)
    assert calls == []


def test_manifest_tasks_reuse_one_descriptor_parse_during_construction(
    manifest_package_factory,
    monkeypatch,
):
    package, _manifest = manifest_package_factory()
    calls = []
    descriptors = {}
    parse_calls = 0
    real_load_model_package = face_analysis.load_model_package

    def load_model_package(path):
        nonlocal parse_calls
        parse_calls += 1
        return real_load_model_package(path)

    def get_model(name, **kwargs):
        task = kwargs["model_task"]
        calls.append(task)
        descriptors[task] = kwargs["model_descriptor"]
        assert Path(name) == package
        return _Model(task)

    monkeypatch.setattr(face_analysis, "load_model_package", load_model_package)
    monkeypatch.setattr(face_analysis.model_zoo, "get_model", get_model)
    analysis = FaceAnalysis(
        package,
        allowed_modules=("detection", "verification", "recognition"),
    )

    assert calls == ["detection", "verification", "recognition"]
    assert parse_calls == 1
    assert set(analysis.models) == {
        "detection",
        "verification",
        "recognition",
    }
    for task in calls:
        assert descriptors[task] is analysis.model_package.task(task)


def test_legacy_face_analysis_get_keeps_existing_detector_contract(
    tmp_path,
    monkeypatch,
):
    directory = tmp_path / "legacy"
    directory.mkdir()
    detector_path = directory / "a_detector.onnx"
    recognizer_path = directory / "b_recognizer.onnx"
    detector_path.write_bytes(b"detector")
    recognizer_path.write_bytes(b"recognizer")
    detect_calls = []

    class Detector(_Model):
        def __init__(self):
            super().__init__("detection")

        def detect(self, image, max_num=0, metric="default"):
            detect_calls.append((image, max_num, metric))
            return (
                np.asarray([[1, 2, 5, 6, 0.75]], dtype=np.float32),
                np.asarray([[[1, 2], [3, 4], [2, 3], [1, 4], [4, 4]]]),
            )

    class Recognizer(_Model):
        def __init__(self):
            super().__init__("recognition")

        def get(self, _image, face):
            face.embedding = np.asarray([3.0, 4.0], dtype=np.float32)

    detector = Detector()
    recognizer = Recognizer()

    def get_model(path, **_kwargs):
        return detector if Path(path) == detector_path else recognizer

    monkeypatch.setattr(face_analysis.model_zoo, "get_model", get_model)
    analysis = FaceAnalysis(directory)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    faces = analysis.get(image, max_num=2, det_metric="max")

    assert len(faces) == 1
    assert faces[0].det_score == pytest.approx(0.75)
    assert np.array_equal(faces[0].embedding, [3.0, 4.0])
    assert detect_calls == [(image, 2, "max")]


def test_manifest_get_uses_selected_recognizer_loaded_by_constructor(
    manifest_package_factory,
    monkeypatch,
):
    package, _manifest = manifest_package_factory()
    calls = []

    class Detector(_Model):
        def __init__(self):
            super().__init__("detection")

        def detect(self, _image, max_num=0, metric="default"):
            assert max_num == 0
            assert metric == "default"
            return (
                np.asarray([[1, 2, 5, 6, 0.75]], dtype=np.float32),
                np.asarray(
                    [[[1, 2], [3, 4], [2, 3], [1, 4], [4, 4]]],
                    dtype=np.float32,
                ),
            )

    class Recognizer(_Model):
        def __init__(self):
            super().__init__("recognition")
            self.get_calls = 0

        def get(self, _image, face):
            self.get_calls += 1
            face.embedding = np.asarray([1.0, 2.0], dtype=np.float32)

    detector = Detector()
    recognizer = Recognizer()

    def get_model(_name, **kwargs):
        task = kwargs["model_task"]
        calls.append(task)
        return detector if task == "detection" else recognizer

    monkeypatch.setattr(face_analysis.model_zoo, "get_model", get_model)
    analysis = FaceAnalysis(
        package,
        allowed_modules={"detection", "recognition"},
    )

    assert calls == ["detection", "recognition"]

    faces = analysis.get(np.zeros((8, 8, 3), dtype=np.uint8))

    assert calls == ["detection", "recognition"]
    assert recognizer.get_calls == 1
    assert np.array_equal(faces[0].embedding, [1.0, 2.0])
