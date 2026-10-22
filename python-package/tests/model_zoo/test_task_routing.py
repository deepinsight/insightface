import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from insightface.model_zoo import model_zoo
from insightface.model_zoo.package_manifest import load_model_package


class _FakeSession:
    _providers = ["CPUExecutionProvider"]
    _provider_options = [{}]

    def __init__(self, path, **kwargs):
        self.path = str(path)
        self.kwargs = kwargs


class _InputMeta:
    def __init__(self, tensor_type):
        self.type = tensor_type


class _TypedFakeSession(_FakeSession):
    input_type = "tensor(float)"

    def get_inputs(self):
        return [_InputMeta(self.input_type)]


class _Adapter:
    def __init__(self, model_file=None, session=None, **kwargs):
        self.model_file = str(model_file)
        self.session = session
        self.kwargs = kwargs
        self.input_shape = [1, 3, 112, 112]
        self.input_mean = -1.0
        self.input_std = -1.0


class _Detector(_Adapter):
    taskname = "detection"


class _Recognizer(_Adapter):
    taskname = "recognition"


class _Verifier(_Adapter):
    taskname = "verification"


@pytest.fixture
def explicit_adapters(monkeypatch):
    monkeypatch.setattr(model_zoo, "PickableInferenceSession", _FakeSession)
    monkeypatch.setattr(model_zoo, "SCRFD", _Detector)
    monkeypatch.setattr(model_zoo, "ArcFaceONNX", _Recognizer)
    monkeypatch.setattr(model_zoo, "FaceVerifier", _Verifier)
    monkeypatch.setattr(
        model_zoo,
        "get_default_providers",
        lambda: ["CPUExecutionProvider"],
    )


def test_manifest_tasks_bypass_router_and_apply_metadata(
    manifest_package_factory,
    explicit_adapters,
    monkeypatch,
):
    package, _manifest = manifest_package_factory()
    monkeypatch.setattr(
        model_zoo.ModelRouter,
        "get_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("manifest task reached ModelRouter")
        ),
    )

    detector = model_zoo.get_model(package, model_task="detection")
    verifier = model_zoo.get_model(package, model_task="verification")
    recognizer = model_zoo.get_model(package, model_task="recognition")

    assert isinstance(detector, _Detector)
    assert detector.input_mean == 11.0
    assert detector.input_std == 22.0
    assert isinstance(verifier, _Verifier)
    assert verifier.kwargs["expansion"] == 1.3
    assert verifier.kwargs["preprocessing"] == "embedded"
    assert isinstance(recognizer, _Recognizer)
    assert recognizer.input_mean == 33.0
    assert recognizer.input_std == 44.0


def test_manifest_sha256_mismatch_fails_before_session_construction(
    manifest_package_factory,
    explicit_adapters,
    monkeypatch,
):
    package, manifest = manifest_package_factory()
    manifest["tasks"]["detection"]["sha256"] = "0" * 64
    (package / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    session_calls = []

    class UnexpectedSession:
        def __init__(self, *args, **kwargs):
            session_calls.append((args, kwargs))

    monkeypatch.setattr(model_zoo, "PickableInferenceSession", UnexpectedSession)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        model_zoo.get_model(package, model_task="detection")

    assert session_calls == []


def test_manifest_coreml_cache_signatures_isolate_task_sha_and_input_contract(
    manifest_package_factory,
    explicit_adapters,
    monkeypatch,
    tmp_path,
):
    package, _manifest = manifest_package_factory()
    descriptor_package = load_model_package(package)
    cache_root = tmp_path / "coreml-cache"
    contracts = {
        "detector.onnx": {
            "name": "detector-input",
            "dtype": "float32",
            "shape": [1, 3, 256, 480],
        },
        "verifier.onnx": {
            "name": "verifier-input",
            "dtype": "uint8",
            "shape": ["batch", 3, 128, 128],
        },
        "recognizer.onnx": {
            "name": "recognizer-input",
            "dtype": "float32",
            "shape": ["batch", 3, 112, 112],
        },
    }
    contract_calls = []

    def input_contracts(model_file, dimension_overrides=None):
        name = Path(model_file).name
        contract_calls.append((name, dict(dimension_overrides or {})))
        return [dict(contracts[name])]

    sessions = []

    class ManagedCoreMLSession:
        def __init__(self, path, **kwargs):
            self.path = str(path)
            self.kwargs = kwargs
            self.run_calls = []
            cache_directory = Path(kwargs["provider_options"][0]["ModelCacheDirectory"])
            (cache_directory / "compiled_model.mlmodelc").mkdir(
                parents=True,
                exist_ok=True,
            )
            sessions.append(self)

        def get_inputs(self):
            return [_InputMeta("tensor(float)")]

        def get_providers(self):
            return list(self.kwargs["providers"])

        def run(self, output_names, input_feed):
            self.run_calls.append((output_names, input_feed))
            return []

    monkeypatch.setattr(model_zoo, "_onnx_input_contracts", input_contracts)
    monkeypatch.setattr(
        model_zoo,
        "_detector_dimension_overrides",
        lambda _path, size: ({"height": size[1], "width": size[0]}),
    )
    monkeypatch.setattr(
        model_zoo,
        "PickableInferenceSession",
        ManagedCoreMLSession,
    )

    models = {}
    for task in ("detection", "verification", "recognition"):
        call_kwargs = {
            "model_descriptor": descriptor_package.task(task),
        }
        if task == "detection":
            # The public package+task route must resolve and forward the
            # descriptor before constructing any CoreML Session.
            call_kwargs = {}
        models[task] = model_zoo.get_model(
            package,
            model_task=task,
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
            provider_options=[
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "0",
                    "EnableOnSubgraphs": "0",
                },
                {},
            ],
            _coreml_cache_root=cache_root,
            _coreml_detector_input_size=(480, 256),
            **call_kwargs,
        )

    assert contract_calls == [
        ("detector.onnx", {"height": 256, "width": 480}),
        ("verifier.onnx", {}),
        ("recognizer.onnx", {}),
    ]
    assert len(sessions) == 3
    assert all(len(session.run_calls) == 1 for session in sessions)
    assert sessions[0].kwargs["provider_options"][0]["RequireStaticInputShapes"] == "1"
    assert sessions[1].kwargs["provider_options"][0]["RequireStaticInputShapes"] == "0"
    assert sessions[2].kwargs["provider_options"][0]["RequireStaticInputShapes"] == "0"

    cache_directories = {
        task: Path(model.session.coreml_cache_directory)
        for task, model in models.items()
    }
    assert len(set(cache_directories.values())) == 3
    assert all(path.is_relative_to(cache_root) for path in cache_directories.values())
    for task, cache_directory in cache_directories.items():
        signature_document = json.loads(
            (cache_directory / "signature.json").read_text(encoding="utf-8")
        )
        signature = signature_document["signature"]
        assert signature["task"] == task
        model_path = Path(models[task].model_file)
        assert (
            signature["model_sha256"]
            == hashlib.sha256(model_path.read_bytes()).hexdigest()
        )
        assert signature["inputs"] == [contracts[Path(models[task].model_file).name]]
        assert signature["coreml_options"]["MLComputeUnits"] == "ALL"


def test_scrfd_session_options_are_forwarded_only_to_manifest_detection(
    manifest_package_factory,
    explicit_adapters,
):
    package, _manifest = manifest_package_factory()
    resolution_factory = object()

    detector = model_zoo.get_model(
        package,
        model_task="detection",
        resolution_session_factory=resolution_factory,
        static_shape_sessions=False,
    )
    verifier = model_zoo.get_model(
        package,
        model_task="verification",
        resolution_session_factory=resolution_factory,
        static_shape_sessions=False,
    )
    recognizer = model_zoo.get_model(
        package,
        model_task="recognition",
        resolution_session_factory=resolution_factory,
        static_shape_sessions=False,
    )

    assert detector.kwargs["resolution_session_factory"] is resolution_factory
    assert detector.kwargs["static_shape_sessions"] is False
    assert "resolution_session_factory" not in detector.session.kwargs
    assert "static_shape_sessions" not in detector.session.kwargs
    assert "resolution_session_factory" not in verifier.kwargs
    assert "static_shape_sessions" not in verifier.kwargs
    assert "resolution_session_factory" not in verifier.session.kwargs
    assert "static_shape_sessions" not in verifier.session.kwargs
    assert "resolution_session_factory" not in recognizer.kwargs
    assert "static_shape_sessions" not in recognizer.kwargs
    assert "resolution_session_factory" not in recognizer.session.kwargs
    assert "static_shape_sessions" not in recognizer.session.kwargs


def test_legacy_model_router_forwards_factory_and_session_options_to_scrfd(
    monkeypatch,
):
    class RoutedSession(_FakeSession):
        def get_inputs(self):
            return [
                SimpleNamespace(
                    shape=[1, 3, "height", "width"],
                    type="tensor(float)",
                )
            ]

        def get_outputs(self):
            return [SimpleNamespace(name=f"output-{index}") for index in range(9)]

    monkeypatch.setattr(model_zoo, "PickableInferenceSession", RoutedSession)
    monkeypatch.setattr(model_zoo, "SCRFD", _Detector)
    resolution_factory = object()
    session_options = object()

    detector = model_zoo.ModelRouter("detector.onnx").get_model(
        providers=["CPUExecutionProvider"],
        provider_options=[{}],
        sess_options=session_options,
        resolution_session_factory=resolution_factory,
        static_shape_sessions=False,
    )

    assert detector.kwargs["resolution_session_factory"] is resolution_factory
    assert detector.kwargs["static_shape_sessions"] is False
    assert detector.session.kwargs["sess_options"] is session_options
    assert "resolution_session_factory" not in detector.session.kwargs
    assert "static_shape_sessions" not in detector.session.kwargs


def test_legacy_coreml_detector_uses_static_main_session_without_internal_flag(
    tmp_path,
    monkeypatch,
):
    model_file = tmp_path / "detector.onnx"
    model_file.write_bytes(b"legacy-scrfd")
    calls = []
    option_tokens = []

    class RoutedCoreMLSession(_FakeSession):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            self._providers = list(kwargs["providers"])
            self._provider_options = list(kwargs["provider_options"])

        def get_inputs(self):
            return [
                SimpleNamespace(
                    name="input.1",
                    shape=[1, 3, 640, 640],
                    type="tensor(float)",
                )
            ]

        def get_outputs(self):
            return [SimpleNamespace(name=f"output-{index}") for index in range(9)]

    def copy_options(source=None, dimension_overrides=None):
        token = SimpleNamespace(
            source=source,
            dimension_overrides=dict(dimension_overrides or {}),
        )
        option_tokens.append(token)
        return token

    def create_session(factory, source, **kwargs):
        calls.append((factory, source, kwargs))
        session = factory(
            source,
            providers=kwargs["providers"],
            provider_options=kwargs["provider_options"],
            sess_options=kwargs["sess_options_factory"](),
        )
        return SimpleNamespace(
            session=session,
            compute_units="ALL",
            cache_directory=tmp_path / "cache" / "signature",
            cache_hit=False,
        )

    monkeypatch.setattr(model_zoo, "PickableInferenceSession", RoutedCoreMLSession)
    monkeypatch.setattr(model_zoo, "SCRFD", _Detector)
    monkeypatch.setattr(model_zoo, "_onnx_is_detection_model", lambda _path: True)
    monkeypatch.setattr(
        model_zoo,
        "_detector_dimension_overrides",
        lambda _path, _size: {"?": 640},
    )
    monkeypatch.setattr(
        model_zoo,
        "_onnx_input_contracts",
        lambda _path, overrides: [
            {
                "name": "input.1",
                "dtype": "float32",
                "shape": [1, 3, overrides["?"], overrides["?"]],
            }
        ],
    )
    monkeypatch.setattr(model_zoo, "copy_session_options", copy_options)
    monkeypatch.setattr(model_zoo, "create_coreml_session", create_session)

    detector = model_zoo.get_model(
        model_file,
        providers=[
            (
                "CoreMLExecutionProvider",
                {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},
            ),
            "CPUExecutionProvider",
        ],
        _coreml_cache_root=tmp_path / "cache",
    )

    assert isinstance(detector, _Detector)
    assert len(calls) == 1
    _factory, _source, kwargs = calls[0]
    assert kwargs["providers"] == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert kwargs["provider_options"][0]["ModelFormat"] == "MLProgram"
    assert kwargs["provider_options"][0]["RequireStaticInputShapes"] == "1"
    assert kwargs["input_contracts"][0]["shape"] == [1, 3, 640, 640]
    assert option_tokens[0].dimension_overrides == {"?": 640}
    assert detector.session.kwargs["sess_options"] is option_tokens[0]
    assert detector.session.coreml_dimension_overrides == {"?": 640}


def test_legacy_coreml_dynamic_opt_out_forces_cpu_and_gpu_without_cache(
    tmp_path,
    monkeypatch,
):
    model_file = tmp_path / "detector.onnx"
    model_file.write_bytes(b"legacy-scrfd")
    sessions = []

    class RoutedCoreMLSession(_FakeSession):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            self._providers = list(kwargs["providers"])
            self._provider_options = list(kwargs["provider_options"])
            sessions.append(self)

        def get_inputs(self):
            return [
                SimpleNamespace(
                    name="input.1",
                    shape=[1, 3, "?", "?"],
                    type="tensor(float)",
                )
            ]

        def get_outputs(self):
            return [SimpleNamespace(name=f"output-{index}") for index in range(9)]

    monkeypatch.setattr(model_zoo, "PickableInferenceSession", RoutedCoreMLSession)
    monkeypatch.setattr(model_zoo, "SCRFD", _Detector)
    monkeypatch.setattr(model_zoo, "_onnx_is_detection_model", lambda _path: True)
    monkeypatch.setattr(
        model_zoo,
        "create_coreml_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dynamic SCRFD entered the CoreML cache manager")
        ),
    )
    session_options = object()
    original_coreml_options = {
        "ModelFormat": "MLProgram",
        "MLComputeUnits": "ALL",
        "RequireStaticInputShapes": "1",
    }

    detector = model_zoo.get_model(
        model_file,
        providers=[
            ("CoreMLExecutionProvider", original_coreml_options),
            "CPUExecutionProvider",
        ],
        sess_options=session_options,
        static_shape_sessions=False,
    )

    assert len(sessions) == 1
    assert detector.kwargs["static_shape_sessions"] is False
    assert detector.session.kwargs["sess_options"] is session_options
    assert (
        detector.session.kwargs["provider_options"][0]["MLComputeUnits"] == "CPUAndGPU"
    )
    assert (
        detector.session.kwargs["provider_options"][0]["RequireStaticInputShapes"]
        == "0"
    )
    assert original_coreml_options == {
        "ModelFormat": "MLProgram",
        "MLComputeUnits": "ALL",
        "RequireStaticInputShapes": "1",
    }


def test_manifest_coreml_dynamic_opt_out_avoids_cache_manager(
    manifest_package_factory,
    explicit_adapters,
    monkeypatch,
):
    package, _manifest = manifest_package_factory()
    descriptor = load_model_package(package).task("detection")
    monkeypatch.setattr(
        model_zoo,
        "create_coreml_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dynamic manifest SCRFD entered cache manager")
        ),
    )

    detector = model_zoo.get_model(
        package,
        model_task="detection",
        model_descriptor=descriptor,
        providers=[
            (
                "CoreMLExecutionProvider",
                {"MLComputeUnits": "ALL", "RequireStaticInputShapes": "1"},
            ),
            "CPUExecutionProvider",
        ],
        static_shape_sessions=False,
    )

    assert detector.kwargs["static_shape_sessions"] is False
    assert detector.session.kwargs["providers"] == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert (
        detector.session.kwargs["provider_options"][0]["MLComputeUnits"] == "CPUAndGPU"
    )
    assert (
        detector.session.kwargs["provider_options"][0]["RequireStaticInputShapes"]
        == "0"
    )


def test_manifest_non_detector_preserves_coreml_provider_tuple_options(
    manifest_package_factory,
    explicit_adapters,
    monkeypatch,
):
    package, _manifest = manifest_package_factory()
    descriptor = load_model_package(package).task("recognition")
    captured = {}

    def create_session(factory, source, **kwargs):
        del factory, source
        captured.update(kwargs)
        return SimpleNamespace(
            session=_TypedFakeSession("recognizer.onnx"),
            compute_units="CPUAndGPU",
            cache_directory=None,
            cache_hit=False,
        )

    monkeypatch.setattr(
        model_zoo,
        "_onnx_input_contracts",
        lambda _path: [
            {
                "name": "input.1",
                "dtype": "float32",
                "shape": ["batch", 3, 112, 112],
            }
        ],
    )
    monkeypatch.setattr(model_zoo, "create_coreml_session", create_session)

    model_zoo.get_model(
        package,
        model_task="recognition",
        model_descriptor=descriptor,
        providers=[
            (
                "CoreMLExecutionProvider",
                {
                    "MLComputeUnits": "CPUAndGPU",
                    "ModelFormat": "NeuralNetwork",
                    "EnableOnSubgraphs": "1",
                },
            ),
            "CPUExecutionProvider",
        ],
    )

    assert captured["providers"] == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert captured["provider_options"] == [
        {
            "MLComputeUnits": "CPUAndGPU",
            "ModelFormat": "NeuralNetwork",
            "EnableOnSubgraphs": "1",
        },
        {},
    ]


def test_static_coreml_detector_provider_defaults_match_resolution_sessions(
    monkeypatch,
):
    monkeypatch.setattr(
        model_zoo.platform,
        "mac_ver",
        lambda: ("14.5", ("", "", ""), ""),
    )

    values = model_zoo._coreml_detection_session_values(
        {
            "providers": [
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ],
            "provider_options": None,
        },
        static_shapes=True,
    )

    assert values["provider_options"][0] == {
        "ModelFormat": "MLProgram",
        "EnableOnSubgraphs": "0",
        "RequireStaticInputShapes": "1",
    }


def test_buffalo_shared_spatial_symbol_supports_square_coreml_session(
    monkeypatch,
):
    dimensions = [
        SimpleNamespace(dim_value=1, dim_param=""),
        SimpleNamespace(dim_value=3, dim_param=""),
        SimpleNamespace(dim_value=0, dim_param="?"),
        SimpleNamespace(dim_value=0, dim_param="?"),
    ]
    model = SimpleNamespace(
        graph=SimpleNamespace(
            input=[
                SimpleNamespace(
                    type=SimpleNamespace(
                        tensor_type=SimpleNamespace(
                            shape=SimpleNamespace(dim=dimensions)
                        )
                    )
                )
            ]
        )
    )
    monkeypatch.setattr(model_zoo.onnx, "load_model", lambda *_args, **_kwargs: model)

    assert model_zoo._detector_dimension_overrides(
        "buffalo-detector.onnx",
        (640, 640),
    ) == {"?": 640}
    with pytest.raises(ValueError, match="share the symbolic dimension"):
        model_zoo._detector_dimension_overrides(
            "buffalo-detector.onnx",
            (640, 480),
        )


@pytest.mark.parametrize("task", ["detection", "recognition"])
@pytest.mark.parametrize(
    ("preprocessing", "expected_mean", "expected_std"),
    [
        ("embedded", 0.0, 1.0),
        ({"mean": 19.5, "std": 27.25}, 19.5, 27.25),
    ],
)
def test_detection_and_recognition_apply_both_preprocessing_modes(
    manifest_package_factory,
    explicit_adapters,
    task,
    preprocessing,
    expected_mean,
    expected_std,
):
    package, manifest = manifest_package_factory()
    manifest["tasks"][task]["preprocessing"] = preprocessing
    (package / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    model = model_zoo.get_model(package, model_task=task)

    assert model.input_mean == expected_mean
    assert model.input_std == expected_std


@pytest.mark.parametrize(
    "preprocessing",
    [
        "embedded",
        {"mean": 51.0, "std": 17.0},
    ],
)
def test_verification_receives_both_preprocessing_modes(
    manifest_package_factory,
    explicit_adapters,
    preprocessing,
):
    package, manifest = manifest_package_factory()
    manifest["tasks"]["verification"]["preprocessing"] = preprocessing
    (package / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    verifier = model_zoo.get_model(package, model_task="verification")

    assert verifier.kwargs["preprocessing"] == preprocessing


def test_manifest_package_requires_explicit_task(
    manifest_package_factory,
    explicit_adapters,
):
    package, _manifest = manifest_package_factory()

    with pytest.raises(ValueError, match="explicit model_task"):
        model_zoo.get_model(package)


def test_verifier_is_never_shape_routed(
    manifest_package_factory,
    explicit_adapters,
    monkeypatch,
):
    package, _manifest = manifest_package_factory()
    calls = []

    def forbidden_router(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("verifier used heuristic routing")

    monkeypatch.setattr(model_zoo.ModelRouter, "get_model", forbidden_router)

    verifier = model_zoo.get_model(package, model_task="verification")

    assert isinstance(verifier, _Verifier)
    assert calls == []


@pytest.mark.parametrize("task", ["detection", "verification", "recognition"])
@pytest.mark.parametrize(
    ("metadata", "error_type"),
    [
        ({}, ValueError),
        ({"preprocessing": None}, TypeError),
        ({"preprocessing": "external"}, ValueError),
        ({"preprocessing": {"mean": 0.0}}, ValueError),
        ({"preprocessing": {"mean": 0.0, "std": 0.0}}, ValueError),
        (
            {"preprocessing": {"mean": 0.0, "std": 1.0, "extra": True}},
            ValueError,
        ),
    ],
)
def test_direct_tasks_reject_invalid_preprocessing_before_session(
    tmp_path,
    explicit_adapters,
    monkeypatch,
    task,
    metadata,
    error_type,
):
    model_file = tmp_path / f"{task}.onnx"
    model_file.write_bytes(task.encode())
    monkeypatch.setattr(
        model_zoo,
        "PickableInferenceSession",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid preprocessing created a Session")
        ),
    )

    with pytest.raises(error_type, match="preprocessing"):
        model_zoo.get_model(
            model_file,
            model_task=task,
            model_metadata=metadata,
        )


@pytest.mark.parametrize("task", ["detection", "recognition"])
@pytest.mark.parametrize(
    ("preprocessing", "expected_mean", "expected_std"),
    [
        ("embedded", 0.0, 1.0),
        ({"mean": 7.0, "std": 9.0}, 7.0, 9.0),
    ],
)
def test_direct_detection_and_recognition_support_both_preprocessing_modes(
    tmp_path,
    explicit_adapters,
    task,
    preprocessing,
    expected_mean,
    expected_std,
):
    model_file = tmp_path / f"{task}.onnx"
    model_file.write_bytes(task.encode())

    model = model_zoo.get_model(
        model_file,
        model_task=task,
        model_metadata={"preprocessing": preprocessing},
    )

    assert model.input_mean == expected_mean
    assert model.input_std == expected_std


@pytest.mark.parametrize("task", ["detection", "recognition"])
def test_embedded_detection_and_recognition_configure_uint8_input(
    tmp_path,
    explicit_adapters,
    monkeypatch,
    task,
):
    model_file = tmp_path / f"{task}.onnx"
    model_file.write_bytes(task.encode())

    class Uint8Session(_TypedFakeSession):
        input_type = "tensor(uint8)"

    monkeypatch.setattr(model_zoo, "PickableInferenceSession", Uint8Session)

    model = model_zoo.get_model(
        model_file,
        model_task=task,
        model_metadata={"preprocessing": "embedded"},
    )

    assert model.input_mean == 0.0
    assert model.input_std == 1.0
    assert model.input_type == "tensor(uint8)"
    assert model.input_dtype is np.uint8


@pytest.mark.parametrize("task", ["detection", "recognition"])
def test_mean_std_detection_and_recognition_reject_uint8_input(
    tmp_path,
    explicit_adapters,
    monkeypatch,
    task,
):
    model_file = tmp_path / f"{task}.onnx"
    model_file.write_bytes(task.encode())

    class Uint8Session(_TypedFakeSession):
        input_type = "tensor(uint8)"

    monkeypatch.setattr(model_zoo, "PickableInferenceSession", Uint8Session)

    with pytest.raises(RuntimeError, match=r"mean/std.*tensor\(float\)"):
        model_zoo.get_model(
            model_file,
            model_task=task,
            model_metadata={
                "preprocessing": {"mean": 0.0, "std": 1.0},
            },
        )


@pytest.mark.parametrize(
    "preprocessing",
    ["embedded", {"mean": 7.0, "std": 9.0}],
)
def test_direct_verification_supports_both_preprocessing_modes(
    tmp_path,
    explicit_adapters,
    preprocessing,
):
    model_file = tmp_path / "verification.onnx"
    model_file.write_bytes(b"verification")

    verifier = model_zoo.get_model(
        model_file,
        model_task="verification",
        model_metadata={"preprocessing": preprocessing, "expansion": 1.2},
    )

    assert verifier.kwargs["preprocessing"] == preprocessing
    assert verifier.kwargs["expansion"] == 1.2


def test_manifest_task_resolution_does_not_hash_model_content(
    manifest_package_factory,
    explicit_adapters,
):
    package, _manifest = manifest_package_factory()
    (package / "verifier.onnx").write_bytes(b"corrupt")

    assert isinstance(
        model_zoo.get_model(package, model_task="detection"),
        _Detector,
    )
    assert isinstance(
        model_zoo.get_model(package, model_task="verification"),
        _Verifier,
    )


def test_descriptor_route_does_not_reparse_manifest(
    manifest_package_factory,
    explicit_adapters,
    monkeypatch,
):
    package, _manifest = manifest_package_factory()
    descriptor = load_model_package(package).task("detection")
    monkeypatch.setattr(
        model_zoo,
        "load_model_package",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cached descriptor reparsed its manifest")
        ),
    )

    detector = model_zoo.get_model(
        package,
        model_task="detection",
        model_descriptor=descriptor,
        providers=["CPUExecutionProvider"],
    )

    assert isinstance(detector, _Detector)
    assert detector.input_mean == 11.0


def test_legacy_named_directory_keeps_last_onnx_routing(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "legacy"
    model_dir.mkdir(parents=True)
    (model_dir / "a.onnx").write_bytes(b"a")
    (model_dir / "z.onnx").write_bytes(b"z")
    routed = []

    class Router:
        def __init__(self, path):
            routed.append(Path(path).name)

        def get_model(self, **kwargs):
            return kwargs

    monkeypatch.setattr(model_zoo, "ModelRouter", Router)

    session_options = object()
    resolution_factory = object()
    result = model_zoo.get_model(
        "legacy",
        root=tmp_path,
        providers=["CPUExecutionProvider"],
        sess_options=session_options,
        resolution_session_factory=resolution_factory,
        static_shape_sessions=False,
    )

    assert routed == ["z.onnx"]
    assert result["providers"] == ["CPUExecutionProvider"]
    assert result["sess_options"] is session_options
    assert result["resolution_session_factory"] is resolution_factory
    assert result["static_shape_sessions"] is False


def test_direct_onnx_inside_manifest_package_remains_legacy_without_task(
    manifest_package_factory,
    monkeypatch,
):
    package, _manifest = manifest_package_factory()
    routed = []

    class Router:
        def __init__(self, path):
            routed.append(Path(path).name)

        def get_model(self, **_kwargs):
            return "legacy-model"

    monkeypatch.setattr(model_zoo, "ModelRouter", Router)

    result = model_zoo.get_model(package / "verifier.onnx")

    assert result == "legacy-model"
    assert routed == ["verifier.onnx"]
