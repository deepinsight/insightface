from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from insightface.app.privateframe import models as private_models

_COREML_PROVIDER = "CoreMLExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"


class _SessionOptions:
    def __init__(self) -> None:
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0
        self.log_severity_level = 0
        self.dimension_overrides: dict[str, int] = {}

    def add_free_dimension_override_by_name(
        self,
        name: str,
        value: int,
    ) -> None:
        self.dimension_overrides[name] = value


class _Session:
    def __init__(
        self,
        shape: list[int | str | None],
        *,
        providers: tuple[str, ...] = (_COREML_PROVIDER, _CPU_PROVIDER),
        run_error: Exception | None = None,
    ) -> None:
        self._shape = shape
        self._providers = providers
        self._run_error = run_error
        self.run_calls: list[
            tuple[
                Any,
                Any,
                tuple[Any, ...],
                dict[str, Any],
            ]
        ] = []
        self.run_result = object()

    def get_inputs(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                name="input.1",
                type="tensor(float)",
                shape=list(self._shape),
            )
        ]

    def get_providers(self) -> list[str]:
        return list(self._providers)

    def run(
        self,
        output_names: Any,
        input_feed: Any,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        self.run_calls.append((output_names, input_feed, args, kwargs))
        if self._run_error is not None:
            raise self._run_error
        return self.run_result


class _Model:
    def __init__(self, model_file: str, session: _Session) -> None:
        self.model_file = model_file
        self.session: Any = session
        self.nms_thresh = -1.0


class _Analysis:
    def __init__(self, models: dict[str, _Model]) -> None:
        self.models = models
        self.det_model = models["detection"]


class _InferenceSessionFactory:
    def __init__(
        self,
        *,
        actual_providers: tuple[str, ...] = (
            _COREML_PROVIDER,
            _CPU_PROVIDER,
        ),
        construct_failures: tuple[str, ...] = (),
        warmup_failures: tuple[str, ...] = (),
    ) -> None:
        self.actual_providers = actual_providers
        self.construct_failures = set(construct_failures)
        self.warmup_failures = set(warmup_failures)
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def __call__(
        self,
        model_file: str,
        *,
        sess_options: _SessionOptions,
        providers: list[str],
        provider_options: list[dict[str, str]],
    ) -> _Session:
        width = sess_options.dimension_overrides.get("width")
        height = sess_options.dimension_overrides.get("height")
        call = {
            "model_file": model_file,
            "size": (width, height),
            "providers": providers,
            "provider_options": provider_options,
            "sess_options": sess_options,
        }
        with self._lock:
            self.calls.append(call)
        compute_units = provider_options[0].get("MLComputeUnits")
        if compute_units in self.construct_failures:
            raise RuntimeError(f"{compute_units} construction failed")
        cache_directory = provider_options[0].get("ModelCacheDirectory")
        if cache_directory is not None:
            artifact = Path(cache_directory) / "compiled_model.mlmodelc"
            artifact.mkdir(parents=True, exist_ok=True)
        return _Session(
            [1, 3, height, width],
            providers=self.actual_providers,
            run_error=(
                RuntimeError(f"{compute_units} warmup failed")
                if compute_units in self.warmup_failures
                else None
            ),
        )


class _Package:
    name = "fake-package"
    tasks = {
        "detection": SimpleNamespace(),
        "verification": SimpleNamespace(),
        "recognition": SimpleNamespace(),
    }


def _runtime(*providers: str) -> dict[str, Any]:
    return {
        "providers": list(providers),
        "intra_op_threads": 3,
        "inter_op_threads": 2,
    }


def _config(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "models": {
            "name": "fake-package",
            "detection": {
                "nms_iou_threshold": 0.42,
                "max_detections": 5,
            },
        },
        "runtime": runtime,
        "recognition": {"mode": "all"},
    }


def _install_face_analysis_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    actual_providers: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], list[_Analysis]]:
    calls: list[dict[str, Any]] = []
    analyses: list[_Analysis] = []

    def construct(**kwargs: Any) -> _Analysis:
        calls.append(kwargs)
        options = kwargs["sess_options"]
        width = options.dimension_overrides.get("width", "width")
        height = options.dimension_overrides.get("height", "height")
        providers = (
            actual_providers
            if actual_providers is not None
            else tuple(kwargs["providers"])
        )
        models = {
            "detection": _Model(
                "detector.onnx",
                _Session([1, 3, height, width], providers=providers),
            )
        }
        if "verification" in kwargs["allowed_modules"]:
            models["verification"] = _Model(
                "verifier.onnx",
                _Session([None, 3, 128, 128], providers=providers),
            )
        if "recognition" in kwargs["allowed_modules"]:
            models["recognition"] = _Model(
                "recognizer.onnx",
                _Session([None, 3, 112, 112], providers=providers),
            )
        analysis = _Analysis(models)
        analyses.append(analysis)
        return analysis

    monkeypatch.setattr(private_models.ort, "SessionOptions", _SessionOptions)
    monkeypatch.setattr(
        private_models,
        "_load_selected_model_package",
        lambda _models: _Package(),
    )
    monkeypatch.setattr(private_models, "FaceAnalysis", construct)
    return calls, analyses


def test_coreml_session_options_override_detector_height_and_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(private_models.ort, "SessionOptions", _SessionOptions)

    options = private_models._session_options(
        _runtime(_COREML_PROVIDER, _CPU_PROVIDER),
        detector_input_size=(480, 256),
    )

    assert options.dimension_overrides == {"height": 256, "width": 480}
    assert options.intra_op_num_threads == 3
    assert options.inter_op_num_threads == 2
    assert options.log_severity_level == 3


@pytest.mark.parametrize(
    ("release", "model_format"),
    [("11.7", "NeuralNetwork"), ("12.0", "MLProgram"), ("26.0", "MLProgram")],
)
def test_coreml_model_format_follows_macos_version(
    monkeypatch: pytest.MonkeyPatch,
    release: str,
    model_format: str,
) -> None:
    monkeypatch.setattr(
        private_models.platform,
        "mac_ver",
        lambda: (release, ("", "", ""), "arm64"),
    )

    assert private_models._coreml_provider_options(
        static_shapes=True
    )["ModelFormat"] == model_format


def test_face_analysis_passes_static_resolution_factory_only_to_scrfd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        private_models.platform,
        "mac_ver",
        lambda: ("26.0", ("", "", ""), "arm64"),
    )
    calls, _analyses = _install_face_analysis_fakes(monkeypatch)
    config = _config(_runtime(_COREML_PROVIDER, _CPU_PROVIDER))
    config["scan"] = {"input_size": [480, 256]}

    analysis = private_models.make_face_analysis(config)

    assert len(calls) == 1
    call = calls[0]
    assert call["static_shape_sessions"] is True
    assert call["provider_options"] == [
        {
            "ModelFormat": "MLProgram",
            "MLComputeUnits": "ALL",
            "RequireStaticInputShapes": "0",
            "EnableOnSubgraphs": "0",
        },
        {},
    ]
    assert call["sess_options"].dimension_overrides == {
        "height": 256,
        "width": 480,
    }
    assert analysis.models["verification"].session.get_inputs()[0].shape == [
        None,
        3,
        128,
        128,
    ]
    resolution_factory = call["resolution_session_factory"]
    assert isinstance(
        resolution_factory,
        private_models._CoreMLResolutionSessionFactory,
    )
    assert isinstance(analysis.det_model.session, _Session)
    assert analysis.det_model.session.get_inputs()[0].shape == [1, 3, 256, 480]


def test_coreml_resolution_factory_creates_fresh_static_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inference_sessions = _InferenceSessionFactory()
    monkeypatch.setattr(private_models.ort, "SessionOptions", _SessionOptions)
    monkeypatch.setattr(
        private_models.ort,
        "InferenceSession",
        inference_sessions,
    )
    runtime = _runtime(_COREML_PROVIDER, _CPU_PROVIDER)
    factory = private_models._CoreMLResolutionSessionFactory(
        runtime,
        model_sha256="a" * 64,
        cache_root=tmp_path,
    )
    reference = _Session([1, 3, 320, 640])

    first = factory("detector.onnx", (384, 224), reference)
    second = factory("detector.onnx", (384, 224), reference)

    assert first is not second
    assert len(first.run_calls) == 1
    assert second.run_calls == []
    assert [call["size"] for call in inference_sessions.calls] == [
        (384, 224),
        (384, 224),
    ]
    assert (
        inference_sessions.calls[0]["sess_options"]
        is not inference_sessions.calls[1]["sess_options"]
    )
    for call in inference_sessions.calls:
        options = call["sess_options"]
        assert options.dimension_overrides == {"height": 224, "width": 384}
        assert options.intra_op_num_threads == 3
        assert options.inter_op_num_threads == 2
        assert options.log_severity_level == 3
        assert call["providers"] == [_COREML_PROVIDER, _CPU_PROVIDER]
        assert call["provider_options"][0]["RequireStaticInputShapes"] == "1"
        assert call["provider_options"][0]["MLComputeUnits"] == "ALL"
        cache_directory = Path(
            call["provider_options"][0]["ModelCacheDirectory"]
        )
        assert cache_directory.is_relative_to(tmp_path)
    assert (
        inference_sessions.calls[0]["provider_options"][0][
            "ModelCacheDirectory"
        ]
        == inference_sessions.calls[1]["provider_options"][0][
            "ModelCacheDirectory"
        ]
    )


@pytest.mark.parametrize(
    "failure_stage",
    ["construction", "warmup"],
)
def test_coreml_resolution_factory_falls_back_after_all_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str,
) -> None:
    inference_sessions = _InferenceSessionFactory(
        construct_failures=("ALL",) if failure_stage == "construction" else (),
        warmup_failures=("ALL",) if failure_stage == "warmup" else (),
    )
    monkeypatch.setattr(private_models.ort, "SessionOptions", _SessionOptions)
    monkeypatch.setattr(
        private_models.ort,
        "InferenceSession",
        inference_sessions,
    )
    factory = private_models._CoreMLResolutionSessionFactory(
        _runtime(_COREML_PROVIDER, _CPU_PROVIDER),
        model_sha256="b" * 64,
        cache_root=tmp_path,
    )

    session = factory(
        "detector.onnx",
        (384, 224),
        _Session([1, 3, 320, 640]),
    )

    assert [
        call["provider_options"][0]["MLComputeUnits"]
        for call in inference_sessions.calls
    ] == ["ALL", "CPUAndGPU"]
    assert len(session.run_calls) == 1
    assert Path(
        inference_sessions.calls[-1]["provider_options"][0][
            "ModelCacheDirectory"
        ]
    ).is_relative_to(tmp_path)


def test_resolution_factory_follows_prepare_cpu_provider_with_static_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inference_sessions = _InferenceSessionFactory(
        actual_providers=(_CPU_PROVIDER,)
    )
    monkeypatch.setattr(private_models.ort, "SessionOptions", _SessionOptions)
    monkeypatch.setattr(
        private_models.ort,
        "InferenceSession",
        inference_sessions,
    )
    factory = private_models._CoreMLResolutionSessionFactory(
        _runtime(_COREML_PROVIDER, _CPU_PROVIDER),
        model_sha256="c" * 64,
        cache_root=tmp_path,
    )
    cpu_reference = _Session(
        [1, 3, 320, 640],
        providers=(_CPU_PROVIDER,),
    )

    session = factory("detector.onnx", (384, 224), cpu_reference)

    assert session.get_providers() == [_CPU_PROVIDER]
    assert len(inference_sessions.calls) == 1
    call = inference_sessions.calls[0]
    assert call["providers"] == [_CPU_PROVIDER]
    assert call["provider_options"] == [{}]
    assert call["sess_options"].dimension_overrides == {
        "height": 224,
        "width": 384,
    }
    assert call["sess_options"].intra_op_num_threads == 3
    assert call["sess_options"].inter_op_num_threads == 2
    assert call["sess_options"].log_severity_level == 3
    assert not tmp_path.joinpath("c" * 64).exists()


def test_non_coreml_face_analysis_does_not_wrap_detector_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, analyses = _install_face_analysis_fakes(monkeypatch)
    config = _config(_runtime(_CPU_PROVIDER))
    config["scan"] = {"input_size": [480, 256]}

    analysis = private_models.make_face_analysis(config)

    assert analysis is analyses[0]
    assert isinstance(analysis.det_model.session, _Session)
    assert calls[0]["static_shape_sessions"] is True
    assert "resolution_session_factory" not in calls[0]
    assert "provider_options" not in calls[0]
    assert calls[0]["sess_options"].dimension_overrides == {}


def test_coreml_dynamic_fallback_uses_only_the_unconstrained_main_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, analyses = _install_face_analysis_fakes(monkeypatch)
    config = _config(_runtime(_COREML_PROVIDER, _CPU_PROVIDER))
    config["runtime"]["scrfd_static_shape_sessions"] = False
    config["scan"] = {"input_size": [480, 256]}

    analysis = private_models.make_face_analysis(config)

    assert analysis is analyses[0]
    call = calls[0]
    assert call["static_shape_sessions"] is False
    assert "resolution_session_factory" not in call
    assert call["sess_options"].dimension_overrides == {}
    assert call["provider_options"][0]["RequireStaticInputShapes"] == "0"
    assert analysis.det_model.session.get_inputs()[0].shape == [
        1,
        3,
        "height",
        "width",
    ]


def test_coreml_base_session_silent_cpu_fallback_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _calls, _analyses = _install_face_analysis_fakes(
        monkeypatch,
        actual_providers=(_CPU_PROVIDER,),
    )
    config = _config(_runtime(_COREML_PROVIDER, _CPU_PROVIDER))
    config["scan"] = {"input_size": [480, 256]}

    with pytest.raises(
        RuntimeError,
        match=(
            r"CoreMLExecutionProvider was requested as the primary provider, "
            r"but ONNX Runtime activated \['CPUExecutionProvider'\]"
        ),
    ):
        private_models.make_face_analysis(config)


def test_coreml_resolution_session_silent_cpu_fallback_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inference_sessions = _InferenceSessionFactory(
        actual_providers=(_CPU_PROVIDER,)
    )
    monkeypatch.setattr(private_models.ort, "SessionOptions", _SessionOptions)
    monkeypatch.setattr(
        private_models.ort,
        "InferenceSession",
        inference_sessions,
    )
    factory = private_models._CoreMLResolutionSessionFactory(
        _runtime(_COREML_PROVIDER, _CPU_PROVIDER),
        model_sha256="d" * 64,
        cache_root=tmp_path,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"failed to create a CoreML session .*"
            r"silently fell back.*CPUExecutionProvider"
        ),
    ):
        factory(
            "detector.onnx",
            (384, 224),
            _Session([1, 3, 320, 640]),
        )
    assert [
        call["provider_options"][0]["MLComputeUnits"]
        for call in inference_sessions.calls
    ] == ["ALL", "CPUAndGPU"]


def test_primary_and_review_use_independent_resolution_factories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls, _analyses = _install_face_analysis_fakes(monkeypatch)
    factory = _InferenceSessionFactory()
    monkeypatch.setattr(private_models.ort, "InferenceSession", factory)
    config = _config(_runtime(_COREML_PROVIDER, _CPU_PROVIDER))
    config["scan"] = {
        "passes": [
            {"input_size": [320, 240]},
            {"input_size": [640, 384]},
        ]
    }
    config["revalidation"] = {
        "passes": [
            {"input_size": [160, 160]},
            {"input_size": [256, 192]},
        ]
    }

    primary = private_models.make_face_analysis(config)
    review = private_models.make_review_face_analysis(config)

    primary_factory = calls[0]["resolution_session_factory"]
    review_factory = calls[1]["resolution_session_factory"]
    assert isinstance(
        primary_factory,
        private_models._CoreMLResolutionSessionFactory,
    )
    assert isinstance(
        review_factory,
        private_models._CoreMLResolutionSessionFactory,
    )
    assert primary_factory is not review_factory
    assert isinstance(primary.det_model.session, _Session)
    assert isinstance(review.det_model.session, _Session)
    assert primary.det_model.session.get_inputs()[0].shape == [1, 3, 384, 640]
    assert review.det_model.session.get_inputs()[0].shape == [1, 3, 192, 256]
    assert calls[0]["allowed_modules"] == ("detection", "verification")
    assert calls[1]["allowed_modules"] == ("detection",)

    primary_factory._cache_root = tmp_path
    review_factory._cache_root = tmp_path

    primary_target = primary_factory(
        primary.det_model.model_file,
        (400, 300),
        primary.det_model.session,
    )
    review_target = review_factory(
        review.det_model.model_file,
        (400, 300),
        review.det_model.session,
    )

    assert primary_target is not review_target
    assert Counter(call["size"] for call in factory.calls) == Counter({(400, 300): 2})
