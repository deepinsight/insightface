import copy
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np
import onnx
import pytest
from insightface.model_zoo import coreml_cache as coreml_cache_module
from insightface.model_zoo import scrfd as scrfd_module
from insightface.model_zoo.scrfd import SCRFD


class _Metadata:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _FakeSession:
    def __init__(
        self,
        input_shape=(1, 3, "height", "width"),
        providers=("CPUExecutionProvider",),
        provider_options=None,
        session_options=None,
        output_count=6,
    ):
        self.input_shape = list(input_shape)
        self.providers = list(providers)
        self.provider_options = (
            {provider: {} for provider in self.providers}
            if provider_options is None
            else dict(provider_options)
        )
        self.session_options = (
            object() if session_options is None else session_options
        )
        self.output_count = int(output_count)
        self.runs = []

    def get_inputs(self):
        return [_Metadata("input", list(self.input_shape))]

    def get_outputs(self):
        return [
            _Metadata(f"output_{index}", [1])
            for index in range(self.output_count)
        ]

    def get_providers(self):
        return list(self.providers)

    def get_provider_options(self):
        return dict(self.provider_options)

    def get_session_options(self):
        return self.session_options

    def set_providers(self, providers):
        self.providers = list(providers)
        self.provider_options = {provider: {} for provider in self.providers}

    def run(self, _output_names, input_feed):
        blob = np.asarray(input_feed["input"])
        self.runs.append(tuple(blob.shape))
        height, width = blob.shape[2:4]
        scores = []
        boxes = []
        for stride in (8, 16, 32):
            count = (height // stride) * (width // stride) * 2
            scores.append(np.zeros((count, 1), dtype=np.float32))
            boxes.append(np.zeros((count, 4), dtype=np.float32))
        return scores + boxes


class _ShapeChangingPickleSession(_FakeSession):
    """Model a PickableInferenceSession that loses a shape override."""

    def __getstate__(self):
        return {}

    def __setstate__(self, _values):
        self.__init__()


class _RecordingFactory:
    def __init__(self):
        self.calls = []
        self.sessions = []
        self._lock = Lock()

    def __call__(self, model_file, input_size, reference_session):
        with self._lock:
            width, height = input_size
            session = _FakeSession(
                input_shape=(1, 3, height, width),
                providers=reference_session.get_providers(),
                provider_options=reference_session.get_provider_options(),
                session_options=reference_session.get_session_options(),
            )
            self.calls.append(
                (model_file, tuple(input_size), tuple(session.providers))
            )
            self.sessions.append(session)
            return session


def _pickle_factory(_model_file, input_size, reference_session):
    width, height = input_size
    return _FakeSession(
        input_shape=(1, 3, height, width),
        providers=reference_session.get_providers(),
    )


def _write_model(path, height, width):
    input_info = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [1, 3, height, width],
    )
    output_info = onnx.helper.make_tensor_value_info(
        "identity",
        onnx.TensorProto.FLOAT,
        [1, 3, height, width],
    )
    node = onnx.helper.make_node("Identity", ["input"], ["identity"])
    graph = onnx.helper.make_graph(
        [node],
        "input-shape",
        [input_info],
        [output_info],
    )
    onnx.save(onnx.helper.make_model(graph), path)


def _write_scrfd_shape_model(path, *, height="?", width="?"):
    input_info = onnx.helper.make_tensor_value_info(
        "input",
        onnx.TensorProto.FLOAT,
        [1, 3, height, width],
    )
    outputs = []
    old_candidates = (12800, 3200, 800)
    for feature_width in (1, 4, 10):
        for index, candidates in enumerate(old_candidates):
            outputs.append(
                onnx.helper.make_tensor_value_info(
                    f"output_{feature_width}_{index}",
                    onnx.TensorProto.FLOAT,
                    [candidates, feature_width],
                )
            )
    graph = onnx.helper.make_graph(
        [],
        "scrfd-shapes",
        [input_info],
        outputs,
    )
    onnx.save(onnx.helper.make_model(graph), path)


def _forward(detector, side):
    image = np.zeros((side, side, 3), dtype=np.uint8)
    detector.forward(image, threshold=1.0)


def _serialized_model_input_shape(model_source):
    model = onnx.load_model_from_string(model_source)
    dimensions = model.graph.input[0].type.tensor_type.shape.dim
    return tuple(dimension.dim_value for dimension in dimensions)


def test_static_shape_sessions_are_default_and_main_dynamic_session_is_reference():
    main = _FakeSession()
    factory = _RecordingFactory()
    detector = SCRFD(
        model_file="fake.onnx",
        session=main,
        resolution_session_factory=factory,
    )

    _forward(detector, 64)
    _forward(detector, 128)
    _forward(detector, 128)

    assert detector.static_shape_sessions is True
    assert main.runs == []
    assert [call[1] for call in factory.calls] == [
        (64, 64),
        (128, 128),
    ]
    assert factory.sessions[0].runs == [(1, 3, 64, 64)]
    assert factory.sessions[1].runs == [
        (1, 3, 128, 128),
        (1, 3, 128, 128),
    ]
    assert detector.resolution_session_input_sizes == (
        (64, 64),
        (128, 128),
    )
    assert detector.session is main


def test_direct_scrfd_construction_uses_default_providers(monkeypatch, tmp_path):
    model_file = tmp_path / "detector.onnx"
    model_file.write_bytes(b"not-an-onnx-model")
    selected = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    calls = []
    session = _FakeSession(providers=selected)
    monkeypatch.setattr(
        scrfd_module,
        "get_default_providers",
        lambda: selected,
    )
    monkeypatch.setattr(
        scrfd_module.onnxruntime,
        "InferenceSession",
        lambda model_path, **kwargs: calls.append((model_path, kwargs)) or session,
    )

    detector = SCRFD(model_file=str(model_file))

    assert detector.session is session
    assert calls == [
        (
            str(model_file),
            {"providers": selected},
        )
    ]


def test_injected_scrfd_session_does_not_run_default_selection(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(
        scrfd_module,
        "get_default_providers",
        lambda: (_ for _ in ()).throw(AssertionError("auto selection ran")),
    )

    detector = SCRFD(session=session)

    assert detector.session is session


def test_coreml_uses_the_same_default_static_session_policy():
    main = _FakeSession(
        providers=("CoreMLExecutionProvider", "CPUExecutionProvider"),
    )
    factory = _RecordingFactory()
    detector = SCRFD(
        model_file="fake.onnx",
        session=main,
        resolution_session_factory=factory,
    )

    selected = detector._session_for_input_size((64, 64))

    assert selected is factory.sessions[0]
    assert selected is not main
    assert [call[1] for call in factory.calls] == [(64, 64)]
    assert detector.resolution_session_input_sizes == ((64, 64),)


def test_static_shape_sessions_false_always_reuses_main_dynamic_session():
    main = _FakeSession()
    factory = _RecordingFactory()
    detector = SCRFD(
        model_file="fake.onnx",
        session=main,
        resolution_session_factory=factory,
        static_shape_sessions=False,
    )

    _forward(detector, 64)
    _forward(detector, 128)
    _forward(detector, 64)

    assert detector.static_shape_sessions is False
    assert factory.calls == []
    assert main.runs == [
        (1, 3, 64, 64),
        (1, 3, 128, 128),
        (1, 3, 64, 64),
    ]
    assert detector.resolution_sessions == {
        (64, 64): main,
        (128, 128): main,
    }


@pytest.mark.parametrize("value", [None, 0, 1, "false", np.bool_(True)])
def test_static_shape_sessions_requires_a_boolean(value):
    with pytest.raises(TypeError, match="must be boolean"):
        SCRFD(session=_FakeSession(), static_shape_sessions=value)


def test_static_factory_must_return_exact_fixed_input_shape():
    main = _FakeSession()

    def dynamic_factory(_model_file, _input_size, reference_session):
        return _FakeSession(providers=reference_session.get_providers())

    detector = SCRFD(
        model_file="fake.onnx",
        session=main,
        resolution_session_factory=dynamic_factory,
    )

    with pytest.raises(
        RuntimeError,
        match=r"must have fixed input size \(128, 96\), received None",
    ):
        detector._session_for_input_size((128, 96))

    assert detector.resolution_session_input_sizes == ()


def test_same_resolution_is_constructed_once_under_concurrency():
    main = _FakeSession()
    factory = _RecordingFactory()
    detector = SCRFD(
        model_file="fake.onnx",
        session=main,
        resolution_session_factory=factory,
    )
    detector._session_for_input_size((64, 64))

    with ThreadPoolExecutor(max_workers=8) as executor:
        sessions = list(
            executor.map(
                lambda _index: detector._session_for_input_size((128, 128)),
                range(32),
            )
        )

    assert [call[1] for call in factory.calls] == [
        (64, 64),
        (128, 128),
    ]
    assert all(session is sessions[0] for session in sessions)


def test_rectangular_and_transposed_resolutions_use_distinct_sessions():
    main = _FakeSession()
    factory = _RecordingFactory()
    detector = SCRFD(
        model_file="fake.onnx",
        session=main,
        resolution_session_factory=factory,
    )
    detector._session_for_input_size((64, 64))

    wide = detector._session_for_input_size((128, 96))
    tall = detector._session_for_input_size((96, 128))

    assert wide is not tall
    assert [call[1] for call in factory.calls] == [
        (64, 64),
        (128, 96),
        (96, 128),
    ]
    assert detector.resolution_session_input_sizes == (
        (64, 64),
        (96, 128),
        (128, 96),
    )


def test_failed_creation_is_not_cached_and_can_be_retried():
    main = _FakeSession()
    attempts = []

    def flaky_factory(_model_file, input_size, reference_session):
        attempts.append(tuple(input_size))
        if len(attempts) == 1:
            raise RuntimeError("transient creation failure")
        width, height = input_size
        return _FakeSession(
            input_shape=(1, 3, height, width),
            providers=reference_session.get_providers(),
        )

    detector = SCRFD(
        model_file="fake.onnx",
        session=main,
        resolution_session_factory=flaky_factory,
    )
    with pytest.raises(RuntimeError, match="transient creation failure"):
        detector._session_for_input_size((128, 128))
    session = detector._session_for_input_size((128, 128))

    assert session is not None
    assert attempts == [(128, 128), (128, 128)]


def test_shallow_copies_share_resolution_pool_but_keep_main_session_attribute():
    main = _FakeSession()
    factory = _RecordingFactory()
    detector = SCRFD(
        model_file="fake.onnx",
        session=main,
        resolution_session_factory=factory,
    )
    detector._session_for_input_size((64, 64))
    copied = copy.copy(detector)

    copied._session_for_input_size((128, 128))

    assert copied._resolution_session_pool is detector._resolution_session_pool
    assert detector.resolution_sessions[(128, 128)] is factory.sessions[1]
    assert copied.session is detector.session is main


def test_prepare_cpu_discards_derived_sessions_and_future_factory_uses_cpu():
    main = _FakeSession(providers=("CUDAExecutionProvider", "CPUExecutionProvider"))
    factory = _RecordingFactory()
    detector = SCRFD(
        model_file="fake.onnx",
        session=main,
        resolution_session_factory=factory,
    )
    detector._session_for_input_size((64, 64))
    old_derived = detector._session_for_input_size((128, 128))

    detector.prepare(-1)

    assert detector.resolution_session_input_sizes == ()
    new_first = detector._session_for_input_size((64, 64))
    assert new_first is not main
    new_derived = detector._session_for_input_size((128, 128))
    assert new_derived is not old_derived
    assert factory.calls[-1][2] == ("CPUExecutionProvider",)


def test_external_provider_change_is_detected_and_discards_derived_sessions():
    main = _FakeSession()
    factory = _RecordingFactory()
    detector = SCRFD(
        model_file="fake.onnx",
        session=main,
        resolution_session_factory=factory,
    )
    detector._session_for_input_size((64, 64))
    old_derived = detector._session_for_input_size((128, 128))

    main.set_providers(("OtherExecutionProvider", "CPUExecutionProvider"))

    assert detector.resolution_session_input_sizes == ()
    new_first = detector._session_for_input_size((64, 64))
    assert new_first is not main
    new_derived = detector._session_for_input_size((128, 128))
    assert new_derived is not old_derived
    assert factory.calls[-1][2] == (
        "OtherExecutionProvider",
        "CPUExecutionProvider",
    )


def test_static_session_policy_is_unchanged_across_coreml_cpu_switches():
    main = _FakeSession(
        providers=("CoreMLExecutionProvider", "CPUExecutionProvider"),
    )
    factory = _RecordingFactory()
    detector = SCRFD(
        model_file="fake.onnx",
        session=main,
        resolution_session_factory=factory,
    )
    coreml_session = detector._session_for_input_size((64, 64))
    assert coreml_session is not main

    main.set_providers(("CPUExecutionProvider",))
    assert detector.resolution_session_input_sizes == ()
    cpu_session = detector._session_for_input_size((128, 128))
    assert cpu_session is not main

    main.set_providers(("CoreMLExecutionProvider", "CPUExecutionProvider"))
    assert detector.resolution_session_input_sizes == ()
    second_coreml_session = detector._session_for_input_size((64, 64))
    assert second_coreml_session is not main
    assert second_coreml_session is not coreml_session


def test_default_factory_preserves_runtime_configuration(
    monkeypatch,
    tmp_path,
):
    model_file = tmp_path / "detector.onnx"
    _write_scrfd_shape_model(model_file)
    session_options = object()
    main = _FakeSession(
        providers=("ExampleExecutionProvider", "CPUExecutionProvider"),
        provider_options={
            "ExampleExecutionProvider": {"device_id": "3"},
            "CPUExecutionProvider": {"arena_extend_strategy": "0"},
        },
        session_options=session_options,
    )
    created = []

    def inference_session(model_source, **kwargs):
        created.append((model_source, kwargs))
        return _FakeSession(
            input_shape=_serialized_model_input_shape(model_source),
            providers=kwargs["providers"],
            provider_options=dict(zip(kwargs["providers"], kwargs["provider_options"])),
            session_options=kwargs["sess_options"],
        )

    monkeypatch.setattr(scrfd_module.onnxruntime, "InferenceSession", inference_session)
    detector = SCRFD(model_file=str(model_file), session=main)
    detector._session_for_input_size((128, 96))

    assert len(created) == 1
    model_source, kwargs = created[0]
    assert isinstance(model_source, bytes)
    static_model = onnx.load_model_from_string(model_source)
    dimensions = static_model.graph.input[0].type.tensor_type.shape.dim
    assert [dimension.dim_value for dimension in dimensions] == [
        1,
        3,
        96,
        128,
    ]
    assert kwargs == {
        "sess_options": session_options,
        "providers": [
            "ExampleExecutionProvider",
            "CPUExecutionProvider",
        ],
        "provider_options": [
            {"device_id": "3"},
            {"arena_extend_strategy": "0"},
        ],
    }


@pytest.mark.parametrize(
    "providers",
    [
        ("CPUExecutionProvider",),
        ("CUDAExecutionProvider", "CPUExecutionProvider"),
        (
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ),
    ],
)
def test_default_cpu_and_cuda_factories_receive_static_model_bytes(
    monkeypatch,
    tmp_path,
    providers,
):
    monkeypatch.setattr(
        scrfd_module,
        "create_coreml_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-CoreML-primary path used CoreML cache manager")
        ),
    )
    model_file = tmp_path / "detector.onnx"
    _write_scrfd_shape_model(model_file)
    main = _FakeSession(providers=providers, output_count=9)
    created = []

    def inference_session(model_source, **kwargs):
        created.append((model_source, kwargs))
        return _FakeSession(
            input_shape=_serialized_model_input_shape(model_source),
            providers=kwargs["providers"],
            provider_options=dict(
                zip(kwargs["providers"], kwargs["provider_options"])
            ),
            session_options=kwargs["sess_options"],
            output_count=9,
        )

    monkeypatch.setattr(
        scrfd_module.onnxruntime,
        "InferenceSession",
        inference_session,
    )
    detector = SCRFD(model_file=str(model_file), session=main)

    selected = detector._session_for_input_size((320, 192))

    assert selected is not main
    assert len(created) == 1
    model_source, kwargs = created[0]
    assert isinstance(model_source, bytes)
    static_model = onnx.load_model_from_string(model_source)
    dimensions = static_model.graph.input[0].type.tensor_type.shape.dim
    assert [dimension.dim_value for dimension in dimensions] == [
        1,
        3,
        192,
        320,
    ]
    assert kwargs["providers"] == list(providers)
    assert all(
        "RequireStaticInputShapes" not in options
        for options in kwargs["provider_options"]
    )


def test_plain_session_without_model_source_keeps_legacy_shared_behavior():
    main = _FakeSession()
    detector = SCRFD(session=main)

    first = detector._session_for_input_size((64, 64))
    second = detector._session_for_input_size((128, 96))

    assert first is second is main
    assert detector.resolution_sessions == {
        (64, 64): main,
        (128, 96): main,
    }


def test_default_coreml_factory_serializes_exact_scrfd_shapes(monkeypatch, tmp_path):
    monkeypatch.setattr(
        coreml_cache_module,
        "default_coreml_cache_root",
        lambda: tmp_path / "coreml-cache",
    )
    monkeypatch.setattr(
        scrfd_module.platform,
        "mac_ver",
        lambda: ("26.0", ("", "", ""), "arm64"),
    )
    model_file = tmp_path / "buffalo-style.onnx"
    # Buffalo uses the same anonymous '?' dim_param for height and width.
    _write_scrfd_shape_model(model_file)
    session_options = scrfd_module.onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 3
    session_options.inter_op_num_threads = 2
    session_options.log_severity_level = 2
    session_options.enable_cpu_mem_arena = False
    session_options.enable_mem_pattern = False
    session_options.enable_mem_reuse = False
    session_options.graph_optimization_level = (
        scrfd_module.onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )
    session_options.use_deterministic_compute = True
    session_options.logid = "scrfd-coreml"
    session_options.profile_file_prefix = "scrfd-profile"
    main = _FakeSession(
        providers=("CoreMLExecutionProvider", "CPUExecutionProvider"),
        provider_options={
            "CoreMLExecutionProvider": {
                "MLComputeUnits": "CPUAndGPU",
                "ModelCacheDirectory": "/stale-main-session-cache",
                "RequireStaticInputShapes": "0",
            },
            "CPUExecutionProvider": {"use_arena": "1"},
        },
        session_options=session_options,
        output_count=9,
    )
    created = []

    def inference_session(model_source, **kwargs):
        created.append((model_source, kwargs))
        return _FakeSession(
            input_shape=_serialized_model_input_shape(model_source),
            providers=kwargs["providers"],
            provider_options=dict(
                zip(kwargs["providers"], kwargs["provider_options"])
            ),
            session_options=kwargs["sess_options"],
            output_count=9,
        )

    monkeypatch.setattr(scrfd_module.onnxruntime, "InferenceSession", inference_session)
    detector = SCRFD(model_file=str(model_file), session=main)

    selected = detector._session_for_input_size((320, 192))

    assert selected is not main
    assert len(created) == 1
    model_source, kwargs = created[0]
    assert isinstance(model_source, bytes)
    static_model = onnx.load_model_from_string(model_source)
    input_dimensions = static_model.graph.input[0].type.tensor_type.shape.dim
    assert [dimension.dim_value for dimension in input_dimensions] == [
        1,
        3,
        192,
        320,
    ]
    expected_candidates = [1920, 480, 120] * 3
    actual_shapes = [
        [dimension.dim_value for dimension in output.type.tensor_type.shape.dim]
        for output in static_model.graph.output
    ]
    assert [shape[0] for shape in actual_shapes] == expected_candidates
    assert [shape[1] for shape in actual_shapes] == [1] * 3 + [4] * 3 + [10] * 3
    assert kwargs["sess_options"] is not session_options
    assert kwargs["sess_options"].intra_op_num_threads == 3
    assert kwargs["sess_options"].inter_op_num_threads == 2
    assert kwargs["sess_options"].log_severity_level == 2
    assert kwargs["sess_options"].enable_cpu_mem_arena is False
    assert kwargs["sess_options"].enable_mem_pattern is False
    assert kwargs["sess_options"].enable_mem_reuse is False
    assert kwargs["sess_options"].graph_optimization_level == (
        scrfd_module.onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )
    assert kwargs["sess_options"].use_deterministic_compute is True
    assert kwargs["sess_options"].logid == "scrfd-coreml"
    assert kwargs["sess_options"].profile_file_prefix == "scrfd-profile"
    assert kwargs["providers"] == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    coreml_options = kwargs["provider_options"][0]
    assert coreml_options["EnableOnSubgraphs"] == "0"
    assert coreml_options["MLComputeUnits"] == "CPUAndGPU"
    assert coreml_options["ModelFormat"] == "MLProgram"
    assert coreml_options["RequireStaticInputShapes"] == "1"
    assert coreml_options["ModelCacheDirectory"] != "/stale-main-session-cache"
    assert Path(coreml_options["ModelCacheDirectory"]).is_relative_to(
        tmp_path / "coreml-cache"
    )
    assert kwargs["provider_options"][1] == {"use_arena": "1"}
    assert selected.runs == [(1, 3, 192, 320)]


def test_default_coreml_all_cache_is_shape_scoped_and_reused(
    monkeypatch,
    tmp_path,
):
    cache_root = tmp_path / "coreml-cache"
    monkeypatch.setattr(
        coreml_cache_module,
        "default_coreml_cache_root",
        lambda: cache_root,
    )
    monkeypatch.setattr(
        scrfd_module.platform,
        "mac_ver",
        lambda: ("26.0", ("", "", ""), "arm64"),
    )
    model_file = tmp_path / "dynamic-scrfd.onnx"
    _write_scrfd_shape_model(model_file)
    main = _FakeSession(
        providers=("CoreMLExecutionProvider", "CPUExecutionProvider"),
        provider_options={
            "CoreMLExecutionProvider": {
                "ModelCacheDirectory": "/cache-for-main-session",
            },
            "CPUExecutionProvider": {},
        },
        output_count=9,
    )
    created = []

    def inference_session(model_source, **kwargs):
        cache_directory = Path(
            kwargs["provider_options"][0]["ModelCacheDirectory"]
        )
        (cache_directory / "compiled_model.mlmodelc").mkdir(
            parents=True,
            exist_ok=True,
        )
        session = _FakeSession(
            input_shape=_serialized_model_input_shape(model_source),
            providers=kwargs["providers"],
            provider_options=dict(
                zip(kwargs["providers"], kwargs["provider_options"])
            ),
            session_options=kwargs["sess_options"],
            output_count=9,
        )
        created.append((model_source, kwargs, session))
        return session

    monkeypatch.setattr(
        scrfd_module.onnxruntime,
        "InferenceSession",
        inference_session,
    )
    scrfd_module._model_file_sha256_cached.cache_clear()
    detector = SCRFD(model_file=str(model_file), session=main)

    first = detector._session_for_input_size((128, 96))
    second = detector._session_for_input_size((320, 192))
    repeated_detector = SCRFD(model_file=str(model_file), session=main)
    repeated = repeated_detector._session_for_input_size((128, 96))

    assert len(created) == 3
    assert all(
        call[1]["provider_options"][0]["MLComputeUnits"] == "ALL"
        for call in created
    )
    cache_directories = [
        Path(call[1]["provider_options"][0]["ModelCacheDirectory"])
        for call in created
    ]
    assert cache_directories[0] != cache_directories[1]
    assert cache_directories[0] == cache_directories[2]
    assert all(path.is_relative_to(cache_root) for path in cache_directories)
    assert first.runs == [(1, 3, 96, 128)]
    assert second.runs == [(1, 3, 192, 320)]
    # The repeated Session loads an already validated compiled artifact, so
    # the manager intentionally skips warmup.
    assert repeated.runs == []
    hash_info = scrfd_module._model_file_sha256_cached.cache_info()
    assert hash_info.misses == 1
    assert hash_info.hits == 2


def test_default_coreml_all_failure_retries_cpu_and_gpu_with_fresh_options(
    monkeypatch,
    tmp_path,
):
    cache_root = tmp_path / "coreml-cache"
    monkeypatch.setattr(
        coreml_cache_module,
        "default_coreml_cache_root",
        lambda: cache_root,
    )
    monkeypatch.setattr(
        scrfd_module.platform,
        "mac_ver",
        lambda: ("26.0", ("", "", ""), "arm64"),
    )
    model_file = tmp_path / "dynamic-scrfd.onnx"
    _write_scrfd_shape_model(model_file)
    main = _FakeSession(
        providers=("CoreMLExecutionProvider", "CPUExecutionProvider"),
        output_count=9,
    )
    created = []

    def inference_session(model_source, **kwargs):
        options = kwargs["provider_options"][0]
        cache_directory = Path(options["ModelCacheDirectory"])
        cache_directory.mkdir(parents=True, exist_ok=True)
        created.append((options.copy(), kwargs["sess_options"]))
        if options["MLComputeUnits"] == "ALL":
            (cache_directory / "partial-compilation").write_text(
                "broken",
                encoding="utf-8",
            )
            raise RuntimeError("Core ML execution plan error -6")
        (cache_directory / "compiled_model.mlmodelc").mkdir(exist_ok=True)
        return _FakeSession(
            input_shape=_serialized_model_input_shape(model_source),
            providers=kwargs["providers"],
            provider_options=dict(
                zip(kwargs["providers"], kwargs["provider_options"])
            ),
            session_options=kwargs["sess_options"],
            output_count=9,
        )

    monkeypatch.setattr(
        scrfd_module.onnxruntime,
        "InferenceSession",
        inference_session,
    )
    detector = SCRFD(model_file=str(model_file), session=main)

    selected = detector._session_for_input_size((256, 160))

    assert [options["MLComputeUnits"] for options, _session in created] == [
        "ALL",
        "CPUAndGPU",
    ]
    all_directory = Path(created[0][0]["ModelCacheDirectory"])
    fallback_directory = Path(created[1][0]["ModelCacheDirectory"])
    assert all_directory != fallback_directory
    assert not all_directory.exists()
    assert fallback_directory.exists()
    assert created[0][1] is not created[1][1]
    assert selected.runs == [(1, 3, 160, 256)]


def test_dynamic_graph_with_fixed_effective_main_routes_other_size_to_factory(
    tmp_path,
):
    model_file = tmp_path / "dynamic.onnx"
    _write_model(model_file, "height", "width")
    main = _FakeSession(
        input_shape=(1, 3, 800, 800),
        providers=("CoreMLExecutionProvider", "CPUExecutionProvider"),
    )
    factory = _RecordingFactory()

    detector = SCRFD(
        model_file=str(model_file),
        session=main,
        resolution_session_factory=factory,
    )
    assert detector._session_for_input_size((800, 800)) is main
    _forward(detector, 256)

    assert detector.static_input_size is None
    assert detector.resolution_session_input_sizes == ((256, 256), (800, 800))
    assert main.runs == []
    assert factory.calls[0][1] == (256, 256)
    assert factory.sessions[0].runs == [(1, 3, 256, 256)]


def test_native_static_graph_only_uses_its_original_resolution(tmp_path):
    model_file = tmp_path / "static.onnx"
    _write_model(model_file, 800, 800)
    main = _FakeSession(input_shape=(1, 3, 800, 800))
    factory = _RecordingFactory()
    detector = SCRFD(
        model_file=str(model_file),
        session=main,
        resolution_session_factory=factory,
    )

    assert detector.static_input_size == (800, 800)
    assert detector._resolve_input_sizes((256, 256)) == [(800, 800)]
    with pytest.raises(ValueError, match="only supports resolution"):
        detector._session_for_input_size((256, 256))
    assert factory.calls == []


def test_pickle_drops_derived_sessions_and_rebuilds_pool_lock():
    main = _FakeSession()
    detector = SCRFD(
        model_file="fake.onnx",
        session=main,
        resolution_session_factory=_pickle_factory,
    )
    detector._session_for_input_size((64, 64))
    detector._session_for_input_size((128, 128))

    restored = pickle.loads(pickle.dumps(detector))

    assert restored.resolution_session_input_sizes == ()
    assert restored._session_for_input_size((128, 128)) is not None


def test_pickle_preserves_main_aliases_in_dynamic_compatibility_mode():
    main = _FakeSession()
    detector = SCRFD(
        model_file="fake.onnx",
        session=main,
        resolution_session_factory=_pickle_factory,
        static_shape_sessions=False,
    )
    detector._session_for_input_size((64, 64))
    detector._session_for_input_size((128, 128))

    restored = pickle.loads(pickle.dumps(detector))

    assert restored.static_shape_sessions is False
    assert restored.resolution_session_input_sizes == (
        (64, 64),
        (128, 128),
    )
    assert all(
        session is restored.session
        for session in restored.resolution_sessions.values()
    )


def test_pickle_revalidates_fixed_main_alias_after_session_recreation(tmp_path):
    model_file = tmp_path / "dynamic.onnx"
    _write_model(model_file, "height", "width")
    main = _ShapeChangingPickleSession(input_shape=(1, 3, 128, 128))
    detector = SCRFD(
        model_file=str(model_file),
        session=main,
        resolution_session_factory=_pickle_factory,
    )
    assert detector.resolution_session_input_sizes == ((128, 128),)

    restored = pickle.loads(pickle.dumps(detector))

    assert restored.session.get_inputs()[0].shape == [
        1,
        3,
        "height",
        "width",
    ]
    assert restored.resolution_session_input_sizes == ()
    resolution_session = restored._session_for_input_size((128, 128))
    assert resolution_session is not restored.session
    assert resolution_session.get_inputs()[0].shape == [1, 3, 128, 128]
