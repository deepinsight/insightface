import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from insightface.model_zoo import coreml_cache


MODEL_SHA = "a" * 64
INPUTS = (
    {
        "name": "input.1",
        "dtype": "tensor(float)",
        "shape": [1, 3, 160, 160],
    },
)
PROVIDERS = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
OPTIONS = [
    {
        "ModelFormat": "MLProgram",
        "RequireStaticInputShapes": "1",
        "EnableOnSubgraphs": "0",
    },
    {},
]


class _FakeSession:
    def __init__(self, providers=None, run_hook=None):
        self._providers = list(providers or PROVIDERS)
        self.run_hook = run_hook
        self.runs = []

    def get_providers(self):
        return list(self._providers)

    def run(self, output_names, feed):
        self.runs.append((output_names, feed))
        if self.run_hook is not None:
            return self.run_hook(output_names, feed)
        return []


class _RecordingFactory:
    def __init__(self, behavior=None):
        self.calls = []
        self.behavior = behavior

    def __call__(self, model_source, **kwargs):
        call = {
            "model_source": model_source,
            **kwargs,
        }
        self.calls.append(call)
        cache_directory = Path(
            kwargs["provider_options"][0]["ModelCacheDirectory"]
        )
        cache_directory.mkdir(parents=True, exist_ok=True)
        (cache_directory / "compiled_model.mlmodelc").mkdir(exist_ok=True)
        if self.behavior is not None:
            result = self.behavior(call, len(self.calls))
            if result is not None:
                return result
        return _FakeSession()


def _create(factory, tmp_path, **kwargs):
    values = {
        "providers": PROVIDERS,
        "provider_options": OPTIONS,
        "model_sha256": MODEL_SHA,
        "task": "detection",
        "graph_variant": "static_scrfd_v1",
        "input_contracts": INPUTS,
        "cache_root": tmp_path,
    }
    values.update(kwargs)
    return coreml_cache.create_coreml_session(
        factory,
        b"onnx-model",
        **values,
    )


def test_all_candidate_is_persisted_and_cache_hit_skips_warmup(tmp_path):
    factory = _RecordingFactory()

    first = _create(factory, tmp_path, warmup=True)
    second = _create(factory, tmp_path, warmup=True)

    assert first.compute_units == "ALL"
    assert first.cache_hit is False
    assert first.warmup_performed is True
    assert len(first.session.runs) == 1
    _, feed = first.session.runs[0]
    assert feed["input.1"].shape == (1, 3, 160, 160)
    assert feed["input.1"].dtype == np.float32
    assert second.compute_units == "ALL"
    assert second.cache_hit is True
    assert second.warmup_performed is False
    assert second.session.runs == []
    assert first.cache_directory == second.cache_directory
    assert factory.calls[0]["provider_options"][0]["MLComputeUnits"] == "ALL"
    assert factory.calls[1]["provider_options"][0]["MLComputeUnits"] == "ALL"
    assert (first.cache_directory / ".insightface-validated.json").is_file()
    base_signature = json.loads(
        (first.cache_directory.parent / "signature.json").read_text(
            encoding="utf-8"
        )
    )
    candidate_signature = json.loads(
        (first.cache_directory / "signature.json").read_text(
            encoding="utf-8"
        )
    )
    assert "MLComputeUnits" not in base_signature["signature"]["coreml_options"]
    assert (
        candidate_signature["signature"]["coreml_options"]["MLComputeUnits"]
        == "ALL"
    )


def test_all_failure_clears_its_leaf_and_falls_back_to_cpu_and_gpu(tmp_path):
    all_directory = None

    def behavior(call, _index):
        nonlocal all_directory
        options = call["provider_options"][0]
        if options["MLComputeUnits"] == "ALL":
            all_directory = Path(options["ModelCacheDirectory"])
            (all_directory / "partial").write_text("broken", encoding="utf-8")
            raise RuntimeError("Core ML error -6")
        return _FakeSession()

    factory = _RecordingFactory(behavior)
    result = _create(factory, tmp_path, warmup=True)

    assert result.compute_units == "CPUAndGPU"
    assert [
        call["provider_options"][0]["MLComputeUnits"]
        for call in factory.calls
    ] == ["ALL", "CPUAndGPU"]
    assert all_directory is not None
    assert not all_directory.exists()
    assert result.cache_directory != all_directory
    selection = json.loads(
        (result.cache_directory.parent / "selection.json").read_text(
            encoding="utf-8"
        )
    )
    assert selection["compute_units"] == "CPUAndGPU"


def test_remembered_cpu_and_gpu_is_reused_without_retrying_all(tmp_path):
    def behavior(call, _index):
        if call["provider_options"][0]["MLComputeUnits"] == "ALL":
            raise RuntimeError("unsupported execution plan")
        return _FakeSession()

    first_factory = _RecordingFactory(behavior)
    first = _create(first_factory, tmp_path)
    second_factory = _RecordingFactory()

    second = _create(second_factory, tmp_path)

    assert first.compute_units == "CPUAndGPU"
    assert second.compute_units == "CPUAndGPU"
    assert second.cache_hit is True
    assert len(second_factory.calls) == 1
    assert (
        second_factory.calls[0]["provider_options"][0]["MLComputeUnits"]
        == "CPUAndGPU"
    )


def test_failed_remembered_cpu_and_gpu_reprobes_all(tmp_path):
    def initial_behavior(call, _index):
        if call["provider_options"][0]["MLComputeUnits"] == "ALL":
            raise RuntimeError("ALL was not supported by the old runtime")
        return _FakeSession()

    first = _create(_RecordingFactory(initial_behavior), tmp_path)
    assert first.compute_units == "CPUAndGPU"

    def upgraded_behavior(call, _index):
        if call["provider_options"][0]["MLComputeUnits"] == "CPUAndGPU":
            raise RuntimeError("the remembered plan is no longer loadable")
        return _FakeSession()

    factory = _RecordingFactory(upgraded_behavior)
    with pytest.warns(RuntimeWarning, match="clearing and recompiling"):
        recovered = _create(factory, tmp_path)

    assert recovered.compute_units == "ALL"
    assert [
        call["provider_options"][0]["MLComputeUnits"]
        for call in factory.calls
    ] == ["CPUAndGPU", "CPUAndGPU", "ALL"]


def test_missing_selected_leaf_rebuilds_same_choice_and_runs_warmup(tmp_path):
    def initial_behavior(call, _index):
        if call["provider_options"][0]["MLComputeUnits"] == "ALL":
            raise RuntimeError("unsupported execution plan")
        return _FakeSession()

    first = _create(_RecordingFactory(initial_behavior), tmp_path, warmup=True)
    selection_path = first.cache_directory.parent / "selection.json"
    selection_before = selection_path.read_text(encoding="utf-8")
    coreml_cache.shutil.rmtree(first.cache_directory)
    factory = _RecordingFactory()

    rebuilt = _create(factory, tmp_path, warmup=True)

    assert selection_before
    assert rebuilt.compute_units == "CPUAndGPU"
    assert rebuilt.cache_hit is False
    assert rebuilt.warmup_performed is True
    assert len(rebuilt.session.runs) == 1
    assert len(factory.calls) == 1


def test_marker_without_an_ort_artifact_is_not_a_cache_hit(tmp_path):
    first = _create(_RecordingFactory(), tmp_path, warmup=True)
    for child in first.cache_directory.iterdir():
        if child.name not in {
            ".insightface-validated.json",
            "signature.json",
        }:
            coreml_cache.shutil.rmtree(child)
    factory = _RecordingFactory()

    rebuilt = _create(factory, tmp_path, warmup=True)

    assert rebuilt.cache_hit is False
    assert rebuilt.warmup_performed is True
    assert len(rebuilt.session.runs) == 1


def test_corrupt_selected_cache_is_cleaned_and_rebuilt_once(tmp_path):
    first = _create(_RecordingFactory(), tmp_path, warmup=True)
    attempts = 0

    def behavior(call, _index):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("cached mlmodelc could not be loaded")
        return _FakeSession()

    factory = _RecordingFactory(behavior)
    with pytest.warns(RuntimeWarning, match="clearing and recompiling"):
        rebuilt = _create(factory, tmp_path, warmup=True)

    assert first.compute_units == rebuilt.compute_units == "ALL"
    assert rebuilt.cache_hit is False
    assert rebuilt.warmup_performed is True
    assert attempts == 2
    assert len(rebuilt.session.runs) == 1


@pytest.mark.parametrize("compute_units", ["CPUOnly", "CPUAndNeuralEngine"])
def test_explicit_non_all_compute_units_is_the_only_candidate(
    tmp_path,
    compute_units,
):
    options = [
        {
            **OPTIONS[0],
            "MLComputeUnits": compute_units,
        },
        {},
    ]
    factory = _RecordingFactory()

    result = _create(factory, tmp_path, provider_options=options)

    assert result.compute_units == compute_units
    assert len(factory.calls) == 1
    assert (
        factory.calls[0]["provider_options"][0]["MLComputeUnits"]
        == compute_units
    )


def test_explicit_cpu_and_gpu_selection_does_not_poison_later_auto_mode(
    tmp_path,
):
    explicit_options = [
        {**OPTIONS[0], "MLComputeUnits": "CPUAndGPU"},
        {},
    ]
    explicit = _create(
        _RecordingFactory(),
        tmp_path,
        provider_options=explicit_options,
    )
    automatic_factory = _RecordingFactory()

    automatic = _create(automatic_factory, tmp_path)

    assert explicit.compute_units == "CPUAndGPU"
    assert automatic.compute_units == "ALL"
    assert automatic.cache_hit is False
    assert (
        automatic_factory.calls[0]["provider_options"][0]["MLComputeUnits"]
        == "ALL"
    )


def test_silent_cpu_fallback_is_not_persisted_as_all(tmp_path):
    def behavior(call, _index):
        if call["provider_options"][0]["MLComputeUnits"] == "ALL":
            return _FakeSession(["CPUExecutionProvider"])
        return _FakeSession()

    factory = _RecordingFactory(behavior)

    result = _create(factory, tmp_path)

    assert result.compute_units == "CPUAndGPU"
    assert len(factory.calls) == 2


def test_versions_are_diagnostic_only_and_do_not_change_signature(tmp_path):
    first = _create(
        _RecordingFactory(),
        tmp_path,
        diagnostic_metadata={"onnxruntime": "1.29.0", "macos": "26.0"},
    )
    factory = _RecordingFactory()

    second = _create(
        factory,
        tmp_path,
        diagnostic_metadata={"onnxruntime": "2.0.0", "macos": "27.0"},
    )

    assert first.cache_directory == second.cache_directory
    assert first.signature == second.signature
    assert second.cache_hit is True
    assert len(factory.calls) == 1
    stored = json.loads(
        (second.cache_directory.parent / "selection.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored["created_with"]["onnxruntime"] == "1.29.0"


def test_callback_and_explicit_feed_warmups_are_supported(tmp_path):
    callback_sessions = []
    first = _create(
        _RecordingFactory(),
        tmp_path / "callback",
        warmup=lambda session: callback_sessions.append(session),
    )
    feed = {"input.1": np.ones((1, 3, 4, 4), dtype=np.float32)}
    second = _create(
        _RecordingFactory(),
        tmp_path / "feed",
        warmup=feed,
    )

    assert callback_sessions == [first.session]
    assert second.session.runs[0][1] is not feed
    assert np.array_equal(second.session.runs[0][1]["input.1"], feed["input.1"])


def test_non_coreml_providers_bypass_cache_policy(tmp_path):
    factory = _RecordingFactory()
    # This factory expects a CoreML cache directory, so use a small generic
    # factory to prove the bypass path forwards the caller's options unchanged.
    calls = []

    def cpu_factory(model_source, **kwargs):
        calls.append((model_source, kwargs))
        return _FakeSession(["CPUExecutionProvider"])

    result = coreml_cache.create_coreml_session(
        cpu_factory,
        "model.onnx",
        providers=["CPUExecutionProvider"],
        provider_options=[{"arena_extend_strategy": "kSameAsRequested"}],
        model_sha256=MODEL_SHA,
        task="detection",
        graph_variant="original_onnx_v1",
        input_contracts=INPUTS,
        cache_root=tmp_path,
        warmup=True,
    )

    assert result.compute_units is None
    assert result.cache_directory is None
    assert calls[0][1]["provider_options"] == [
        {"arena_extend_strategy": "kSameAsRequested"}
    ]
    assert list(tmp_path.iterdir()) == []


def test_coreml_later_in_provider_chain_bypasses_cache_policy(tmp_path):
    calls = []

    def factory(model_source, **kwargs):
        calls.append((model_source, kwargs))
        return _FakeSession(["CUDAExecutionProvider", "CoreMLExecutionProvider"])

    result = coreml_cache.create_coreml_session(
        factory,
        "model.onnx",
        providers=["CUDAExecutionProvider", "CoreMLExecutionProvider"],
        provider_options=[{}, {"MLComputeUnits": "CPUOnly"}],
        model_sha256=MODEL_SHA,
        task="detection",
        graph_variant="original_onnx_v1",
        input_contracts=INPUTS,
        cache_root=tmp_path,
    )

    assert result.compute_units is None
    assert calls[0][1]["provider_options"][1] == {
        "MLComputeUnits": "CPUOnly"
    }
    assert list(tmp_path.iterdir()) == []


def test_thread_lock_allows_only_one_warmup_for_a_signature(tmp_path):
    barrier = threading.Barrier(2)
    warmups = []
    results = []

    def worker():
        barrier.wait()
        result = _create(
            _RecordingFactory(),
            tmp_path,
            warmup=lambda session: warmups.append(session),
        )
        results.append(result)

    threads = [threading.Thread(target=worker) for _index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert len(results) == 2
    assert len(warmups) == 1
    assert sorted(result.cache_hit for result in results) == [False, True]


def test_copy_session_options_returns_a_fresh_object_with_overrides(monkeypatch):
    class FakeOptions:
        def __init__(self):
            self.logid = ""
            self.intra_op_num_threads = 0
            self.overrides = []

        def add_free_dimension_override_by_name(self, name, value):
            self.overrides.append((name, value))

    source = FakeOptions()
    source.logid = "source"
    source.intra_op_num_threads = 3
    monkeypatch.setattr(coreml_cache, "_new_session_options", FakeOptions)

    copied = coreml_cache.copy_session_options(
        source,
        {"height": 256, "width": np.int64(320)},
    )

    assert copied is not source
    assert copied.logid == "source"
    assert copied.intra_op_num_threads == 3
    assert copied.overrides == [("height", 256), ("width", 320)]


@pytest.mark.parametrize(
    "contract",
    [
        {"name": "x", "dtype": "float32"},
        {"name": "x", "dtype": "float32", "shape": [], "extra": 1},
    ],
)
def test_input_contracts_have_an_exact_stable_schema(tmp_path, contract):
    with pytest.raises(ValueError, match="keys must be exactly"):
        _create(
            _RecordingFactory(),
            tmp_path,
            input_contracts=[contract],
        )
