import pytest

from insightface.model_zoo import model_zoo
from insightface.model_zoo import onnxruntime_utils
from insightface.model_zoo.onnxruntime_utils import get_default_providers


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        (
            [
                "CPUExecutionProvider",
                "CUDAExecutionProvider",
                "CoreMLExecutionProvider",
            ],
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        ),
        (
            ["CPUExecutionProvider", "CUDAExecutionProvider"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        (["CPUExecutionProvider"], ["CPUExecutionProvider"]),
        (["CoreMLExecutionProvider"], ["CoreMLExecutionProvider"]),
        (["CUDAExecutionProvider"], ["CUDAExecutionProvider"]),
        (["CustomExecutionProvider"], ["CustomExecutionProvider"]),
        (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] * 2,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
    ],
)
def test_default_provider_priority(available, expected):
    assert get_default_providers(available) == expected


def test_default_provider_priority_rejects_an_empty_runtime():
    with pytest.raises(RuntimeError, match="no available execution providers"):
        get_default_providers([])


def test_explicit_model_zoo_providers_do_not_run_auto_selection(monkeypatch):
    explicit = ["CPUExecutionProvider"]
    monkeypatch.setattr(
        model_zoo,
        "get_default_providers",
        lambda: (_ for _ in ()).throw(AssertionError("auto selection ran")),
    )

    assert model_zoo._session_kwargs({"providers": explicit})["providers"] is explicit


def test_model_zoo_none_providers_use_auto_selection(monkeypatch):
    selected = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.setattr(model_zoo, "get_default_providers", lambda: selected)

    assert model_zoo._session_kwargs({"providers": None})["providers"] is selected


def test_runtime_cuda_selection_preloads_environment_libraries(monkeypatch):
    calls = []
    monkeypatch.setattr(onnxruntime_utils, "_CUDA_PRELOAD_COMPLETE", False)
    monkeypatch.setattr(
        onnxruntime_utils.onnxruntime,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        onnxruntime_utils.onnxruntime,
        "preload_dlls",
        lambda: calls.append(True),
        raising=False,
    )

    assert get_default_providers() == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert calls == [True]


def test_injected_provider_inventory_has_no_preload_side_effect(monkeypatch):
    monkeypatch.setattr(onnxruntime_utils, "_CUDA_PRELOAD_COMPLETE", False)
    monkeypatch.setattr(
        onnxruntime_utils.onnxruntime,
        "preload_dlls",
        lambda: (_ for _ in ()).throw(AssertionError("preload ran")),
        raising=False,
    )

    assert get_default_providers(
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ) == ["CUDAExecutionProvider", "CPUExecutionProvider"]
