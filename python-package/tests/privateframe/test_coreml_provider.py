from __future__ import annotations

from collections.abc import Sequence

import pytest
from insightface.app.privateframe import base_config


def _resolve(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: Sequence[str],
    requested: str = "auto",
) -> dict[str, object]:
    monkeypatch.setattr(
        base_config.ort,
        "get_available_providers",
        lambda: list(available),
    )
    config: dict[str, object] = {"runtime": {"provider": requested}}
    base_config.resolve_runtime_provider(config)
    runtime = config["runtime"]
    assert isinstance(runtime, dict)
    return runtime


def test_auto_prefers_coreml_on_macos_even_when_cuda_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _resolve(
        monkeypatch,
        available=(
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ),
    )

    assert runtime["resolved_provider"] == "CoreMLExecutionProvider"
    assert runtime["providers"] == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_auto_falls_back_to_cuda_when_coreml_is_unavailable_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _resolve(
        monkeypatch,
        available=("CUDAExecutionProvider", "CPUExecutionProvider"),
    )

    assert runtime["resolved_provider"] == "CUDAExecutionProvider"
    assert runtime["providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_auto_uses_global_coreml_precedence_when_reported_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _resolve(
        monkeypatch,
        available=(
            "CoreMLExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ),
    )

    assert runtime["resolved_provider"] == "CoreMLExecutionProvider"
    assert runtime["providers"] == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_auto_uses_cpu_when_no_accelerated_provider_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _resolve(
        monkeypatch,
        available=("CPUExecutionProvider",),
    )

    assert runtime["resolved_provider"] == "CPUExecutionProvider"
    assert runtime["providers"] == ["CPUExecutionProvider"]


def test_explicit_unavailable_provider_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        base_config.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    config = {"runtime": {"provider": "CoreMLExecutionProvider"}}

    with pytest.raises(
        RuntimeError,
        match=r"CoreMLExecutionProvider.*unavailable",
    ):
        base_config.resolve_runtime_provider(config)


def test_accelerated_provider_has_no_cpu_fallback_when_cpu_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _resolve(
        monkeypatch,
        available=("CoreMLExecutionProvider",),
    )

    assert runtime["providers"] == ["CoreMLExecutionProvider"]
