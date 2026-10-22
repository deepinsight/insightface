"""Shared ONNX Runtime provider selection for InsightFace inference."""

from __future__ import annotations

import threading
from collections.abc import Sequence

import onnxruntime


DEFAULT_PROVIDER_PRIORITY = (
    "CoreMLExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)
_CUDA_PRELOAD_LOCK = threading.Lock()
_CUDA_PRELOAD_COMPLETE = False


def preload_cuda_libraries(providers: Sequence[object]) -> None:
    """Ask ORT to discover CUDA/cuDNN installed in the active environment."""

    global _CUDA_PRELOAD_COMPLETE
    names = [
        str(provider[0] if isinstance(provider, tuple) else provider)
        for provider in providers
    ]
    if _CUDA_PRELOAD_COMPLETE or "CUDAExecutionProvider" not in names:
        return
    preload = getattr(onnxruntime, "preload_dlls", None)
    if not callable(preload):
        return
    with _CUDA_PRELOAD_LOCK:
        if _CUDA_PRELOAD_COMPLETE:
            return
        preload()
        _CUDA_PRELOAD_COMPLETE = True


def get_default_providers(
    available_providers: Sequence[str] | None = None,
) -> list[str]:
    """Select one preferred execution provider plus CPU fallback.

    Explicit provider lists supplied by callers never use this policy. Unknown
    providers are retained only as a last resort for custom ORT builds that do
    not expose CoreML, CUDA, or CPU.
    """

    inspect_runtime = available_providers is None
    available = list(
        dict.fromkeys(
            str(provider)
            for provider in (
                onnxruntime.get_available_providers()
                if available_providers is None
                else available_providers
            )
        )
    )
    for provider in DEFAULT_PROVIDER_PRIORITY:
        if provider not in available:
            continue
        selected = [provider]
        if (
            provider != "CPUExecutionProvider"
            and "CPUExecutionProvider" in available
        ):
            selected.append("CPUExecutionProvider")
        if inspect_runtime:
            preload_cuda_libraries(selected)
        return selected
    if available:
        selected = [available[0]]
        if inspect_runtime:
            preload_cuda_libraries(selected)
        return selected
    raise RuntimeError("ONNX Runtime reports no available execution providers")


__all__ = [
    "DEFAULT_PROVIDER_PRIORITY",
    "get_default_providers",
    "preload_cuda_libraries",
]
