"""Bounded-latency streaming ONNX face tracking pipeline.

The public pipeline callables are resolved lazily so importing the package or
starting ``doctor`` does not require the optional PyAV runtime first.
"""

from __future__ import annotations

import importlib
from typing import Any

from .output_paths import PrivateFrameOutputPaths, default_output_paths

_PIPELINE_EXPORTS = {
    "analyze_streaming_pipeline",
    "render_streaming_artifacts",
    "run_streaming_pipeline",
}


def __getattr__(name: str) -> Any:
    if name not in _PIPELINE_EXPORTS:
        raise AttributeError(name)
    pipeline = importlib.import_module(f"{__name__}.pipeline")
    value = getattr(pipeline, name)
    globals()[name] = value
    return value


__all__ = [
    "PrivateFrameOutputPaths",
    "analyze_streaming_pipeline",
    "default_output_paths",
    "render_streaming_artifacts",
    "run_streaming_pipeline",
]
