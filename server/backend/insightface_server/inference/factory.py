from __future__ import annotations

from typing import Any

from ..config import (
    DEFAULT_DETECTOR_INPUT_SIZES,
    DetectionProfile,
    default_inference_max_concurrency,
)
from .base import InferenceEngine
from .mock import MockInferenceEngine


def _setting(settings: object, name: str, default: Any = None) -> Any:
    if isinstance(settings, dict):
        return settings.get(name, default)
    return getattr(settings, name, default)


def create_engine(settings: object) -> InferenceEngine:
    """Create, but do not start, the configured process-wide inference engine."""

    mode = str(_setting(settings, "inference_mode", "onnx")).lower()
    provider = str(_setting(settings, "execution_provider", "CPUExecutionProvider"))
    configured_concurrency = _setting(settings, "inference_max_concurrency", None)
    max_concurrency = (
        default_inference_max_concurrency(provider)
        if configured_concurrency is None
        else int(configured_concurrency)
    )
    if mode == "mock":
        if _setting(settings, "addons", ()):
            raise ValueError("Model addons require INSIGHTFACE_INFERENCE_MODE=onnx")
        default_profile = getattr(settings, "detection_profile", None)
        if not isinstance(default_profile, DetectionProfile):
            default_profile = None
        return MockInferenceEngine(
            provider=provider,
            embedding_dimension=int(_setting(settings, "embedding_dimension", 512)),
            detector_input_sizes=_setting(
                settings, "detector_input_sizes", DEFAULT_DETECTOR_INPUT_SIZES
            ),
            detection_profile=default_profile,
            max_concurrency=max_concurrency,
        )
    if mode != "onnx":
        raise ValueError(f"Unsupported inference_mode: {mode}")

    from .onnx_engine import OnnxInsightFaceEngine

    return OnnxInsightFaceEngine(settings)
