from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import DetectionProfile


@dataclass(frozen=True, slots=True)
class EngineSummary:
    model_id: str
    model_digest: str
    embedding_dimension: int
    preprocessing_version: str
    provider: str
    models: tuple[dict[str, object], ...] = ()
    license: dict[str, object] | None = None
    addons: tuple[dict[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_digest": self.model_digest,
            "embedding_dimension": self.embedding_dimension,
            "preprocessing_version": self.preprocessing_version,
            "provider": self.provider,
            "models": [dict(model) for model in self.models],
            "license": dict(self.license) if self.license is not None else None,
            **({"addons": [dict(addon) for addon in self.addons]} if self.addons else {}),
        }


@dataclass(slots=True)
class FaceObservation:
    """Internal face result. Coordinates are pixel-space ``(x1, y1, x2, y2)``."""

    bbox: tuple[float, float, float, float]
    detection_score: float
    landmarks: np.ndarray | None
    embedding: np.ndarray | None
    quality_score: float = 0.0
    sharpness: float = 0.0
    brightness: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    rejection_reasons: list[str] = field(default_factory=list)
    liveness: dict[str, object] | None = None

    @property
    def confidence(self) -> float:
        """Compatibility alias; new API responses call this ``detection_score``."""

        return self.detection_score

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0]) * max(
            0.0, self.bbox[3] - self.bbox[1]
        )

    @property
    def bbox_xywh(self) -> tuple[float, float, float, float]:
        left, top, right, bottom = self.bbox
        return left, top, max(0.0, right - left), max(0.0, bottom - top)

    def normalized_bbox(
        self, image_width: int, image_height: int
    ) -> tuple[float, float, float, float]:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        left, top, right, bottom = self.bbox
        clipped_left = float(np.clip(left, 0.0, image_width))
        clipped_top = float(np.clip(top, 0.0, image_height))
        clipped_right = float(np.clip(right, clipped_left, image_width))
        clipped_bottom = float(np.clip(bottom, clipped_top, image_height))
        return (
            clipped_left / image_width,
            clipped_top / image_height,
            (clipped_right - clipped_left) / image_width,
            (clipped_bottom - clipped_top) / image_height,
        )


@runtime_checkable
class InferenceEngine(Protocol):
    summary: EngineSummary

    def startup(self) -> None: ...

    def analyze(
        self,
        image: np.ndarray,
        *,
        require_embeddings: bool = True,
        max_faces: int | None = None,
        min_score: float | None = None,
        detection_profile: DetectionProfile | None = None,
        single_face: bool = False,
        apply_liveness: bool = True,
    ) -> list[FaceObservation]: ...

    def validate_detection_profile(self, profile: DetectionProfile) -> None: ...

    def runtime_summary(self) -> dict[str, object]: ...

    def close(self) -> None: ...


def validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must be a non-empty BGR array with three channels")
    if image.dtype != np.uint8:
        raise ValueError("image must use uint8 pixels")


def cpu_summary() -> dict[str, object]:
    model = platform.processor().strip()
    cpuinfo = Path("/proc/cpuinfo")
    if not model and cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return {"model": model or "unknown", "logical_cores": os.cpu_count()}


def l2_normalize(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Model returned a zero or non-finite embedding")
    result = vector / norm
    if not np.all(np.isfinite(result)):
        raise ValueError("Model returned a non-finite embedding")
    return result.astype(np.float32, copy=False)


def raw_cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_normalized = l2_normalize(left)
    right_normalized = l2_normalize(right)
    if left_normalized.shape != right_normalized.shape:
        raise ValueError("Embedding dimensions do not match")
    return float(np.clip(np.dot(left_normalized, right_normalized), -1.0, 1.0))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return the raw cosine similarity in the inclusive ``[-1, 1]`` range."""

    return raw_cosine_similarity(left, right)
