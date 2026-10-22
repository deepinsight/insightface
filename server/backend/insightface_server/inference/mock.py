from __future__ import annotations

import hashlib
import platform
import threading

import cv2
import numpy as np

from ..config import (
    DEFAULT_DETECTOR_INPUT_SIZES,
    DetectionProfile,
    default_inference_max_concurrency,
    normalize_detector_input_sizes,
    normalize_single_face_selection,
)
from .base import EngineSummary, FaceObservation, cpu_summary, l2_normalize, validate_image
from .concurrency import InferenceConcurrencyLimiter
from .quality import enrich_quality


class MockInferenceEngine:
    """Deterministic synthetic engine for public CI and explicit development mode."""

    def __init__(
        self,
        provider: str = "CPUExecutionProvider",
        embedding_dimension: int = 512,
        detector_input_sizes: object = DEFAULT_DETECTOR_INPUT_SIZES,
        detection_profile: DetectionProfile | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self.summary = EngineSummary(
            model_id="synthetic-recognition",
            model_digest=hashlib.sha256(b"simple-insightface-server-mock-v1").hexdigest(),
            embedding_dimension=embedding_dimension,
            preprocessing_version="mock-1",
            provider=provider,
            models=(
                {
                    "model_id": "synthetic-detection",
                    "task": "face_detection",
                    "sha256": hashlib.sha256(b"synthetic-detection").hexdigest(),
                },
                {
                    "model_id": "synthetic-recognition",
                    "task": "face_recognition",
                    "sha256": hashlib.sha256(b"synthetic-recognition").hexdigest(),
                },
            ),
        )
        self._embedding_dimension = embedding_dimension
        self._default_detection_profile = detection_profile or DetectionProfile(
            input_sizes=normalize_detector_input_sizes(detector_input_sizes)
        )
        self._detector_input_sizes = self._default_detection_profile.input_sizes
        self._started = False
        self._closed = False
        self._lifecycle_lock = threading.RLock()
        self._concurrency = InferenceConcurrencyLimiter(
            max_concurrency or default_inference_max_concurrency(provider)
        )

    def startup(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Inference engine is closed")
            self._started = True

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
    ) -> list[FaceObservation]:
        validate_image(image)
        if max_faces is not None and max_faces <= 0:
            raise ValueError("max_faces must be positive")
        if min_score is not None and not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        with self._concurrency.slot():
            with self._lifecycle_lock:
                if not self._started or self._closed:
                    raise RuntimeError("Inference engine has not been started")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if min(image.shape[:2]) < 16 or float(np.std(gray)) < 1.0:
                return []

            height, width = image.shape[:2]
            face_count = 2 if width >= height * 1.8 else 1
            observations = [
                self._observation(image, index=index, count=face_count, embeddings=require_embeddings)
                for index in range(face_count)
            ]
            profile = detection_profile or self._default_detection_profile
            self.validate_detection_profile(profile)
            threshold = profile.threshold if min_score is None else max(profile.threshold, min_score)
            observations = [face for face in observations if face.detection_score >= threshold]
            observations.sort(key=lambda face: face.area, reverse=True)
            if single_face and observations:
                def selection_score(face):
                    if profile.single_face_selection == "largest":
                        return face.area
                    left, top, right, bottom = face.bbox
                    return face.area - 2.0 * (
                        ((left + right - width) * 0.5) ** 2
                        + ((top + bottom - height) * 0.5) ** 2
                    )
                observations = [max(observations, key=selection_score)]
            return observations if max_faces is None else observations[:max_faces]

    def validate_detection_profile(self, profile: DetectionProfile) -> None:
        normalize_detector_input_sizes(profile.input_sizes)
        if not 0.0 <= profile.threshold <= 1.0:
            raise ValueError("detection threshold must be between 0 and 1")
        if not 0.0 <= profile.nms_threshold <= 1.0:
            raise ValueError("detection nms_threshold must be between 0 and 1")
        normalize_single_face_selection(profile.single_face_selection)

    def _observation(
        self, image: np.ndarray, *, index: int, count: int, embeddings: bool
    ) -> FaceObservation:
        height, width = image.shape[:2]
        side = float(min(height, width) * (0.62 if index == 0 else 0.46))
        center_x = width * ((index + 1) / (count + 1))
        center_y = height / 2.0
        left, top = center_x - side / 2.0, center_y - side / 2.0
        right, bottom = center_x + side / 2.0, center_y + side / 2.0
        landmarks = np.asarray(
            [
                [center_x - side * 0.18, center_y - side * 0.12],
                [center_x + side * 0.18, center_y - side * 0.12],
                [center_x, center_y],
                [center_x - side * 0.13, center_y + side * 0.20],
                [center_x + side * 0.13, center_y + side * 0.20],
            ],
            dtype=np.float32,
        )
        embedding = None
        if embeddings:
            resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
            seed = hashlib.sha256(resized.tobytes() + index.to_bytes(2, "big")).digest()
            generator = np.random.default_rng(int.from_bytes(seed[:8], "big"))
            embedding = l2_normalize(
                generator.standard_normal(self._embedding_dimension, dtype=np.float32)
            )
        face = FaceObservation(
            bbox=(left, top, right, bottom),
            detection_score=0.99 - index * 0.02,
            landmarks=landmarks,
            embedding=embedding,
        )
        enrich_quality(image, face)
        return face

    def runtime_summary(self) -> dict[str, object]:
        with self._lifecycle_lock:
            summary: dict[str, object] = {
                "mode": "mock",
                "os": platform.platform(),
                "architecture": platform.machine(),
                "cpu": cpu_summary(),
                "execution_provider": self.summary.provider,
                "detector_input_sizes": [list(size) for size in self._detector_input_sizes],
                "detection_profile": self._default_detection_profile.as_dict(),
                "strict_provider_audit": {},
            }
        summary["inference_concurrency"] = self._concurrency.summary()
        return summary

    def runtime_details(self) -> dict[str, object]:
        """Compatibility alias for internal callers during the server transition."""

        return self.runtime_summary()

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._started = False
        self._concurrency.close_and_wait()
