"""InsightFace runtime wrapper used by the GUI."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from ...app.face_analysis import FaceAnalysis
from ...model_zoo.onnxruntime_utils import get_default_providers
from ...model_zoo.package_manifest import (
    SUPPORTED_MANIFEST_PACKAGES,
    has_model_package_manifest,
    load_model_package,
)
from .constants import AUTO_DET_SIZES, DEFAULT_DET_SIZE, DEFAULT_MODEL_NAME, DEFAULT_THRESHOLD
from .i18n import tr
from .logging import get_logger
from .models import CompareResult, FaceRecord
from .model_packages import PERSON_MODEL_PACKAGES
from .quality import score_face
from .recognition import compare_embeddings, cosine_similarity, normalize_embedding
from .utils import crop_bbox

LOGGER = get_logger("face_engine")


class ModelNotLoadedError(RuntimeError):
    pass


class FaceEngine:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        providers: Optional[Iterable[str]] = None,
        det_size: tuple[int, int] = DEFAULT_DET_SIZE,
        root: str | os.PathLike[str] | None = None,
        custom_model_dir: str | os.PathLike[str] | None = None,
    ):
        self.model_name = model_name
        self.root = Path(os.path.expanduser(str(root or "~/.insightface")))
        self.custom_model_dir = Path(os.path.expanduser(str(custom_model_dir))) if custom_model_dir else None
        requested_providers = list(providers) if providers is not None else []
        self.requested_providers = (
            requested_providers or get_default_providers()
        )
        self.det_size = det_size
        self.det_sizes = self._resolve_det_sizes(det_size)
        self.auto_det_size = len(self.det_sizes) > 1
        self.models: Dict[str, Any] = {}
        self.det_model = None
        self.face_analysis: Optional[FaceAnalysis] = None
        self.loaded = False
        self.last_error = ""
        self.embedding_dim: Optional[int] = None
        self.last_latency_ms: Dict[str, float] = {}
        self.active_providers: List[str] = []
        self.ctx_id = -1
        self._prepared_det_size = None
        self._lock = threading.RLock()

    def is_loaded(self) -> bool:
        return self.loaded and self.det_model is not None

    def resolve_model_dir(self) -> Path:
        if self.custom_model_dir and self.custom_model_dir.exists():
            return self.custom_model_dir
        model_path = Path(os.path.expanduser(self.model_name))
        if model_path.exists() and model_path.is_dir():
            return model_path
        return self.root / "models" / self.model_name

    def load(self) -> None:
        with self._lock:
            self.loaded = False
            self.last_error = ""
            self.models = {}
            self.det_model = None
            self.face_analysis = None
            self.embedding_dim = None
            self.active_providers = []
            self.ctx_id = -1
            self._prepared_det_size = None
            if self.model_name in PERSON_MODEL_PACKAGES and not self._custom_model_directory_selected():
                self.last_error = (
                    "Cheetah packages require PersonAnalysis. Use Person Analysis "
                    "or select a face model in Models."
                )
                LOGGER.warning(self.last_error)
                return
            start = time.perf_counter()
            model_dir = self.resolve_model_dir()
            if not model_dir.is_dir():
                self.last_error = (
                    f"Model directory not found: {model_dir}. Configure a local model directory "
                    "or install the model pack first."
                )
                LOGGER.warning(self.last_error)
                return
            official_manifest_package = (
                self.model_name in SUPPORTED_MANIFEST_PACKAGES
                and not self._custom_model_directory_selected()
            )
            if official_manifest_package:
                try:
                    package = load_model_package(model_dir)
                    if package.name != self.model_name:
                        raise ValueError(
                            f"manifest model_id {package.name!r} does not match "
                            f"the selected package {self.model_name!r}"
                        )
                    missing_tasks = [
                        task
                        for task in ("detection", "recognition")
                        if task not in package.tasks
                    ]
                    if missing_tasks:
                        raise ValueError(
                            "manifest is missing required GUI task(s): "
                            + ", ".join(missing_tasks)
                        )
                except Exception as exc:
                    self.last_error = (
                        f"Official model package {self.model_name!r} requires a valid "
                        f"V2 manifest: {exc}"
                    )
                    LOGGER.warning(self.last_error)
                    return
                manifest_backed = True
            else:
                manifest_backed = has_model_package_manifest(model_dir)
            if not manifest_backed and not any(model_dir.glob("*.onnx")):
                self.last_error = f"No .onnx model files found in {model_dir}."
                LOGGER.warning(self.last_error)
                return
            try:
                analysis = self._create_face_analysis(
                    model_dir,
                    manifest_backed=manifest_backed,
                )
                if "detection" not in analysis.models:
                    self.last_error = f"No detection model found in {model_dir}."
                    return
                ctx_id = -1 if self._primary_provider() == "CPUExecutionProvider" else 0
                analysis.prepare(
                    ctx_id,
                    det_size=self._detector_input_size(),
                    det_thresh=0.5,
                )
                self.ctx_id = ctx_id
                self.face_analysis = analysis
                self.models = analysis.models
                self.det_model = analysis.det_model
                self._prepared_det_size = self._detector_input_size()
                self.loaded = True
                self.embedding_dim = self._infer_embedding_dim()
                self.active_providers = self.requested_providers
                self.last_latency_ms["load"] = (time.perf_counter() - start) * 1000.0
            except AssertionError as exc:
                detail = str(exc).strip()
                self.last_error = (
                    f"Model load failed: {detail}"
                    if detail
                    else f"No detection model found in {model_dir}."
                )
                LOGGER.exception(self.last_error)
                self.loaded = False
            except Exception as exc:
                self.last_error = f"Model load failed: {exc}"
                LOGGER.exception(self.last_error)
                self.loaded = False

    def _custom_model_directory_selected(self) -> bool:
        if self.custom_model_dir and self.custom_model_dir.is_dir():
            return True
        return Path(os.path.expanduser(self.model_name)).is_dir()

    def _create_face_analysis(
        self,
        model_dir: Path,
        *,
        manifest_backed: bool,
    ) -> FaceAnalysis:
        """Create a local-only model host for the GUI's supported tasks."""

        kwargs: Dict[str, Any] = {
            "providers": self.requested_providers,
            "static_shape_sessions": True,
        }
        session_options = self._quiet_onnxruntime_session_options()
        if session_options is not None:
            kwargs["sess_options"] = session_options
        if self._primary_provider() == "CoreMLExecutionProvider":
            kwargs["_coreml_detector_input_size"] = max(
                self.det_sizes,
                key=lambda value: int(value[0]) * int(value[1]),
            )
        allowed_modules = (
            ("detection", "recognition") if manifest_backed else None
        )
        # Passing the verified directory, rather than model_name/root, keeps
        # the GUI on FaceAnalysis's local-directory path and prevents an
        # implicit model download.
        return FaceAnalysis(
            name=model_dir.resolve(),
            allowed_modules=allowed_modules,
            **kwargs,
        )

    def _primary_provider(self) -> str:
        provider = self.requested_providers[0] if self.requested_providers else ""
        if isinstance(provider, (list, tuple)) and provider:
            return str(provider[0])
        return str(provider)

    @staticmethod
    def _quiet_onnxruntime_session_options():
        try:
            import onnxruntime

            onnxruntime.set_default_logger_severity(3)
            session_options = onnxruntime.SessionOptions()
            session_options.log_severity_level = 3
            return session_options
        except Exception:
            return None

    def _infer_embedding_dim(self) -> Optional[int]:
        for model in self.models.values():
            if getattr(model, "taskname", "") == "recognition":
                output_shape = getattr(model, "output_shape", None)
                if output_shape:
                    try:
                        return int(output_shape[-1])
                    except Exception:
                        return None
        return None

    def warmup(self) -> Dict[str, Any]:
        if not self.is_loaded():
            raise ModelNotLoadedError("Model is not loaded. Please open Models.")
        warmup_width = max(size[0] for size in self.det_sizes)
        warmup_height = max(size[1] for size in self.det_sizes)
        image = np.zeros((warmup_height, warmup_width, 3), dtype=np.uint8)
        start = time.perf_counter()
        self.detect_faces(image)
        elapsed = (time.perf_counter() - start) * 1000.0
        self.last_latency_ms["warmup"] = elapsed
        return {"warmup_ms": elapsed}

    def detect_faces(self, image: np.ndarray, source_path: Optional[str] = None) -> List[FaceRecord]:
        with self._lock:
            if not self.is_loaded():
                raise ModelNotLoadedError("Model is not loaded. Please open Models.")
            if image is None:
                return []
            image = np.ascontiguousarray(image)
            analysis = self.face_analysis
            if analysis is None:
                raise ModelNotLoadedError("Model is not loaded. Please open Models.")

            start = time.perf_counter()
            faces = analysis.get(image, max_num=0, det_metric="default")
            records: List[FaceRecord] = []
            if not faces:
                self.last_latency_ms["detect"] = (time.perf_counter() - start) * 1000.0
                return records
            for idx, face in enumerate(faces):
                bbox = np.asarray(face.bbox, dtype=np.float32).reshape(-1)[:4]
                det_score = float(face.det_score)
                kps = face.kps
                embedding = getattr(face, "embedding", None)
                normed_embedding = getattr(face, "normed_embedding", None)
                if normed_embedding is None and embedding is not None:
                    normed_embedding = normalize_embedding(embedding)
                crop = crop_bbox(image, bbox)
                quality_score, quality_flags = score_face(image, bbox, kps, det_score)
                records.append(
                    FaceRecord(
                        face_id=f"{source_path or 'image'}:{idx}",
                        bbox=[float(v) for v in bbox],
                        kps=kps.tolist() if kps is not None else None,
                        det_score=det_score,
                        embedding=np.asarray(embedding, dtype=np.float32).reshape(-1) if embedding is not None else None,
                        normed_embedding=(
                            np.asarray(normed_embedding, dtype=np.float32).reshape(-1)
                            if normed_embedding is not None
                            else None
                        ),
                        gender=getattr(face, "gender", None),
                        age=getattr(face, "age", None),
                        quality_score=quality_score,
                        quality_flags=quality_flags,
                        crop=crop,
                        source_path=source_path,
                    )
                )
            self.last_latency_ms["detect"] = (time.perf_counter() - start) * 1000.0
            return records

    def detect_best_face(self, image: np.ndarray, source_path: Optional[str] = None) -> Optional[FaceRecord]:
        faces = self.detect_faces(image, source_path=source_path)
        if not faces:
            return None
        return max(
            faces,
            key=lambda face: (
                (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
                face.det_score,
            ),
        )

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return cosine_similarity(emb1, emb2)

    def compare_images(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: float = DEFAULT_THRESHOLD,
        path1: Optional[str] = None,
        path2: Optional[str] = None,
    ) -> CompareResult:
        with self._lock:
            face_a = self.detect_best_face(image1, source_path=path1)
            face_b = self.detect_best_face(image2, source_path=path2)
            if face_a is None:
                raise ValueError("No face detected in Image A.")
            if face_b is None:
                raise ValueError("No face detected in Image B.")
            if face_a.normed_embedding is None or face_b.normed_embedding is None:
                raise ValueError("Recognition embedding is unavailable for one or both faces.")
            comparison = compare_embeddings(face_a.normed_embedding, face_b.normed_embedding, threshold)
            notes = []
            for label, face in (("Image A", face_a), ("Image B", face_b)):
                if face.quality_score is not None and face.quality_score < 0.45:
                    notes.append(f"{label} has low quality. Recognition may be unreliable.")
                if face.quality_flags:
                    notes.append(f"{label}: {', '.join(face.quality_flags)}")
            return CompareResult(
                similarity=float(comparison["similarity"]),
                threshold=threshold,
                decision=str(comparison["decision"]),
                face_a=face_a,
                face_b=face_b,
                notes=notes,
            )

    def get_runtime_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_dir": str(self.resolve_model_dir()),
            "providers": self.active_providers or self.requested_providers,
            "detector": type(self.det_model).__name__ if self.det_model is not None else None,
            "det_size": self._det_size_label(),
            "embedding_dim": self.embedding_dim,
            "loaded": self.is_loaded(),
            "last_error": self.last_error,
            "last_latency_ms": self.last_latency_ms,
        }

    def _resolve_det_sizes(self, det_size: tuple[int, int]) -> list[tuple[int, int]]:
        width, height = int(det_size[0]), int(det_size[1])
        if width <= 0 or height <= 0:
            return [(int(width), int(height)) for width, height in AUTO_DET_SIZES]
        return [(width, height)]

    def _det_size_label(self) -> str:
        if self.auto_det_size:
            return "Auto (" + " + ".join(f"{width}x{height}" for width, height in self.det_sizes) + ")"
        width, height = self.det_sizes[0]
        return f"{width}x{height}"

    def _detector_input_size(self):
        if self.auto_det_size:
            return list(self.det_sizes)
        return self.det_sizes[0]

def available_execution_providers() -> List[str]:
    try:
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        return [str(provider) for provider in providers]
    except Exception as exc:
        LOGGER.debug("Unable to inspect ONNX Runtime providers: %s", exc)
        return []


def is_cuda_provider_available() -> bool:
    return "CUDAExecutionProvider" in available_execution_providers()


def providers_from_choice(choice: str) -> List[str]:
    normalized = (choice or "CPU").strip().lower()
    if normalized == "cpu":
        return ["CPUExecutionProvider"]
    if normalized == "cuda":
        available = available_execution_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider"]
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
            return providers
        LOGGER.warning("CUDA provider was requested but CUDAExecutionProvider is not available; using CPU.")
        return ["CPUExecutionProvider"]
    available = available_execution_providers()
    return get_default_providers(available)


def provider_runtime_display(
    choice: str,
    language: str | None = None,
) -> tuple[str, str]:
    """Describe the provider chain the GUI can use for a saved choice.

    ``Auto`` is a configuration policy, not an execution provider.  Display
    surfaces should show the resolved primary provider while keeping the full
    fallback chain in the tooltip.  The settings control itself still shows
    ``Auto`` so users can retain automatic selection.
    """

    configured = str(choice or "Auto").strip() or "Auto"

    def localized(text: str) -> str:
        # Preserve the historical English contract when callers omit language.
        return tr(text, language) if language is not None else text

    available = available_execution_providers()
    if not available:
        return (
            localized("Unavailable"),
            localized("ONNX Runtime reports no available execution providers."),
        )
    try:
        selected = [
            provider
            for provider in providers_from_choice(configured)
            if provider in available
        ]
    except Exception as exc:
        LOGGER.debug("Unable to resolve GUI execution provider: %s", exc)
        selected = []
    if not selected:
        return (
            localized("Unavailable"),
            localized(
                "Configured selection: {selection}. No matching ONNX Runtime "
                "execution provider is currently available."
            ).format(selection=configured),
        )
    primary, *fallbacks = selected
    if fallbacks:
        chain = primary + "".join(
            " → "
            + localized("{provider} (fallback)").format(provider=provider)
            for provider in fallbacks
        )
    else:
        chain = primary
    return (
        primary,
        localized(
            "Configured selection: {selection}. Resolved provider chain: {chain}."
        ).format(selection=configured, chain=chain),
    )
