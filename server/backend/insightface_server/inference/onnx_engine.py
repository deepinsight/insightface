from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..addons import LivenessUnavailable, addon_summary, require_installed_addon
from ..config import (
    DEFAULT_DETECTOR_INPUT_SIZES,
    DEFAULT_DETECTOR_NMS_THRESHOLD,
    DetectionProfile,
    default_inference_max_concurrency,
    normalize_detector_input_sizes,
    normalize_single_face_selection,
)
from ..licensing import ModelLicenseError, inspect_model_package_license
from ..models import ModelBundle, ModelSpec, load_manifest
from .base import EngineSummary, FaceObservation, cpu_summary, l2_normalize, validate_image
from .concurrency import InferenceConcurrencyLimiter
from .quality import enrich_quality

_CPU_METADATA_OPERATORS = frozenset(
    {"Cast", "Concat", "Gather", "Shape", "Size", "Slice", "Squeeze", "Unsqueeze"}
)
_CPU_METADATA_DTYPES = frozenset({"int32", "int64"})
_MAX_CPU_METADATA_OUTPUT_BYTES = 4096
_SUPPORTED_COMPUTE_CAPABILITIES = frozenset(
    {(7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0)}
)
_BLACKWELL_COMPUTE_CAPABILITIES = frozenset({(10, 0), (10, 3), (12, 0)})


def _setting(settings: object, name: str, default: Any = None) -> Any:
    if isinstance(settings, dict):
        return settings.get(name, default)
    return getattr(settings, name, default)


def _integer_metadata_shapes(value: object) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for tensor in value:
        if not isinstance(tensor, dict) or len(tensor) != 1:
            return False
        dtype, shape = next(iter(tensor.items()))
        if dtype not in _CPU_METADATA_DTYPES or not isinstance(shape, list):
            return False
        if any(not isinstance(dimension, int) or dimension < 0 for dimension in shape):
            return False
    return True


def audit_cuda_profile(events: object, *, model_name: str) -> dict[str, object]:
    """Reject silent CPU model compute while permitting tiny shape bookkeeping nodes."""

    if not isinstance(events, list):
        raise RuntimeError(f"Invalid ONNX Runtime profile for {model_name}")
    cuda_kernels = 0
    cpu_metadata_operators: Counter[str] = Counter()
    rejected: list[dict[str, object]] = []
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "Node":
            continue
        args = event.get("args")
        if not isinstance(args, dict):
            continue
        provider = args.get("provider")
        operator = args.get("op_name")
        if provider == "CUDAExecutionProvider":
            cuda_kernels += 1
            continue
        if not provider:
            continue
        if provider != "CPUExecutionProvider":
            rejected.append(
                {
                    "name": event.get("name"),
                    "operator": operator,
                    "provider": provider,
                }
            )
            continue
        try:
            output_bytes = int(args.get("output_size", -1))
        except (TypeError, ValueError):
            output_bytes = -1
        metadata_only = (
            operator in _CPU_METADATA_OPERATORS
            and 0 <= output_bytes <= _MAX_CPU_METADATA_OUTPUT_BYTES
            and _integer_metadata_shapes(args.get("output_type_shape"))
        )
        if metadata_only:
            cpu_metadata_operators[str(operator)] += 1
        else:
            rejected.append(
                {
                    "name": event.get("name"),
                    "operator": operator,
                    "provider": provider,
                    "output_bytes": output_bytes,
                    "output_type_shape": args.get("output_type_shape"),
                }
            )
    if cuda_kernels == 0:
        raise RuntimeError(f"GPU strict audit found no CUDA kernels for {model_name}")
    if rejected:
        compact = json.dumps(rejected, separators=(",", ":"), sort_keys=True)
        raise RuntimeError(
            f"GPU strict audit rejected CPU/non-CUDA compute for {model_name}: {compact}"
        )
    return {
        "accepted": True,
        "cuda_kernel_count": cuda_kernels,
        "cpu_shape_kernel_count": sum(cpu_metadata_operators.values()),
        "cpu_shape_operators": dict(sorted(cpu_metadata_operators.items())),
        "policy": "CUDA compute required; CPU limited to small integer shape metadata",
    }


def _cuda_runtime_version() -> str | None:
    try:
        library = ctypes.CDLL("libcudart.so.12")
        value = ctypes.c_int()
        if library.cudaRuntimeGetVersion(ctypes.byref(value)) != 0:
            return None
    except (OSError, AttributeError, ValueError):
        return None
    raw = int(value.value)
    return f"{raw // 1000}.{(raw % 1000) // 10}"


def _cudnn_version() -> str | None:
    try:
        library = ctypes.CDLL("libcudnn.so.9")
        getter = library.cudnnGetVersion
        getter.restype = ctypes.c_size_t
        raw = int(getter())
    except (OSError, AttributeError, ValueError):
        return None
    if raw <= 0:
        return None
    return f"{raw // 10000}.{(raw % 10000) // 100}.{raw % 100}"


def _gpu_details() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    gpus: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) == 3:
            gpus.append(
                {
                    "name": values[0],
                    "compute_capability": values[1],
                    "driver_version": values[2],
                }
            )
    return gpus


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for item in value.split("."):
        digits = "".join(character for character in item if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def validate_gpu_requirements(gpus: list[dict[str, str]], cuda: str | None) -> None:
    if not gpus:
        raise RuntimeError("CUDA Session exists but NVIDIA GPU diagnostics failed")
    for gpu in gpus:
        capability = _version_tuple(gpu["compute_capability"])
        driver = _version_tuple(gpu["driver_version"])
        architecture = capability[:2]
        if architecture not in _SUPPORTED_COMPUTE_CAPABILITIES:
            raise RuntimeError(
                f"Unsupported CUDA Compute Capability {gpu['compute_capability']} "
                f"on {gpu['name']}"
            )
        if architecture in _BLACKWELL_COMPUTE_CAPABILITIES:
            if driver < (570, 26):
                raise RuntimeError(
                    "Blackwell requires NVIDIA Driver 570.26 or newer; "
                    f"found {gpu['driver_version']}"
                )
            if cuda is None or _version_tuple(cuda) < (12, 8):
                raise RuntimeError(
                    f"Blackwell requires CUDA Runtime 12.8 or newer; found {cuda}"
                )
        elif driver < (535,):
            raise RuntimeError(
                "Turing through Hopper require NVIDIA Driver R535 or newer; "
                f"found {gpu['driver_version']}"
            )


class OnnxInsightFaceEngine:
    """Shared SCRFD + ArcFace implementation with exactly one Session per model."""

    def __init__(self, settings: object) -> None:
        self._settings = settings
        self._provider = str(
            _setting(settings, "execution_provider", "CPUExecutionProvider")
        )
        if self._provider not in {"CPUExecutionProvider", "CUDAExecutionProvider"}:
            raise ValueError(f"Unsupported execution_provider: {self._provider}")
        models_dir = Path(str(_setting(settings, "models_dir", "/models")))
        self._liveness_path = (
            require_installed_addon("liveness", models_dir)
            if "liveness" in _setting(settings, "addons", ()) else None
        )
        self._liveness_mode = _setting(settings, "liveness_mode", "normal")
        self._liveness_threshold = _setting(settings, "liveness_threshold", 0.8)
        self._liveness = None
        self._liveness_session = None
        self.bundle: ModelBundle = load_manifest(models_dir)
        license_inspection = inspect_model_package_license(
            models_dir,
            expected_model_id=self.bundle.model_id,
        )
        if license_inspection.verified:
            assert license_inspection.license is not None
            license_summary = license_inspection.license.public_summary()
        elif license_inspection.defaulted:
            license_summary = license_inspection.public_summary()
        else:
            raise ModelLicenseError(
                f"Model license {license_inspection.status}: "
                f"{license_inspection.message}"
            )
        recognizer = self.bundle.recognizer
        bundle_identity = [
            {
                "sha256": model.sha256,
                "input_size": model.input_size,
                "input_mean": model.input_mean,
                "input_std": model.input_std,
                "preprocessing_version": model.preprocessing_version,
            }
            for model in self.bundle.models
        ]
        bundle_digest = hashlib.sha256(
            json.dumps(bundle_identity, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        self.summary = EngineSummary(
            model_id=self.bundle.model_id,
            # Collection semantics include detection/alignment as well as recognition.
            model_digest=bundle_digest,
            embedding_dimension=int(recognizer.embedding_dimension or 0),
            preprocessing_version=recognizer.preprocessing_version,
            provider=self._provider,
            models=tuple(model.public_summary() for model in self.bundle.models),
            license=license_summary,
            addons=(addon_summary("liveness"),) if self._liveness_path else (),
        )
        self._device_id = int(_setting(settings, "device_id", 0))
        self._detector_threshold = float(_setting(settings, "detector_threshold", 0.50))
        self._detector_input_sizes = normalize_detector_input_sizes(
            _setting(settings, "detector_input_sizes", DEFAULT_DETECTOR_INPUT_SIZES)
        )
        self._detector_nms_threshold = float(
            _setting(settings, "detector_nms_threshold", DEFAULT_DETECTOR_NMS_THRESHOLD)
        )
        if not 0.0 <= self._detector_threshold <= 1.0:
            raise ValueError("detector_threshold must be between 0 and 1")
        if not 0.0 <= self._detector_nms_threshold <= 1.0:
            raise ValueError("detector_nms_threshold must be between 0 and 1")
        self._default_detection_profile = DetectionProfile.from_mapping(
            {
                "input_sizes": self._detector_input_sizes,
                "threshold": self._detector_threshold,
                "nms_threshold": self._detector_nms_threshold,
                "single_face_selection": _setting(
                    settings, "detector_single_face_selection", "largest"
                ),
            }
        )
        configured_concurrency = _setting(settings, "inference_max_concurrency", None)
        self._max_concurrency = (
            default_inference_max_concurrency(self._provider)
            if configured_concurrency is None
            else int(configured_concurrency)
        )
        self._concurrency = InferenceConcurrencyLimiter(self._max_concurrency)
        self._lifecycle_lock = threading.RLock()
        self._startup_attempted = False
        self._started = False
        self._closed = False
        self._startup_error: BaseException | None = None
        self._detector: Any = None
        self._recognizer: Any = None
        self._detector_session: Any = None
        self._recognizer_session: Any = None
        self._runtime: dict[str, object] = {}

    def _new_session(self, path: Path) -> Any:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self._provider == "CUDAExecutionProvider":
            options.enable_profiling = True
            profile_id = hashlib.sha256(str(path).encode()).hexdigest()[:12]
            options.profile_file_prefix = f"/tmp/insightface-ort-{profile_id}"
            providers: list[Any] = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": self._device_id,
                        "do_copy_in_default_stream": True,
                    },
                ),
                # Dynamic SCRFD graphs contain a few shape-only nodes which ORT assigns
                # to CPU. The warm-up profile below audits every such assignment.
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=providers
        )
        actual = session.get_providers()
        if not actual or actual[0] != self._provider:
            raise RuntimeError(
                f"Required primary {self._provider} is unavailable; "
                f"Session providers are {actual}"
            )
        return session

    @staticmethod
    def _configure_image_preprocessing(model: Any, session: Any, spec: ModelSpec) -> None:
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError(
                f"{spec.task} requires exactly one model input; received {len(inputs)}"
            )
        input_type = str(inputs[0].type)
        if input_type not in {"tensor(float)", "tensor(uint8)"}:
            raise RuntimeError(
                f"{spec.task} preprocessing requires a float32 or uint8 input; "
                f"received {input_type}"
            )
        if spec.preprocessing == "embedded":
            input_dtype = np.uint8 if input_type == "tensor(uint8)" else np.float32
        elif spec.preprocessing == "mean_std":
            if input_type != "tensor(float)":
                raise RuntimeError(
                    f"{spec.task} mean/std preprocessing requires tensor(float) input"
                )
            input_dtype = np.float32
        else:  # pragma: no cover - manifest construction owns this invariant
            raise RuntimeError(f"Unsupported {spec.task} preprocessing mode: {spec.preprocessing}")
        model.preprocessing = spec.preprocessing
        model.input_mean = spec.input_mean
        model.input_std = spec.input_std
        model.input_type = input_type
        model.input_dtype = input_dtype

    @staticmethod
    def _finish_cuda_profile(session: Any, *, model_name: str) -> dict[str, object]:
        profile_value = session.end_profiling()
        if not profile_value:
            raise RuntimeError(f"ONNX Runtime did not produce a profile for {model_name}")
        profile_path = Path(profile_value)
        try:
            events = json.loads(profile_path.read_text(encoding="utf-8"))
            return audit_cuda_profile(events, model_name=model_name)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unable to audit ONNX Runtime profile for {model_name}"
            ) from exc
        finally:
            profile_path.unlink(missing_ok=True)

    def startup(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Inference engine is closed")
            if self._started:
                return
            if self._startup_attempted:
                message = "Previous inference engine startup failed"
                raise RuntimeError(message) from self._startup_error
            self._startup_attempted = True
            try:
                self._startup_once()
            except BaseException as exc:
                self._startup_error = exc
                self._detector = None
                self._recognizer = None
                self._detector_session = None
                self._recognizer_session = None
                raise
            self._started = True

    def _startup_once(self) -> None:
        import onnxruntime as ort
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX
        from insightface.model_zoo.scrfd import SCRFD

        available = list(ort.get_available_providers())
        if self._provider not in available:
            raise RuntimeError(
                f"Required {self._provider} is unavailable; ONNX Runtime reports {available}"
            )

        # Sessions are constructed once and shared by requests and RTSP frames.
        self._detector_session = self._new_session(self.bundle.detector.path)
        self._recognizer_session = self._new_session(self.bundle.recognizer.path)
        self._detector = SCRFD(
            model_file=str(self.bundle.detector.path),
            session=self._detector_session,
            static_shape_sessions=False,
        )
        self._configure_image_preprocessing(
            self._detector, self._detector_session, self.bundle.detector
        )
        self._assert_detector_sizes_supported(self._detector_input_sizes)
        self._detector.prepare(
            self._device_id if self._provider == "CUDAExecutionProvider" else -1,
            input_size=self._detector_input_sizes,
            det_thresh=self._detector_threshold,
            nms_thresh=self._detector_nms_threshold,
        )
        self._recognizer = ArcFaceONNX(
            model_file=str(self.bundle.recognizer.path),
            session=self._recognizer_session,
        )
        self._configure_image_preprocessing(
            self._recognizer, self._recognizer_session, self.bundle.recognizer
        )
        if self._liveness_path is not None:
            from insightface.addons import Liveness

            self._liveness_session = self._new_session(self._liveness_path)
            self._liveness = Liveness(
                self._liveness_path, session=self._liveness_session,
                threshold=self._liveness_threshold,
            )

        self._warm_up()
        provider_audit: dict[str, object] = {}
        if self._provider == "CUDAExecutionProvider":
            provider_audit = {
                "detector": self._finish_cuda_profile(
                    self._detector_session, model_name="detector"
                ),
                "recognizer": self._finish_cuda_profile(
                    self._recognizer_session, model_name="recognizer"
                ),
            }
            if self._liveness_session is not None:
                provider_audit["liveness"] = self._finish_cuda_profile(
                    self._liveness_session, model_name="liveness"
                )

        cuda = _cuda_runtime_version()
        cudnn = _cudnn_version()
        gpus = _gpu_details()
        if self._provider == "CUDAExecutionProvider":
            if cuda is None or cudnn is None:
                raise RuntimeError(
                    "CUDA provider initialized but actual CUDA/cuDNN runtime versions "
                    f"could not be read (cuda={cuda}, cudnn={cudnn})"
                )
            validate_gpu_requirements(gpus, cuda)

        self._runtime = {
            "mode": "onnx",
            "os": platform.platform(),
            "architecture": platform.machine(),
            "cpu": cpu_summary(),
            "onnx_runtime_version": ort.__version__,
            "available_execution_providers": available,
            "execution_provider": self._provider,
            "detector_input_sizes": [list(size) for size in self._detector_input_sizes],
            "detector_session_providers": self._detector_session.get_providers(),
            "recognizer_session_providers": self._recognizer_session.get_providers(),
            "strict_provider_audit": provider_audit,
            "cuda_runtime_version": cuda,
            "cuda_image_version": os.getenv("CUDA_VERSION"),
            "cudnn_version": cudnn,
            "gpus": gpus,
            "models": [dict(model) for model in self.summary.models],
            "addons": [dict(addon) for addon in self.summary.addons],
            "liveness_mode": self._liveness_mode,
        }

    def _warm_up(self) -> None:
        self._warm_up_detector_sizes(self._detector_input_sizes)
        if self._liveness is not None:
            from insightface.addons.liveness import DESTINATION_LANDMARKS

            self._liveness.predict(
                np.zeros((80, 80, 3), dtype=np.uint8), DESTINATION_LANDMARKS
            )

        recognizer_input = self._recognizer_session.get_inputs()[0]
        width, height = self.bundle.recognizer.input_size
        recognition_data = np.zeros(
            (1, 3, height, width),
            dtype=getattr(self._recognizer, "input_dtype", np.float32),
        )
        recognition_outputs = self._recognizer_session.run(
            None, {recognizer_input.name: recognition_data}
        )
        if not recognition_outputs:
            raise RuntimeError("Recognition model warm-up returned no outputs")
        actual_dimension = np.asarray(recognition_outputs[0]).reshape(1, -1).shape[1]
        if actual_dimension != self.summary.embedding_dimension:
            raise RuntimeError(
                "Recognition warm-up output dimension does not match manifest: "
                f"expected {self.summary.embedding_dimension}, got {actual_dimension}"
            )

    def _assert_detector_sizes_supported(
        self, sizes: tuple[tuple[int, int], ...]
    ) -> None:
        static_size = getattr(self._detector, "static_input_size", None)
        if static_size is not None and sizes != (tuple(static_size),):
            raise ValueError(
                "The detection model has a static input and only supports "
                f"{tuple(static_size)}"
            )

    def _warm_up_detector_sizes(self, sizes: tuple[tuple[int, int], ...]) -> None:
        detector_input = self._detector_session.get_inputs()[0]
        for detector_width, detector_height in sizes:
            detector_data = np.zeros(
                (1, 3, detector_height, detector_width),
                dtype=getattr(self._detector, "input_dtype", np.float32),
            )
            detector_outputs = self._detector_session.run(
                None, {detector_input.name: detector_data}
            )
            if not detector_outputs:
                raise RuntimeError(
                    "Detection model warm-up returned no outputs for input size "
                    f"{detector_width}x{detector_height}"
                )

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
        from insightface.utils import face_align

        validate_image(image)
        if max_faces is not None and max_faces <= 0:
            raise ValueError("max_faces must be positive")
        if min_score is not None and not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")

        with self._concurrency.slot():
            with self._lifecycle_lock:
                if not self._started or self._closed:
                    raise RuntimeError("Inference engine has not been started")
                detector = self._detector
                recognizer = self._recognizer
                liveness = self._liveness if apply_liveness else None
                profile = detection_profile or self._default_detection_profile
                self.validate_detection_profile(profile)

            # SCRFD's policy and anchor cache live on its Python wrapper. Each
            # request gets independent mutable wrapper state while all copies
            # share the one process-wide, thread-safe ORT Session.
            request_detector = copy.copy(detector)
            request_detector.center_cache = dict(detector.center_cache)
            request_detector.det_thresh = profile.threshold
            request_detector.nms_thresh = profile.nms_threshold
            boxes, keypoints = request_detector.detect(
                image, input_size=profile.input_sizes, max_num=0
            )
            image_height, image_width = image.shape[:2]
            observations: list[FaceObservation] = []
            threshold = profile.threshold if min_score is None else max(profile.threshold, min_score)
            for index, row in enumerate(boxes):
                detection_score = float(np.clip(row[4], 0.0, 1.0))
                if detection_score < threshold:
                    continue
                left = float(np.clip(row[0], 0, image_width))
                top = float(np.clip(row[1], 0, image_height))
                right = float(np.clip(row[2], 0, image_width))
                bottom = float(np.clip(row[3], 0, image_height))
                landmarks = (
                    None if keypoints is None else np.asarray(keypoints[index], dtype=np.float32)
                )
                observation = FaceObservation(
                    bbox=(left, top, right, bottom),
                    detection_score=detection_score,
                    landmarks=landmarks,
                    embedding=None,
                )
                enrich_quality(image, observation)
                observations.append(observation)

            observations.sort(key=lambda face: face.area, reverse=True)
            if single_face and observations:
                def selection_score(face):
                    if profile.single_face_selection == "largest":
                        return face.area
                    left, top, right, bottom = face.bbox
                    return face.area - 2.0 * (
                        ((left + right - image_width) * 0.5) ** 2
                        + ((top + bottom - image_height) * 0.5) ** 2
                    )
                observations = [max(observations, key=selection_score)]
            retained = []
            for observation in observations:
                if liveness is not None:
                    try:
                        observation.liveness = liveness.predict(image, observation.landmarks)
                    except Exception as exc:
                        raise LivenessUnavailable("Liveness inference failed") from exc
                blocked = (
                    observation.liveness is not None
                    and self._liveness_mode == "normal"
                    and observation.liveness["is_live"] is not True
                )
                if require_embeddings and not blocked:
                    landmarks = observation.landmarks
                    if landmarks is None or landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
                        if liveness is None:
                            continue
                    else:
                        aligned = face_align.norm_crop(
                            image, landmark=landmarks,
                            image_size=self.bundle.recognizer.input_size[0],
                        )
                        observation.embedding = l2_normalize(recognizer.get_feat(aligned)[0])
                retained.append(observation)
                if max_faces is not None and len(retained) >= max_faces:
                    break
            return retained

    def validate_detection_profile(self, profile: DetectionProfile) -> None:
        sizes = normalize_detector_input_sizes(profile.input_sizes)
        if not 0.0 <= profile.threshold <= 1.0:
            raise ValueError("detection threshold must be between 0 and 1")
        if not 0.0 <= profile.nms_threshold <= 1.0:
            raise ValueError("detection nms_threshold must be between 0 and 1")
        normalize_single_face_selection(profile.single_face_selection)
        self._assert_detector_sizes_supported(sizes)

    def runtime_summary(self) -> dict[str, object]:
        with self._lifecycle_lock:
            summary = dict(self._runtime)
            summary["detector_input_sizes"] = [
                list(size) for size in self._detector_input_sizes
            ]
            summary["detection_profile"] = self._default_detection_profile.as_dict()
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
        with self._lifecycle_lock:
            self._detector = None
            self._recognizer = None
            self._detector_session = None
            self._recognizer_session = None
            self._liveness = None
            self._liveness_session = None
