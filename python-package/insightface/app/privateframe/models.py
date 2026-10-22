"""All neural inference used by the streaming profile through ONNX Runtime."""

from __future__ import annotations

import hashlib
import math
import platform
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from ...model_zoo.onnxruntime_utils import preload_cuda_libraries
from ...model_zoo.coreml_cache import create_coreml_session
from ...model_zoo.package_manifest import (
    ModelPackageDescriptor,
    load_model_package,
)
from ..face_analysis import FaceAnalysis
from .model_catalog import (
    DETECTION_TASK,
    RECOGNITION_TASK,
    VERIFICATION_TASK,
)

_COREML_PROVIDER = "CoreMLExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"


def _uses_coreml(runtime: Mapping[str, Any]) -> bool:
    providers = runtime.get("providers", [])
    return bool(providers) and str(providers[0]) == _COREML_PROVIDER


def _coreml_provider_options(*, static_shapes: bool) -> dict[str, str]:
    """Use the modern Core ML format and let the cache policy select units."""

    release = platform.mac_ver()[0]
    try:
        major = int(release.split(".", 1)[0])
    except (TypeError, ValueError):
        major = 0
    return {
        # MLProgram requires Core ML 5 (macOS 12). Older supported macOS
        # releases retain the NeuralNetwork format instead of failing startup.
        "ModelFormat": "MLProgram" if major >= 12 else "NeuralNetwork",
        # Managed manifest and fixed-resolution Session creation first tries
        # ALL and explicitly rebuilds with CPUAndGPU only when Core ML cannot
        # create or execute that exact Session signature.
        "MLComputeUnits": "ALL",
        "RequireStaticInputShapes": "1" if static_shapes else "0",
        "EnableOnSubgraphs": "0",
    }


def _provider_options(
    runtime: Mapping[str, Any],
    *,
    static_shapes: bool,
) -> list[dict[str, str]]:
    providers = [str(provider) for provider in runtime["providers"]]
    return [
        _coreml_provider_options(static_shapes=static_shapes)
        if provider == _COREML_PROVIDER
        else {}
        for provider in providers
    ]


def active_face_detector(
    config: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    """Return the detection task from the selected raccoon package."""

    models = config["models"]
    model = models.get(DETECTION_TASK)
    if not isinstance(model, Mapping):
        raise TypeError("models.detection configuration is required")
    return DETECTION_TASK, model


def active_face_verifier(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the verification task from the selected raccoon package."""

    models = config["models"]
    model = models.get(VERIFICATION_TASK)
    if not isinstance(model, Mapping):
        raise TypeError("models.verification configuration is required")
    return model


def packaged_face_recognizer(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the packaged recognizer without creating an inference session."""

    model = config["models"].get(RECOGNITION_TASK)
    if not isinstance(model, Mapping):
        raise TypeError("models.recognition must be a mapping")
    return model


def _load_selected_model_package(
    models: Mapping[str, Any],
) -> ModelPackageDescriptor:
    """Load the selected V2 package without pinning a manifest byte snapshot."""

    manifest_path = Path(str(models["manifest_path"])).resolve()
    package = load_model_package(manifest_path.parent)
    expected_name = str(models["name"])
    if package.name != expected_name:
        raise RuntimeError(
            f"model package name mismatch for {manifest_path}: "
            f"{package.name} != {expected_name}"
        )
    if manifest_path != package.manifest_path:
        raise RuntimeError(
            "effective manifest path does not match the selected model package: "
            f"{manifest_path} != {package.manifest_path}"
        )
    return package


def _session_options(
    runtime: Mapping[str, Any],
    *,
    detector_input_size: tuple[int, int] | None = None,
) -> ort.SessionOptions:
    options = ort.SessionOptions()
    intra_threads = int(runtime.get("intra_op_threads", 0))
    inter_threads = int(runtime.get("inter_op_threads", 0))
    if intra_threads > 0:
        options.intra_op_num_threads = intra_threads
    if inter_threads > 0:
        options.inter_op_num_threads = inter_threads
    if _uses_coreml(runtime) and detector_input_size is not None:
        width, height = detector_input_size
        options.add_free_dimension_override_by_name("height", int(height))
        options.add_free_dimension_override_by_name("width", int(width))
    options.log_severity_level = 3
    return options


class _CoreMLResolutionSessionFactory:
    """Create a fixed-shape CoreML Session for one SCRFD resolution."""

    def __init__(
        self,
        runtime: Mapping[str, Any],
        model_sha256: str | None = None,
        cache_root: str | Path | None = None,
    ) -> None:
        self._runtime = dict(runtime)
        self._model_sha256 = model_sha256
        self._cache_root = cache_root

    def __call__(
        self,
        model_file: str | Path,
        input_size: tuple[int, int],
        reference_session: Any,
    ) -> Any:
        # SCRFD owns the resolution cache and passes the primary Session as
        # context. CoreML needs a new SessionOptions object for every shape:
        # free-dimension overrides are mutable and cannot safely be reused.
        width, height = (int(input_size[0]), int(input_size[1]))
        if width <= 0 or height <= 0:
            raise ValueError("SCRFD resolution width and height must be positive")
        get_providers = getattr(reference_session, "get_providers", None)
        providers = (
            [str(provider) for provider in get_providers()]
            if callable(get_providers)
            else [str(provider) for provider in self._runtime["providers"]]
        )
        if not providers:
            raise RuntimeError("SCRFD reference Session has no active provider")
        use_coreml = providers[0] == _COREML_PROVIDER
        active_runtime = dict(self._runtime)
        active_runtime["providers"] = providers
        provider_options = _provider_options(
            active_runtime,
            static_shapes=use_coreml,
        )
        if use_coreml:
            inputs_getter = getattr(reference_session, "get_inputs", None)
            inputs = inputs_getter() if callable(inputs_getter) else ()
            input_meta = inputs[0] if inputs else None
            input_name = str(getattr(input_meta, "name", "input.1"))
            input_dtype = str(getattr(input_meta, "type", "tensor(float)"))
            shape = list(getattr(input_meta, "shape", [1, 3, height, width]))
            if len(shape) != 4:
                shape = [1, 3, height, width]
            else:
                shape[0] = 1
                shape[-2] = height
                shape[-1] = width
            model_path = Path(model_file).expanduser().resolve()
            model_sha256 = self._model_sha256
            if model_sha256 is None:
                if model_path.is_file():
                    digest = hashlib.sha256()
                    with model_path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    model_sha256 = digest.hexdigest()
                else:
                    # Only injected/fake factories lack a readable model.
                    model_sha256 = hashlib.sha256(
                        str(model_path).encode("utf-8")
                    ).hexdigest()
            result = create_coreml_session(
                ort.InferenceSession,
                str(model_file),
                providers=providers,
                provider_options=provider_options,
                model_sha256=model_sha256,
                task=DETECTION_TASK,
                graph_variant="scrfd_dimension_override_v1",
                input_contracts=[
                    {
                        "name": input_name,
                        "dtype": input_dtype,
                        "shape": shape,
                    }
                ],
                sess_options_factory=lambda: _session_options(
                    self._runtime,
                    detector_input_size=(width, height),
                ),
                warmup=True,
                cache_root=self._cache_root,
            )
            session = result.session
        else:
            session = ort.InferenceSession(
                str(model_file),
                sess_options=_session_options(
                    self._runtime,
                    detector_input_size=(width, height),
                ),
                providers=providers,
                provider_options=provider_options,
            )
        actual = list(session.get_providers())
        if use_coreml and (not actual or actual[0] != _COREML_PROVIDER):
            raise RuntimeError(
                "CoreMLExecutionProvider was requested for SCRFD resolution "
                f"{width}x{height}, but ONNX Runtime activated {actual}"
            )
        return session


def _validate_primary_provider(model: Any, runtime: Mapping[str, Any]) -> None:
    requested = [str(provider) for provider in runtime["providers"]]
    if not requested or requested[0] == _CPU_PROVIDER:
        return
    primary = requested[0]
    session = getattr(model, "session", None)
    get_providers = getattr(session, "get_providers", None)
    if not callable(get_providers):
        raise TypeError("manifest model must expose its ONNX Runtime session")
    actual = list(get_providers())
    if not actual or actual[0] != primary:
        raise RuntimeError(
            f"{primary} was requested as the primary provider, "
            f"but ONNX Runtime activated {actual}"
        )


def _configured_detector_input_sizes(
    config: Mapping[str, Any],
    section: str,
) -> list[tuple[int, int]]:
    settings = config.get(section, {})
    if not isinstance(settings, Mapping):
        return []
    raw_passes = settings.get("passes")
    raw_sizes = (
        [item.get("input_size") for item in raw_passes if isinstance(item, Mapping)]
        if isinstance(raw_passes, Sequence)
        and not isinstance(raw_passes, (str, bytes))
        else [settings.get("input_size")]
    )
    sizes: list[tuple[int, int]] = []
    for raw in raw_sizes:
        if isinstance(raw, bool) or raw is None:
            continue
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if len(raw) != 2:
                continue
            size = (int(raw[0]), int(raw[1]))
        else:
            side = int(raw)
            size = (side, side)
        if size[0] > 0 and size[1] > 0 and size not in sizes:
            sizes.append(size)
    return sizes


def _make_face_analysis(
    config: Mapping[str, Any],
    allowed_modules: Sequence[str],
    *,
    detector_input_size: tuple[int, int] | None = None,
) -> FaceAnalysis:
    """Build one manifest-backed model host with the requested task set."""

    models = config["models"]
    runtime = config["runtime"]
    package = _load_selected_model_package(models)
    missing = [task for task in allowed_modules if task not in package.tasks]
    if missing:
        raise ValueError(
            f"PrivateFrame model package {package.name!r} is missing required "
            f"task(s): {', '.join(missing)}"
        )
    detection_settings = active_face_detector(config)[1]
    nms_threshold = float(detection_settings["nms_iou_threshold"])
    if not math.isfinite(nms_threshold) or not 0.0 <= nms_threshold <= 1.0:
        raise ValueError(
            "models.detection.nms_iou_threshold must be between 0 and 1"
        )
    detector_max_detections(config)
    preload_cuda_libraries(runtime["providers"])
    static_shape_sessions = runtime.get(
        "scrfd_static_shape_sessions",
        True,
    )
    if not isinstance(static_shape_sessions, bool):
        raise TypeError(
            "runtime.scrfd_static_shape_sessions must be boolean"
        )

    constructor_kwargs: dict[str, Any] = {
        "name": package,
        "allowed_modules": tuple(allowed_modules),
        "providers": [str(provider) for provider in runtime["providers"]],
        "sess_options": _session_options(
            runtime,
            detector_input_size=(
                detector_input_size if static_shape_sessions else None
            ),
        ),
        "static_shape_sessions": static_shape_sessions,
    }
    if _uses_coreml(runtime):
        constructor_kwargs["_coreml_detector_input_size"] = (
            detector_input_size
        )
        if static_shape_sessions and detector_input_size is None:
            raise ValueError(
                "CoreML FaceAnalysis requires a fixed detector input size"
            )
        constructor_kwargs["provider_options"] = _provider_options(
            runtime,
            # The same FaceAnalysis options also create the dynamic-batch
            # verifier and recognizer. The detector dimensions are fixed above,
            # while these two task contracts must retain arbitrary batch size.
            static_shapes=False,
        )
        if static_shape_sessions:
            constructor_kwargs["resolution_session_factory"] = (
                _CoreMLResolutionSessionFactory(runtime)
            )
    analysis = FaceAnalysis(
        **constructor_kwargs,
    )
    detector = analysis.det_model
    for taskname in allowed_modules:
        model = analysis.models.get(taskname)
        if model is None:
            raise RuntimeError(f"FaceAnalysis did not load requested module: {taskname}")
        _validate_primary_provider(model, runtime)
    detector.nms_thresh = nms_threshold
    return analysis


def make_face_analysis(
    config: Mapping[str, Any],
) -> FaceAnalysis:
    """Build the standard manifest-backed model host for a PrivateFrame run.

    ``allowed_modules`` is the sole model-selection interface: detector and
    verifier are always required, while recognition is included only for a
    selective identity policy.
    """

    selective = str(config.get("recognition", {}).get("mode", "all")) != "all"
    allowed_modules = [DETECTION_TASK, VERIFICATION_TASK]
    if selective:
        allowed_modules.append(RECOGNITION_TASK)
    scan_sizes = _configured_detector_input_sizes(config, "scan")
    primary_size = max(
        scan_sizes or [(640, 640)],
        key=lambda value: value[0] * value[1],
    )
    return _make_face_analysis(
        config,
        allowed_modules,
        detector_input_size=primary_size,
    )


def make_review_face_analysis(
    config: Mapping[str, Any],
) -> FaceAnalysis:
    """Build the independent detection-only host used by local review."""

    review_sizes = _configured_detector_input_sizes(config, "revalidation")
    review_size = max(
        review_sizes or [(160, 160)],
        key=lambda value: value[0] * value[1],
    )
    return _make_face_analysis(
        config,
        (DETECTION_TASK,),
        detector_input_size=review_size,
    )


def detector_max_detections(config: Mapping[str, Any]) -> int:
    raw = active_face_detector(config)[1]["max_detections"]
    if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
        raise TypeError("models.detection.max_detections must be an integer")
    maximum = int(raw)
    if maximum < 1:
        raise ValueError("models.detection.max_detections must be positive")
    return maximum


def detect_faces(
    analysis: FaceAnalysis,
    image: np.ndarray,
    *,
    input_sizes: Sequence[int | Sequence[int]],
    confidence_threshold: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    """Run SCRFD and apply PrivateFrame's deterministic output cap."""

    maximum = int(max_detections)
    if maximum < 1:
        raise ValueError("max_detections must be positive")
    normalized_sizes = [
        (int(value), int(value))
        if isinstance(value, (int, np.integer))
        else (int(value[0]), int(value[1]))
        for value in input_sizes
    ]
    result = analysis.det_model.detect(
        image,
        input_size=normalized_sizes,
        det_thresh=float(confidence_threshold),
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("SCRFD detector must return a (bboxes, kpss) tuple")
    bboxes, landmarks = result
    bboxes = np.asarray(bboxes)
    detections = []
    for index in range(len(bboxes)):
        row = np.asarray(bboxes[index], dtype=np.float64).reshape(-1)
        if row.size < 4:
            raise ValueError(
                "SCRFD detector bbox rows must contain at least four values"
            )
        detection = {
            "box": row[:4].tolist(),
            "confidence": float(row[4]) if row.size >= 5 else 1.0,
        }
        if landmarks is not None:
            detection["landmarks"] = np.asarray(landmarks[index]).tolist()
        detections.append(detection)
    indexed = list(enumerate(detections))
    for _index, detection in indexed:
        confidence = float(detection["confidence"])
        if not math.isfinite(confidence):
            raise ValueError("SCRFD confidence must be finite")
    indexed.sort(
        key=lambda value: (-float(value[1]["confidence"]), value[0])
    )
    return [detection for _index, detection in indexed[:maximum]]


def padded_square_crop(
    frame: np.ndarray,
    value: Sequence[float],
    expansion: float,
    *,
    minimum_side: int = 2,
) -> tuple[np.ndarray, int, int]:
    """Crop the requested square without shrinking it at image boundaries.

    The returned origin is expressed in source-frame coordinates and may be
    negative. Pixels outside the source frame are represented by zero-valued
    pixels in the output canvas.
    """

    if expansion <= 0.0:
        raise ValueError("crop expansion must be positive")
    if minimum_side <= 0:
        raise ValueError("minimum crop side must be positive")
    target = np.asarray(value, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(target)):
        raise ValueError("crop box must be finite")
    c = (target[:2] + target[2:]) * 0.5
    side = max(
        minimum_side,
        math.ceil(max(target[2] - target[0], target[3] - target[1]) * expansion),
    )
    x1, y1 = math.floor(c[0] - side * 0.5), math.floor(c[1] - side * 0.5)
    canvas = np.zeros((side, side, 3), dtype=frame.dtype)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(frame.shape[1], x1 + side), min(frame.shape[0], y1 + side)
    if sx2 > sx1 and sy2 > sy1:
        canvas[sy1 - y1 : sy2 - y1, sx1 - x1 : sx2 - x1] = frame[sy1:sy2, sx1:sx2]
    return canvas, x1, y1


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == -90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError(f"unsupported cardinal angle {angle}")


__all__ = [
    "active_face_detector",
    "active_face_verifier",
    "detect_faces",
    "detector_max_detections",
    "make_face_analysis",
    "make_review_face_analysis",
    "packaged_face_recognizer",
    "padded_square_crop",
    "rotate_image",
]
