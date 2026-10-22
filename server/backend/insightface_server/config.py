from __future__ import annotations

import math
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

SingleFaceSelection = Literal["largest", "center_largest"]

SUPPORTED_SEARCH_PROFILES = (
    "fp32_v1",
    "fp16_v1",
    "bf16_v1",
    "int8_x736_v1",
    "int8_x1000_v1",
)
SUPPORTED_SEARCH_LOAD_POLICIES = ("eager", "lazy")
SUPPORTED_SEARCH_BACKENDS = ("auto", "reference", "native_cpu", "native_cuda")
SUPPORTED_SEARCH_TOPK_MODES = ("auto", "host", "device")
DEFAULT_DETECTOR_INPUT_SIZES = ((96, 96), (512, 512))
DEFAULT_DETECTOR_THRESHOLD = 0.50
DEFAULT_DETECTOR_NMS_THRESHOLD = 0.40
DEFAULT_SINGLE_FACE_SELECTION: SingleFaceSelection = "largest"
DEFAULT_MAX_DETECTED_FACES = 100
DEFAULT_CPU_INFERENCE_MAX_CONCURRENCY = 4
DEFAULT_CUDA_INFERENCE_MAX_CONCURRENCY = 8
MAX_INFERENCE_MAX_CONCURRENCY = 256
SUPPORTED_SINGLE_FACE_SELECTIONS = ("largest", "center_largest")
SUPPORTED_ADDONS = ("liveness",)
MAX_DETECTOR_INPUT_SIZE_COUNT = 4
MIN_DETECTOR_INPUT_SIDE = 32
MAX_DETECTOR_INPUT_SIDE = 2048
MAX_DETECTOR_INPUT_PIXELS = 4 * 1024 * 1024


def _finite_unit_interval(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and within 0.0..1.0")
    return normalized


def normalize_addons(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{name} must be a list of addon names")
    if any(not isinstance(item, str) or item not in SUPPORTED_ADDONS for item in value):
        raise ValueError(f"{name} supports only: {', '.join(SUPPORTED_ADDONS)}")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicate names")
    return tuple(value)


def normalize_single_face_selection(value: object) -> SingleFaceSelection:
    if not isinstance(value, str):
        raise ValueError("single_face_selection must be a string")
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_SINGLE_FACE_SELECTIONS:
        allowed = ", ".join(SUPPORTED_SINGLE_FACE_SELECTIONS)
        raise ValueError(f"single_face_selection must be one of: {allowed}")
    return cast(SingleFaceSelection, normalized)


def normalize_detector_input_sizes(value: object) -> tuple[tuple[int, int], ...]:
    """Validate and canonicalize the startup-only detector-size setting.

    SCRFD's largest feature-map stride is 32. Requiring aligned dimensions
    prevents ambiguous partial feature-map behavior, while the count, side and
    total-pixel limits bound a configuration mistake before it
    can exhaust CPU RAM or GPU VRAM.
    """

    if not isinstance(value, list | tuple) or not value:
        raise ValueError("detector_input_sizes must be a non-empty list of [width, height]")
    if len(value) > MAX_DETECTOR_INPUT_SIZE_COUNT:
        raise ValueError(
            f"detector_input_sizes supports at most {MAX_DETECTOR_INPUT_SIZE_COUNT} sizes"
        )
    normalized: list[tuple[int, int]] = []
    total_pixels = 0
    for item in value:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise ValueError("each detector input size must be [width, height]")
        width, height = item
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
        ):
            raise ValueError("detector input width and height must be integers")
        if not (
            MIN_DETECTOR_INPUT_SIDE <= width <= MAX_DETECTOR_INPUT_SIDE
            and MIN_DETECTOR_INPUT_SIDE <= height <= MAX_DETECTOR_INPUT_SIDE
        ):
            raise ValueError(
                "detector input width and height must be within "
                f"{MIN_DETECTOR_INPUT_SIDE}..{MAX_DETECTOR_INPUT_SIDE}"
            )
        if width % 32 or height % 32:
            raise ValueError("detector input width and height must be multiples of 32")
        size = (width, height)
        if size in normalized:
            raise ValueError("detector input sizes must not contain duplicates")
        normalized.append(size)
        total_pixels += width * height
    if total_pixels > MAX_DETECTOR_INPUT_PIXELS:
        raise ValueError(
            "combined detector input pixels exceed the supported system limit "
            f"of {MAX_DETECTOR_INPUT_PIXELS}"
        )
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class DetectionProfile:
    """One immutable detector policy used for the whole request."""

    input_sizes: tuple[tuple[int, int], ...] = DEFAULT_DETECTOR_INPUT_SIZES
    threshold: float = DEFAULT_DETECTOR_THRESHOLD
    nms_threshold: float = DEFAULT_DETECTOR_NMS_THRESHOLD
    single_face_selection: SingleFaceSelection = DEFAULT_SINGLE_FACE_SELECTION

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        defaults: DetectionProfile | None = None,
    ) -> DetectionProfile:
        base = defaults or cls()
        document = value or {}
        unknown = set(document) - {
            "input_sizes",
            "threshold",
            "nms_threshold",
            "single_face_selection",
        }
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unsupported detection setting(s): {names}")
        return cls(
            input_sizes=normalize_detector_input_sizes(
                document.get("input_sizes", base.input_sizes)
            ),
            threshold=_finite_unit_interval(
                document.get("threshold", base.threshold), name="detection threshold"
            ),
            nms_threshold=_finite_unit_interval(
                document.get("nms_threshold", base.nms_threshold),
                name="detection nms_threshold",
            ),
            single_face_selection=normalize_single_face_selection(
                document.get("single_face_selection", base.single_face_selection)
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "input_sizes": [list(size) for size in self.input_sizes],
            "threshold": self.threshold,
            "nms_threshold": self.nms_threshold,
            "single_face_selection": self.single_face_selection,
        }


@dataclass(frozen=True, slots=True)
class ServerFileConfig:
    detection: DetectionProfile = DetectionProfile()
    max_detected_faces: int = DEFAULT_MAX_DETECTED_FACES
    inference_max_concurrency: int | None = None
    web_ui_disabled: bool = False
    addons: tuple[str, ...] = ()
    auto_download_addons: tuple[str, ...] = ()
    liveness_mode: str = "normal"
    liveness_threshold: float = 0.8
    liveness_compare_scope: str = "both"
    liveness_on_registration: bool = False


def default_inference_max_concurrency(execution_provider: str) -> int:
    return (
        DEFAULT_CUDA_INFERENCE_MAX_CONCURRENCY
        if execution_provider == "CUDAExecutionProvider"
        else DEFAULT_CPU_INFERENCE_MAX_CONCURRENCY
    )


def normalize_inference_max_concurrency(
    value: object, *, name: str = "[inference].max_concurrency"
) -> int | None:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be 'auto' or an integer")
    if not 1 <= value <= MAX_INFERENCE_MAX_CONCURRENCY:
        raise ValueError(
            f"{name} must be within 1..{MAX_INFERENCE_MAX_CONCURRENCY}"
        )
    return value


def load_server_config(
    path: Path | None,
) -> ServerFileConfig:
    """Read and validate the immutable startup configuration from TOML."""

    if path is None:
        return ServerFileConfig()
    try:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
    except OSError as exc:
        raise ValueError(f"Unable to read INSIGHTFACE_CONFIG_FILE '{path}': {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"INSIGHTFACE_CONFIG_FILE '{path}' is not valid TOML: {exc}") from exc

    unknown_sections = set(document) - {"detection", "inference", "web", "addons"}
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ValueError(f"Unsupported server configuration section(s): {names}")
    detection = document.get("detection", {})
    if not isinstance(detection, dict):
        raise ValueError("[detection] must be a TOML table")
    unknown_detection = set(detection) - {
        "input_sizes",
        "threshold",
        "nms_threshold",
        "single_face_selection",
        "max_detected_faces",
    }
    if unknown_detection:
        names = ", ".join(sorted(unknown_detection))
        raise ValueError(f"Unsupported [detection] setting(s): {names}")
    web = document.get("web", {})
    if not isinstance(web, dict):
        raise ValueError("[web] must be a TOML table")
    unknown_web = set(web) - {"disabled"}
    if unknown_web:
        names = ", ".join(sorted(unknown_web))
        raise ValueError(f"Unsupported [web] setting(s): {names}")
    web_ui_disabled = web.get("disabled", False)
    if not isinstance(web_ui_disabled, bool):
        raise ValueError("[web].disabled must be a boolean")
    inference = document.get("inference", {})
    if not isinstance(inference, dict):
        raise ValueError("[inference] must be a TOML table")
    unknown_inference = set(inference) - {
        "max_concurrency", "addons", "liveness_mode", "liveness_threshold",
        "liveness_compare_scope", "liveness_on_registration",
    }
    if unknown_inference:
        names = ", ".join(sorted(unknown_inference))
        raise ValueError(f"Unsupported [inference] setting(s): {names}")
    inference_max_concurrency = normalize_inference_max_concurrency(
        inference.get("max_concurrency", "auto")
    )
    addon_config = document.get("addons", {})
    if not isinstance(addon_config, dict) or set(addon_config) - {"auto_download"}:
        raise ValueError("[addons] supports only the auto_download list")
    addons = normalize_addons(inference.get("addons", []), name="[inference].addons")
    auto_download_addons = normalize_addons(
        addon_config.get("auto_download", []), name="[addons].auto_download"
    )
    liveness_mode = inference.get("liveness_mode", "normal")
    if liveness_mode not in ("normal", "observe"):
        raise ValueError("[inference].liveness_mode must be normal or observe")
    compare_scope = inference.get("liveness_compare_scope", "both")
    if compare_scope not in ("both", "source", "target"):
        raise ValueError("[inference].liveness_compare_scope must be both, source or target")
    liveness_on_registration = inference.get("liveness_on_registration", False)
    if not isinstance(liveness_on_registration, bool):
        raise ValueError("[inference].liveness_on_registration must be a boolean")
    max_detected_faces = detection.get(
        "max_detected_faces", DEFAULT_MAX_DETECTED_FACES
    )
    if (
        isinstance(max_detected_faces, bool)
        or not isinstance(max_detected_faces, int)
        or not 1 <= max_detected_faces <= 100
    ):
        raise ValueError("[detection].max_detected_faces must be an integer within 1..100")
    profile_values = {
        key: value
        for key, value in detection.items()
        if key != "max_detected_faces"
    }
    return ServerFileConfig(
        detection=DetectionProfile.from_mapping(profile_values),
        max_detected_faces=max_detected_faces,
        inference_max_concurrency=inference_max_concurrency,
        web_ui_disabled=web_ui_disabled,
        addons=addons,
        auto_download_addons=auto_download_addons,
        liveness_mode=liveness_mode,
        liveness_threshold=_finite_unit_interval(
            inference.get("liveness_threshold", 0.8), name="[inference].liveness_threshold"
        ),
        liveness_compare_scope=compare_scope,
        liveness_on_registration=liveness_on_registration,
    )


def load_detector_input_sizes(path: Path | None) -> tuple[tuple[int, int], ...]:
    """Backward-compatible helper for the detector portion of server TOML."""

    return load_server_config(path).detection.input_sizes


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be an explicit boolean")


def _int(name: str, default: int, *, minimum: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite and within {minimum}..{maximum}")
    return value


def _choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        allowed = ", ".join(choices)
        raise ValueError(f"{name} must be one of: {allowed}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    models_dir: Path
    inference_mode: str
    execution_provider: str
    auth_enabled: bool
    startup_api_key: str | None
    max_image_bytes: int
    max_image_pixels: int
    max_request_bytes: int
    max_registration_images: int
    request_timeout_seconds: int
    detector_threshold: float
    registration_min_score: float
    registration_min_quality: float
    registration_min_face_size: int
    default_threshold: float
    cors_origins: tuple[str, ...]
    log_level: str
    save_face_crops: bool
    device_id: int
    inference_max_concurrency: int | None = None
    detector_input_sizes: tuple[tuple[int, int], ...] = DEFAULT_DETECTOR_INPUT_SIZES
    detector_nms_threshold: float = DEFAULT_DETECTOR_NMS_THRESHOLD
    detector_single_face_selection: SingleFaceSelection = DEFAULT_SINGLE_FACE_SELECTION
    max_detected_faces: int = DEFAULT_MAX_DETECTED_FACES
    config_file: Path | None = None
    default_search_profile: str = "fp32_v1"
    default_search_capacity_rows: int = 100_000
    max_search_capacity_rows: int = 10_000_000
    default_max_faces_per_person: int = 20
    default_search_load_policy: str = "lazy"
    search_backend: str = "auto"
    search_library_path: Path | None = None
    search_device_id: int = 0
    search_topk_mode: str = "auto"
    search_build_batch_rows: int = 4096
    web_ui_disabled: bool = False
    rtsp_max_streams: int = 4
    rtsp_preview_fps: float = 5.0
    rtsp_jpeg_quality: int = 82
    rtsp_open_timeout_seconds: float = 5.0
    rtsp_read_timeout_seconds: float = 5.0
    rtsp_reconnect_delay_seconds: float = 1.0
    addons: tuple[str, ...] = ()
    auto_download_addons: tuple[str, ...] = ()
    liveness_mode: str = "normal"
    liveness_threshold: float = 0.8
    liveness_compare_scope: str = "both"
    liveness_on_registration: bool = False

    @property
    def detection_profile(self) -> DetectionProfile:
        return DetectionProfile(
            input_sizes=self.detector_input_sizes,
            threshold=self.detector_threshold,
            nms_threshold=self.detector_nms_threshold,
            single_face_selection=self.detector_single_face_selection,
        )

    @property
    def effective_inference_max_concurrency(self) -> int:
        return self.inference_max_concurrency or default_inference_max_concurrency(
            self.execution_provider
        )

    @classmethod
    def from_env(cls) -> Settings:
        mode = os.getenv("INSIGHTFACE_INFERENCE_MODE", "onnx").strip().lower()
        if mode not in {"onnx", "mock"}:
            raise ValueError("INSIGHTFACE_INFERENCE_MODE must be onnx or mock")
        provider = os.getenv(
            "INSIGHTFACE_EXECUTION_PROVIDER", "CPUExecutionProvider"
        ).strip()
        if provider not in {"CPUExecutionProvider", "CUDAExecutionProvider"}:
            raise ValueError("Unsupported INSIGHTFACE_EXECUTION_PROVIDER")
        origins = tuple(
            part.strip()
            for part in os.getenv("INSIGHTFACE_CORS_ORIGINS", "").split(",")
            if part.strip()
        )
        device_id = _int("INSIGHTFACE_DEVICE_ID", 0, minimum=0)
        search_library = os.getenv("INSIGHTFACE_SEARCH_LIBRARY")
        config_file_value = os.getenv("INSIGHTFACE_CONFIG_FILE")
        config_file = (
            Path(config_file_value).expanduser() if config_file_value else None
        )
        file_config = load_server_config(config_file)
        inference_concurrency_value = os.getenv(
            "INSIGHTFACE_INFERENCE_MAX_CONCURRENCY"
        )
        if inference_concurrency_value is None:
            inference_max_concurrency = file_config.inference_max_concurrency
        else:
            raw_value = inference_concurrency_value.strip().lower()
            try:
                raw_concurrency: object = (
                    int(inference_concurrency_value) if raw_value != "auto" else "auto"
                )
            except ValueError as exc:
                raise ValueError(
                    "INSIGHTFACE_INFERENCE_MAX_CONCURRENCY must be 'auto' or an integer"
                ) from exc
            inference_max_concurrency = normalize_inference_max_concurrency(
                raw_concurrency, name="INSIGHTFACE_INFERENCE_MAX_CONCURRENCY"
            )
        default_capacity_rows = _int(
            "INSIGHTFACE_COLLECTION_DEFAULT_CAPACITY_ROWS", 100_000
        )
        max_capacity_rows = _int(
            "INSIGHTFACE_COLLECTION_MAX_CAPACITY_ROWS", 10_000_000
        )
        if default_capacity_rows > max_capacity_rows:
            raise ValueError(
                "INSIGHTFACE_COLLECTION_DEFAULT_CAPACITY_ROWS must not exceed "
                "INSIGHTFACE_COLLECTION_MAX_CAPACITY_ROWS"
            )
        rtsp_jpeg_quality = _int("INSIGHTFACE_RTSP_JPEG_QUALITY", 82)
        if rtsp_jpeg_quality > 100:
            raise ValueError("INSIGHTFACE_RTSP_JPEG_QUALITY must be within 1..100")
        return cls(
            data_dir=Path(os.getenv("INSIGHTFACE_DATA_DIR", "/data")).expanduser(),
            models_dir=Path(os.getenv("INSIGHTFACE_MODELS_DIR", "/models")).expanduser(),
            inference_mode=mode,
            execution_provider=provider,
            auth_enabled=_bool("INSIGHTFACE_AUTH_ENABLED", True),
            startup_api_key=os.getenv("INSIGHTFACE_API_KEY") or None,
            max_image_bytes=_int("INSIGHTFACE_MAX_IMAGE_BYTES", 10 * 1024 * 1024),
            max_image_pixels=_int("INSIGHTFACE_MAX_IMAGE_PIXELS", 40_000_000),
            max_request_bytes=_int("INSIGHTFACE_MAX_REQUEST_BYTES", 64 * 1024 * 1024),
            max_registration_images=_int("INSIGHTFACE_MAX_REGISTRATION_IMAGES", 20),
            request_timeout_seconds=_int("INSIGHTFACE_REQUEST_TIMEOUT_SECONDS", 60),
            detector_threshold=_float(
                "INSIGHTFACE_DETECTOR_THRESHOLD",
                file_config.detection.threshold,
                minimum=0.0,
                maximum=1.0,
            ),
            registration_min_score=_float(
                "INSIGHTFACE_REGISTRATION_MIN_SCORE", 0.6, minimum=0.0, maximum=1.0
            ),
            registration_min_quality=_float(
                "INSIGHTFACE_REGISTRATION_MIN_QUALITY", 0.35, minimum=0.0, maximum=1.0
            ),
            registration_min_face_size=_int(
                "INSIGHTFACE_REGISTRATION_MIN_FACE_SIZE", 40
            ),
            default_threshold=_float(
                "INSIGHTFACE_DEFAULT_THRESHOLD", 0.4, minimum=0.0, maximum=1.0
            ),
            cors_origins=origins,
            log_level=os.getenv("INSIGHTFACE_LOG_LEVEL", "INFO").upper(),
            save_face_crops=_bool("INSIGHTFACE_SAVE_FACE_CROPS", False),
            device_id=device_id,
            inference_max_concurrency=(
                inference_max_concurrency
                or default_inference_max_concurrency(provider)
            ),
            detector_input_sizes=file_config.detection.input_sizes,
            detector_nms_threshold=file_config.detection.nms_threshold,
            detector_single_face_selection=file_config.detection.single_face_selection,
            max_detected_faces=file_config.max_detected_faces,
            config_file=config_file,
            addons=file_config.addons,
            auto_download_addons=file_config.auto_download_addons,
            liveness_mode=file_config.liveness_mode,
            liveness_threshold=file_config.liveness_threshold,
            liveness_compare_scope=file_config.liveness_compare_scope,
            liveness_on_registration=file_config.liveness_on_registration,
            default_search_profile=_choice(
                "INSIGHTFACE_COLLECTION_DEFAULT_SEARCH_PROFILE",
                "fp32_v1",
                SUPPORTED_SEARCH_PROFILES,
            ),
            default_search_capacity_rows=default_capacity_rows,
            max_search_capacity_rows=max_capacity_rows,
            default_max_faces_per_person=_int(
                "INSIGHTFACE_COLLECTION_DEFAULT_MAX_FACES_PER_PERSON", 20
            ),
            default_search_load_policy=_choice(
                "INSIGHTFACE_COLLECTION_DEFAULT_LOAD_POLICY",
                "lazy",
                SUPPORTED_SEARCH_LOAD_POLICIES,
            ),
            search_backend=_choice(
                "INSIGHTFACE_SEARCH_BACKEND", "auto", SUPPORTED_SEARCH_BACKENDS
            ),
            search_library_path=(
                Path(search_library).expanduser() if search_library else None
            ),
            search_device_id=_int(
                "INSIGHTFACE_SEARCH_DEVICE_ID", device_id, minimum=0
            ),
            search_topk_mode=_choice(
                "INSIGHTFACE_SEARCH_TOPK_MODE", "auto", SUPPORTED_SEARCH_TOPK_MODES
            ),
            search_build_batch_rows=_int(
                "INSIGHTFACE_SEARCH_BUILD_BATCH_ROWS", 4096
            ),
            web_ui_disabled=file_config.web_ui_disabled,
            rtsp_max_streams=_int("INSIGHTFACE_RTSP_MAX_STREAMS", 4),
            rtsp_preview_fps=_float(
                "INSIGHTFACE_RTSP_PREVIEW_FPS", 5.0, minimum=1.0, maximum=30.0
            ),
            rtsp_jpeg_quality=rtsp_jpeg_quality,
            rtsp_open_timeout_seconds=_float(
                "INSIGHTFACE_RTSP_OPEN_TIMEOUT_SECONDS",
                5.0,
                minimum=0.5,
                maximum=60.0,
            ),
            rtsp_read_timeout_seconds=_float(
                "INSIGHTFACE_RTSP_READ_TIMEOUT_SECONDS",
                5.0,
                minimum=0.5,
                maximum=60.0,
            ),
            rtsp_reconnect_delay_seconds=_float(
                "INSIGHTFACE_RTSP_RECONNECT_DELAY_SECONDS",
                1.0,
                minimum=0.1,
                maximum=30.0,
            ),
        )
