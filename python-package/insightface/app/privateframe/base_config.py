"""Configuration loading and shared validation for the streaming pipeline."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import onnxruntime as ort
import yaml

from ...model_zoo.onnxruntime_utils import get_default_providers
from .model_catalog import (
    DETECTION_TASK,
    RECOGNITION_TASK,
    SUPPORTED_MODEL_PACKAGES,
    VERIFICATION_TASK,
    declared_model_sha256,
    materialize_model_package,
    normalize_preprocessing,
    validate_model_package_selection,
)

_RECOGNITION_DEFAULTS: dict[str, Any] = {
    "mode": "all",
    "reference_dir": None,
    "unknown_action": "auto",
    "profile": "balanced",
    "similarity_threshold": 0.40,
}

DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "base.yaml"


def read_default_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read base defaults without resolving providers or loading any models.

    Each call returns an independent mapping. GUI defaults and reset actions
    can share the pipeline's YAML values without running its startup work.
    """

    source = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with source.expanduser().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if (
        not isinstance(config, dict)
        or type(config.get("schema_version")) is not int
        or config["schema_version"] != 1
    ):
        raise ValueError("Base defaults must be a schema_version: 1 mapping")
    return config


_OPTIONAL_CONFIG_SCHEMA: dict[str, Any] = {
    "recognition": {"max_frames_per_track": None},
    "scan": {
        "passes": [
            {
                "candidate_filter": {
                    "max_height_width_ratio": None,
                }
            }
        ]
    },
    "revalidation": {
        "passes": [
            {
                "name": None,
                "input_size": None,
                "crop_expansion": None,
            }
        ]
    },
    "render": {
        "redaction": {
            "mosaic": {
                "block_size_ratio": None,
                "min_block_size": None,
            },
            "feather": {
                "ratio": None,
                "min_pixels": None,
            },
        },
        "video_output": {
            "rate_control": {
                "bitrate": None,
                "max_bitrate": None,
                "buffer_size": None,
            }
        },
    },
}


def _schema_from_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _schema_from_value(item) for key, item in value.items()}
    if isinstance(value, list):
        item_schema: dict[str, Any] = {}
        for item in value:
            if isinstance(item, dict):
                _merge_schema(item_schema, _schema_from_value(item))
        return [item_schema] if item_schema else [None]
    return None


def _merge_schema(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _merge_schema(current, value)
        elif (
            isinstance(current, list)
            and current
            and isinstance(current[0], dict)
            and isinstance(value, list)
            and value
            and isinstance(value[0], dict)
        ):
            _merge_schema(current[0], value[0])
        else:
            target[key] = deepcopy(value)


def _current_config_schema() -> dict[str, Any]:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"invalid built-in config schema: {DEFAULT_CONFIG_PATH}")
    schema = _schema_from_value(raw)
    _merge_schema(schema, _OPTIONAL_CONFIG_SCHEMA)
    schema["base_config"] = None
    return schema


def _validate_config_keys(value: Any, schema: Any, path: str) -> None:
    if isinstance(value, dict) and isinstance(schema, dict):
        unknown = sorted(set(value) - set(schema))
        if unknown:
            field = f"{path}.{unknown[0]}" if path else unknown[0]
            raise ValueError(f"unknown configuration setting: {field}")
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else key
            _validate_config_keys(item, schema[key], child_path)
    elif isinstance(value, list) and isinstance(schema, list) and schema:
        for index, item in enumerate(value):
            _validate_config_keys(item, schema[0], f"{path}[{index}]")


def validate_config_keys(
    config: dict[str, Any],
    *,
    allow_base_config: bool = False,
) -> None:
    """Reject unknown keys against the current Base configuration contract."""

    schema = _current_config_schema()
    if not allow_base_config:
        schema.pop("base_config", None)
    _validate_config_keys(config, schema, "")


def resolve_runtime_provider(config: dict[str, Any]) -> None:
    """Resolve the YAML-configured ONNX Runtime provider into execution order."""

    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise TypeError("runtime settings must be a mapping")
    static_shape_sessions = runtime.setdefault(
        "scrfd_static_shape_sessions",
        True,
    )
    if not isinstance(static_shape_sessions, bool):
        raise TypeError(
            "runtime.scrfd_static_shape_sessions must be boolean"
        )
    available = ort.get_available_providers()
    requested = runtime.get("provider", "auto")
    if not isinstance(requested, str):
        raise TypeError("runtime.provider must be a string")
    if requested == "auto":
        providers = get_default_providers(available)
        requested = providers[0]
    else:
        providers = [requested]
        if (
            requested != "CPUExecutionProvider"
            and "CPUExecutionProvider" in available
        ):
            providers.append("CPUExecutionProvider")
    if requested not in available:
        raise RuntimeError(f"ONNX Runtime provider {requested} is unavailable; available={available}")
    runtime["providers"] = providers
    runtime["resolved_provider"] = requested


def validate_recognition(config: dict[str, Any], root: Path) -> None:
    """Validate photo matching and the action for unconfirmed faces."""

    raw = config.setdefault("recognition", deepcopy(_RECOGNITION_DEFAULTS))
    if not isinstance(raw, dict):
        raise TypeError("recognition settings must be a mapping")
    unknown = set(raw) - {*set(_RECOGNITION_DEFAULTS), "max_frames_per_track"}
    if unknown:
        raise ValueError("unknown recognition settings: " + ", ".join(sorted(unknown)))
    for key, value in _RECOGNITION_DEFAULTS.items():
        raw.setdefault(key, deepcopy(value))
    mode = str(raw["mode"])
    if mode not in {"all", "blur_only", "exempt"}:
        raise ValueError("recognition.mode must be all, blur_only, or exempt")
    raw["mode"] = mode
    action = raw["unknown_action"]
    if not isinstance(action, str) or action not in {"auto", "blur", "keep"}:
        raise ValueError("recognition.unknown_action must be auto, blur, or keep")
    # The all policy is deliberately a highest-level early exit. Reference
    # paths and selective settings must not cause filesystem access or model
    # loading when identity selection is not requested.
    if mode == "all":
        return

    profile = str(raw["profile"])
    if profile not in {"fast", "balanced", "accurate"}:
        raise ValueError("recognition.profile must be fast, balanced, or accurate")
    raw["profile"] = profile
    if "max_frames_per_track" in raw and raw["max_frames_per_track"] is not None:
        maximum = raw["max_frames_per_track"]
        if isinstance(maximum, bool) or not isinstance(maximum, int):
            raise TypeError("recognition.max_frames_per_track must be an integer or null")
        if not 1 <= maximum <= 32:
            raise ValueError("recognition.max_frames_per_track must be between 1 and 32")
    threshold = raw["similarity_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("recognition.similarity_threshold must be a number")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("recognition.similarity_threshold must be in [0, 1]")
    raw["similarity_threshold"] = threshold

    reference_text = raw.get("reference_dir")
    if not isinstance(reference_text, str) or not reference_text.strip():
        raise ValueError("recognition.reference_dir is required for selective modes")
    unresolved = Path(reference_text.strip()).expanduser()
    candidate = unresolved if unresolved.is_absolute() else root / unresolved
    if candidate.is_symlink():
        raise ValueError("recognition.reference_dir must not be a symlink")
    reference_dir = candidate.resolve()
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"Reference photo directory does not exist: {reference_dir}")
    raw["reference_dir"] = str(reference_dir)


_SUPPORTED_DETECTION_ANGLES = frozenset({-90, 0, 90})


def _validate_detector_input_size(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value <= 0 or value % 32:
        raise ValueError(f"{field} must be a positive multiple of 32")
    return value


def _validate_detection_angles(value: Any, *, field: str) -> None:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{field} must be a non-empty list")
    for angle in value:
        if isinstance(angle, bool) or not isinstance(angle, int):
            raise TypeError(f"{field} entries must be integers")
        if angle not in _SUPPORTED_DETECTION_ANGLES:
            raise ValueError(f"{field} entries must be one of -90, 0, or 90")


def _validate_confidence_threshold(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return threshold


def _validate_positive_finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _validate_scan_padding(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    ratio = float(value)
    # Padding is applied independently to both sides of the source dimension.
    # One source dimension per side is already a 3x canvas, so larger ratios
    # have no useful scan contract and create disproportionate memory pressure.
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return ratio


def validate_endpoint_conflicts(config: dict[str, Any]) -> None:
    """Validate endpoint-evidence gates and hardware-independent work limits."""

    prefix = "tracking.endpoint_conflicts"
    tracking = config.get("tracking")
    settings = tracking.get("endpoint_conflicts") if isinstance(tracking, dict) else None
    if not isinstance(settings, dict):
        raise TypeError(f"{prefix} must be a mapping")
    fractions = {
        "nearby_iou", "match_min_iou", "match_min_confidence", "match_min_margin",
    }
    positive_numbers = {"nearby_center_distance", "match_max_center_distance"}
    nonnegative_integers = {"max_calls_per_frame", "max_calls_total", "cache_entries"}
    expected = {
        "enabled", "angles", "equivalence_iou", "match_max_area_ratio",
        "recheck_min_frame_gap", "max_calls_per_video_second",
        *fractions, *positive_numbers, *nonnegative_integers,
    }
    unknown = set(settings) - expected
    if unknown:
        raise ValueError(f"unknown {prefix} settings: " + ", ".join(sorted(unknown)))
    missing = expected - set(settings)
    if missing:
        raise ValueError(f"missing {prefix} settings: " + ", ".join(sorted(missing)))
    if not isinstance(settings["enabled"], bool):
        raise TypeError(f"{prefix}.enabled must be boolean")
    _validate_detection_angles(settings["angles"], field=f"{prefix}.angles")
    if len(set(settings["angles"])) != len(settings["angles"]):
        raise ValueError(f"{prefix}.angles must not contain duplicates")
    for key in sorted(fractions):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{prefix}.{key} must be a number")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{prefix}.{key} must be finite and in [0, 1]")
    for key in sorted(positive_numbers):
        _validate_positive_finite(settings[key], field=f"{prefix}.{key}")
    for key, minimum, exclusive in (
        ("equivalence_iou", 0.0, True),
        ("match_max_area_ratio", 1.0, False),
        ("max_calls_per_video_second", 0.0, False),
    ):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{prefix}.{key} must be a number")
        number = float(value)
        if not math.isfinite(number) or (number <= minimum if exclusive else number < minimum):
            bound = "greater than" if exclusive else "at least"
            raise ValueError(f"{prefix}.{key} must be finite and {bound} {minimum:g}")
        if key == "equivalence_iou" and number > 1.0:
            raise ValueError(f"{prefix}.{key} must be in (0, 1]")
    for key in sorted(nonnegative_integers | {"recheck_min_frame_gap"}):
        value = settings[key]
        minimum = 1 if key == "recheck_min_frame_gap" else 0
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{prefix}.{key} must be an integer")
        if value < minimum:
            raise ValueError(f"{prefix}.{key} must be at least {minimum}")


def validate_revalidation_passes(config: dict[str, Any]) -> None:
    """Validate the shared SCRFD local-review settings and optional cascade."""

    settings = config["revalidation"]
    if not isinstance(settings, dict):
        raise TypeError("revalidation configuration must be a mapping")
    _validate_detector_input_size(
        settings.get("input_size"),
        field="revalidation.input_size",
    )
    _validate_detection_angles(
        settings.get("angles"),
        field="revalidation.angles",
    )
    _validate_confidence_threshold(
        settings.get("confidence_threshold"),
        field="revalidation.confidence_threshold",
    )
    _validate_positive_finite(
        settings.get("crop_expansion"),
        field="revalidation.crop_expansion",
    )

    passes = settings.get("passes")
    if passes is None:
        return
    if not isinstance(passes, list) or not passes:
        raise TypeError("revalidation.passes must be a non-empty list")
    names: set[str] = set()
    for detector_pass in passes:
        if not isinstance(detector_pass, dict):
            raise TypeError("each revalidation pass must be a mapping")
        name = str(detector_pass.get("name", ""))
        if not name or name in names:
            raise ValueError("revalidation pass names must be non-empty and unique")
        names.add(name)
        _validate_detector_input_size(
            detector_pass.get("input_size"),
            field=f"revalidation pass {name} input_size",
        )
        _validate_positive_finite(
            detector_pass.get("crop_expansion"),
            field=f"revalidation pass {name} crop_expansion",
        )


def validate_scan_passes(config: dict[str, Any]) -> None:
    """Validate full-frame SCRFD views and optional raw-candidate gates."""

    scan = config["scan"]
    scan["max_analysis_fps"] = _validate_positive_finite(
        scan.get("max_analysis_fps", 15),
        field="scan.max_analysis_fps",
    )
    tracking = config.get("tracking")
    if isinstance(tracking, dict):
        endpoint_extension = tracking.get("endpoint_extension")
        if endpoint_extension is not None:
            if isinstance(endpoint_extension, bool) or not isinstance(endpoint_extension, int):
                raise TypeError("tracking.endpoint_extension must be an integer")
        association_scan_gap = tracking.get("association_max_scan_gap")
        if association_scan_gap is not None:
            if (
                isinstance(association_scan_gap, bool)
                or not isinstance(association_scan_gap, int)
            ):
                raise TypeError(
                    "tracking.association_max_scan_gap must be an integer"
                )
            if association_scan_gap < 1:
                raise ValueError(
                    "tracking.association_max_scan_gap must be positive"
                )
        association_gap_seconds = tracking.get("association_max_gap_seconds")
        if association_gap_seconds is not None:
            _validate_positive_finite(
                association_gap_seconds,
                field="tracking.association_max_gap_seconds",
            )
        strict_geometry_seconds = tracking.get(
            "association_strict_geometry_after_seconds"
        )
        if strict_geometry_seconds is not None:
            strict_geometry_seconds = _validate_positive_finite(
                strict_geometry_seconds,
                field="tracking.association_strict_geometry_after_seconds",
            )
            if (
                association_gap_seconds is not None
                and strict_geometry_seconds > float(association_gap_seconds)
            ):
                raise ValueError(
                    "tracking.association_strict_geometry_after_seconds cannot "
                    "exceed tracking.association_max_gap_seconds"
                )

    passes = scan.get("passes")
    if not isinstance(passes, list) or not passes:
        raise TypeError("scan.passes must be a non-empty list")
    names: set[str] = set()
    for detector_pass in passes:
        if not isinstance(detector_pass, dict):
            raise TypeError("each scan pass must be a mapping")
        name = str(detector_pass.get("name", ""))
        if not name or name in names:
            raise ValueError("scan pass names must be non-empty and unique")
        names.add(name)
        _validate_detector_input_size(
            detector_pass.get("input_size"),
            field=f"scan pass {name} input_size",
        )
        _validate_detection_angles(
            detector_pass.get("angles"),
            field=f"scan pass {name} angles",
        )
        _validate_confidence_threshold(
            detector_pass.get("confidence_threshold"),
            field=f"scan pass {name} confidence_threshold",
        )
        _validate_scan_padding(
            detector_pass.get("horizontal_padding_ratio"),
            field=f"scan pass {name} horizontal_padding_ratio",
        )
        _validate_scan_padding(
            detector_pass.get("vertical_padding_ratio"),
            field=f"scan pass {name} vertical_padding_ratio",
        )
        candidate_filter = detector_pass.get("candidate_filter")
        if candidate_filter is None:
            continue
        if not isinstance(candidate_filter, dict) or not isinstance(
            candidate_filter.get("enabled", False), bool
        ):
            raise TypeError("scan pass candidate_filter.enabled must be boolean")
        if not bool(candidate_filter.get("enabled", False)):
            continue
        area_fraction = float(candidate_filter.get("min_box_area_fraction", -1.0))
        if not 0.0 <= area_fraction <= 1.0:
            raise ValueError("scan pass candidate_filter.min_box_area_fraction must be in [0, 1]")
        minimum_ratio = float(candidate_filter.get("min_height_width_ratio", 0.0))
        maximum_ratio_raw = candidate_filter.get("max_height_width_ratio")
        maximum_ratio = float(maximum_ratio_raw) if maximum_ratio_raw is not None else None
        if minimum_ratio <= 0.0 or (maximum_ratio is not None and maximum_ratio < minimum_ratio):
            raise ValueError(
                "scan pass candidate_filter height/width limits must be positive "
                "and increasing when a maximum is configured"
            )
        aspect_exemption = float(candidate_filter.get("aspect_ratio_exempt_min_area_fraction", 1.0))
        if not area_fraction <= aspect_exemption <= 1.0:
            raise ValueError(
                "scan pass candidate_filter aspect-ratio exemption must be between its minimum area and 1"
            )


def validate_bidirectional_fusion(config: dict[str, Any]) -> None:
    """Validate the one supported detector-bounded fusion algorithm."""

    flow = config["tracking"]["kalman_optical_flow"]
    settings = flow.get("bidirectional_fusion")
    if not isinstance(settings, dict):
        raise TypeError("tracking.kalman_optical_flow.bidirectional_fusion must be a mapping")
    required = {
        "max_gap_frames",
        "max_materialized_bytes",
        "association_rescue",
        "corridor_expansion",
        "max_corridor_side_pixels",
        "mutual_min_iou",
        "mutual_max_center_distance",
        "anchor_max_center_distance",
        "anchor_max_area_ratio",
        "local_review_min_confidence",
        "local_review_min_iou",
        "local_review_min_iou_margin",
        "soft_bias_beta",
        "soft_bias_radius",
        "soft_confidence_low",
        "soft_confidence_high",
        "geometry_bridge",
    }
    missing = required - set(settings)
    extra = set(settings) - required
    if missing or extra:
        raise ValueError(
            "bidirectional_fusion keys must match the current algorithm; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if float(settings["soft_bias_beta"]) < 0.0:
        raise ValueError("bidirectional_fusion.soft_bias_beta cannot be negative")
    if int(settings["soft_bias_radius"]) < 0:
        raise ValueError("bidirectional_fusion.soft_bias_radius cannot be negative")
    confidence_low = float(settings["soft_confidence_low"])
    confidence_high = float(settings["soft_confidence_high"])
    if not 0.0 <= confidence_low < confidence_high <= 1.0:
        raise ValueError("bidirectional_fusion soft confidence range must increase within [0, 1]")
    if float(settings["corridor_expansion"]) <= 1.0:
        raise ValueError("bidirectional_fusion.corridor_expansion must exceed 1")
    value = settings["max_gap_frames"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("bidirectional_fusion.max_gap_frames must be a positive integer")
    value = settings["max_materialized_bytes"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1024 * 1024:
        raise ValueError("bidirectional_fusion.max_materialized_bytes must be an integer of at least 1 MiB")
    if int(settings["max_corridor_side_pixels"]) < int(flow["roi_size"]):
        raise ValueError("bidirectional_fusion.max_corridor_side_pixels must cover roi_size")
    for key in (
        "mutual_min_iou",
        "local_review_min_confidence",
        "local_review_min_iou",
        "local_review_min_iou_margin",
    ):
        value = float(settings[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"bidirectional_fusion.{key} must be in [0, 1]")
    for key in (
        "mutual_max_center_distance",
        "anchor_max_center_distance",
    ):
        if float(settings[key]) <= 0.0:
            raise ValueError(f"bidirectional_fusion.{key} must be positive")
    if float(settings["anchor_max_area_ratio"]) < 1.0:
        raise ValueError("bidirectional_fusion.anchor_max_area_ratio must be at least 1")
    geometry_bridge = settings.get("geometry_bridge")
    if not isinstance(geometry_bridge, dict):
        raise TypeError("bidirectional_fusion.geometry_bridge must be a mapping")
    required_geometry_bridge = {
        "enabled",
        "edge_epsilon_pixels",
        "min_edge_expansion_ratio",
        "min_both_trusted_fraction",
        "min_mutual_consistent_fraction",
    }
    if set(geometry_bridge) != required_geometry_bridge:
        raise ValueError("geometry_bridge keys must match the current algorithm")
    if not isinstance(geometry_bridge["enabled"], bool):
        raise TypeError("geometry_bridge.enabled must be boolean")
    if float(geometry_bridge["edge_epsilon_pixels"]) < 0.0:
        raise ValueError("geometry_bridge.edge_epsilon_pixels cannot be negative")
    if float(geometry_bridge["min_edge_expansion_ratio"]) <= 1.0:
        raise ValueError("geometry_bridge.min_edge_expansion_ratio must exceed 1")
    for key in (
        "min_both_trusted_fraction",
        "min_mutual_consistent_fraction",
    ):
        value = float(geometry_bridge[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"geometry_bridge.{key} must be in [0, 1]")
    rescue = settings.get("association_rescue")
    if not isinstance(rescue, dict):
        raise TypeError("bidirectional_fusion.association_rescue must be a mapping")
    required_rescue = {
        "enabled",
        "max_endpoint_center_speed",
        "max_area_ratio",
        "min_endpoint_iou",
        "max_endpoint_center_distance",
    }
    if set(rescue) != required_rescue:
        raise ValueError("association_rescue keys must match the current algorithm")
    if not isinstance(rescue["enabled"], bool):
        raise TypeError("association_rescue.enabled must be boolean")
    if float(rescue["max_endpoint_center_speed"]) <= 0.0:
        raise ValueError("association_rescue.max_endpoint_center_speed must be positive")
    if float(rescue["max_area_ratio"]) < 1.0:
        raise ValueError("association_rescue.max_area_ratio must be at least 1")
    minimum_iou = float(rescue["min_endpoint_iou"])
    if not 0.0 <= minimum_iou <= 1.0:
        raise ValueError("association_rescue.min_endpoint_iou must be in [0, 1]")
    if float(rescue["max_endpoint_center_distance"]) <= 0.0:
        raise ValueError("association_rescue.max_endpoint_center_distance must be positive")


def validate_scene_cut_detector(scan: dict[str, Any]) -> None:
    """Validate the adaptive LK scene-cut detector."""

    settings = scan.get("scene_cut_detector")
    if not isinstance(settings, dict):
        raise TypeError("scan.scene_cut_detector must be a mapping")
    if "mode" in settings:
        raise ValueError("scan.scene_cut_detector.mode is no longer configurable")
    if int(settings.get("signature_size", 0)) < 32:
        raise ValueError("scene_cut_detector.signature_size must be at least 32")
    if int(settings.get("history_frames", 0)) < 1:
        raise ValueError("scene_cut_detector.history_frames must be positive")
    minimum = float(settings.get("min_mean_absdiff", -1.0))
    bootstrap = float(settings.get("bootstrap_min_mean_absdiff", -1.0))
    if minimum < 0.0 or bootstrap < minimum:
        raise ValueError(
            "scene-cut mean-absdiff thresholds must be non-negative and "
            "bootstrap must not be lower than the steady-state minimum"
        )
    if float(settings.get("relative_multiplier", 0.0)) <= 1.0:
        raise ValueError("scene_cut_detector.relative_multiplier must exceed 1")
    if float(settings.get("relative_offset", -1.0)) < 0.0:
        raise ValueError("scene_cut_detector.relative_offset cannot be negative")
    minimum_corners = int(settings.get("min_corners", 0))
    maximum_corners = int(settings.get("max_corners", 0))
    if minimum_corners < 4 or maximum_corners < minimum_corners:
        raise ValueError("scene-cut LK corners require max_corners >= min_corners >= 4")
    quality = float(settings.get("quality_level", 0.0))
    if not 0.0 < quality <= 1.0:
        raise ValueError("scene_cut_detector.quality_level must be in (0, 1]")
    if float(settings.get("min_distance", 0.0)) <= 0.0:
        raise ValueError("scene_cut_detector.min_distance must be positive")
    block = int(settings.get("block_size", 0))
    window = int(settings.get("lk_window_size", 0))
    if block < 3 or block % 2 == 0:
        raise ValueError("scene_cut_detector.block_size must be odd and at least 3")
    if window < 3 or window % 2 == 0:
        raise ValueError("scene_cut_detector.lk_window_size must be odd and at least 3")
    if int(settings.get("lk_max_level", -1)) < 0:
        raise ValueError("scene_cut_detector.lk_max_level cannot be negative")
    if int(settings.get("lk_max_iterations", 0)) < 1:
        raise ValueError("scene_cut_detector.lk_max_iterations must be positive")
    for key in (
        "lk_epsilon",
        "lk_min_eigenvalue",
        "max_forward_backward_error",
        "ransac_reprojection_threshold",
    ):
        if float(settings.get(key, 0.0)) <= 0.0:
            raise ValueError(f"scene_cut_detector.{key} must be positive")
    inlier_limit = float(settings.get("max_flow_inlier_fraction", -1.0))
    if not 0.0 <= inlier_limit <= 1.0:
        raise ValueError("scene_cut_detector.max_flow_inlier_fraction must be in [0, 1]")
    appearance = settings.get("appearance_confirmation")
    if appearance is not None:
        if not isinstance(appearance, dict) or not isinstance(appearance.get("enabled"), bool):
            raise TypeError("scene-cut appearance_confirmation.enabled must be boolean")
        if float(appearance.get("min_mean_absdiff", -1.0)) < 0.0:
            raise ValueError("appearance_confirmation.min_mean_absdiff cannot be negative")
        for key in (
            "max_histogram_correlation",
            "max_spatial_correlation",
        ):
            correlation = float(appearance.get(key, -2.0))
            if not -1.0 <= correlation <= 1.0:
                raise ValueError(f"appearance_confirmation.{key} must be in [-1, 1]")
        appearance_flow_limit = float(appearance.get("max_flow_inlier_fraction", -1.0))
        if not 0.0 <= appearance_flow_limit <= 1.0:
            raise ValueError("appearance_confirmation.max_flow_inlier_fraction must be in [0, 1]")
    flash = settings.get("flash_suppression")
    if not isinstance(flash, dict) or not isinstance(flash.get("enabled"), bool):
        raise TypeError("scene-cut flash_suppression.enabled must be boolean")
    if float(flash.get("max_skip_mean_absdiff", -1.0)) < 0.0:
        raise ValueError("flash max_skip_mean_absdiff cannot be negative")
    skip_ratio = float(flash.get("max_skip_to_transition_ratio", -1.0))
    if not 0.0 < skip_ratio <= 1.0:
        raise ValueError("flash max_skip_to_transition_ratio must be in (0, 1]")
    correlation = float(flash.get("min_skip_spatial_correlation", -2.0))
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("flash min_skip_spatial_correlation must be in [-1, 1]")


def validate_admission_policy(config: dict[str, Any]) -> None:
    """Validate the one supported track-admission rule gate."""

    policy = config["revalidation"]["policy"]
    if not isinstance(policy, dict):
        raise TypeError("revalidation.policy must be a mapping")
    expected = {"rule_gate", "continuity"}
    if set(policy) != expected:
        raise ValueError("revalidation.policy keys must be exactly rule_gate and continuity")
    gate = policy.get("rule_gate")
    if not isinstance(gate, dict):
        raise TypeError("rule_gate policy settings are required")
    if not gate:
        raise ValueError("rule_gate policy settings must not be empty")
    if gate:
        standard_minimum = int(gate.get("min_detector_frames", 0))
        if standard_minimum < 1:
            raise ValueError("rule_gate.min_detector_frames must be positive")
        for key in (
            "min_local_match_fraction",
            "min_local_confidence_fraction_gte_035",
            "min_joint_strong_anchor",
            "strong_joint_anchor",
            "min_verifier_pass_fraction",
        ):
            value = float(gate.get(key, -1.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"rule_gate.{key} must be between 0 and 1")
        if float(gate["strong_joint_anchor"]) < float(gate["min_joint_strong_anchor"]):
            raise ValueError("strong_joint_anchor must not be below min_joint_strong_anchor")
        if int(gate.get("strong_anchor_window_frames", 0)) < 2:
            raise ValueError("strong_anchor_window_frames must be at least 2")
        if not isinstance(gate.get("normalizers"), dict):
            raise ValueError("rule_gate.normalizers are required")
        normalizers = gate["normalizers"]
        for key in (
            "local_confidence_low",
            "local_confidence_high",
            "verifier_score_low",
            "verifier_score_high",
        ):
            if key not in normalizers:
                raise ValueError(f"rule_gate.normalizers.{key} is required")

        short = gate.get("short_track")
        if not isinstance(short, dict):
            raise TypeError("rule_gate.short_track must be a mapping")
        if not isinstance(short.get("enabled"), bool):
            raise TypeError("rule_gate.short_track.enabled must be boolean")
        short_minimum = int(short.get("min_detector_frames", 0))
        short_maximum = int(short.get("max_detector_frames", 0))
        if short_minimum < 1:
            raise ValueError("short_track.min_detector_frames must be positive")
        if short_maximum < short_minimum:
            raise ValueError("short_track detector frame range must be increasing")
        if short_maximum >= standard_minimum:
            raise ValueError("short_track.max_detector_frames must stay below rule_gate.min_detector_frames")
        for key in (
            "min_local_match_fraction",
            "moderate_local_confidence_p50",
            "moderate_verifier_p50",
            "strong_local_confidence_p50",
            "strong_verifier_p50",
        ):
            value = float(short.get(key, -1.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"short_track.{key} must be between 0 and 1")
        if float(short["strong_local_confidence_p50"]) < float(short["moderate_local_confidence_p50"]):
            raise ValueError("short_track strong local threshold must not be below its moderate threshold")
        if float(short["strong_verifier_p50"]) < float(
            short["moderate_verifier_p50"]
        ):
            raise ValueError(
                "short_track strong Verifier threshold must not be below its moderate threshold"
            )
        video_start = gate.get("video_start_short_track")
        if video_start is not None:
            if not isinstance(video_start, dict):
                raise TypeError("rule_gate.video_start_short_track must be a mapping")
            if not isinstance(video_start.get("enabled"), bool):
                raise TypeError("rule_gate.video_start_short_track.enabled must be boolean")
            video_start_minimum = int(video_start.get("min_detector_frames", 0))
            if not 1 <= video_start_minimum <= short_maximum:
                raise ValueError(
                    "video_start_short_track.min_detector_frames must fit the short-track detector range"
                )
            for key in (
                "min_detector_confidence_p50",
                "min_local_match_fraction",
                "min_local_confidence_p50",
            ):
                value = float(video_start.get(key, -1.0))
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"video_start_short_track.{key} must be between 0 and 1")
        continuity = policy.get("continuity")
        if not isinstance(continuity, dict):
            raise TypeError("rule_gate continuity settings are required")
        if float(continuity.get("segment_max_center_jump", 0.0)) <= 0.0:
            raise ValueError("continuity.segment_max_center_jump must be positive")
        if float(continuity.get("segment_max_area_ratio", 0.0)) < 1.0:
            raise ValueError("continuity.segment_max_area_ratio must be at least 1")


def validate_model_package_contracts(config: dict[str, Any]) -> None:
    """Validate effective inference settings against the selected model pack."""

    models = config.get("models")
    if not isinstance(models, dict):
        raise TypeError("models configuration is required")
    expected_model_keys = {
        "name",
        "root",
        "manifest_path",
        DETECTION_TASK,
        VERIFICATION_TASK,
    }
    recognition = config.get("recognition", {})
    selective = (
        str(recognition.get("mode", "all")) != "all"
        if isinstance(recognition, Mapping)
        else False
    )
    if selective:
        expected_model_keys.add(RECOGNITION_TASK)
    if set(models) != expected_model_keys:
        raise ValueError("effective models keys do not match the raccoon package contract")
    if models.get("name") not in SUPPORTED_MODEL_PACKAGES:
        raise ValueError("model package must be raccoon_s or raccoon_l")
    root = models.get("root")
    if not isinstance(root, str):
        raise TypeError("models.root must be a string")
    if not root.strip():
        raise ValueError("models.root must be a non-empty path")
    expected_task_keys = {
        DETECTION_TASK: {
            "file",
            "preprocessing",
            "preprocessing_version",
            "path",
            "nms_iou_threshold",
            "max_detections",
        },
        VERIFICATION_TASK: {
            "file",
            "expansion",
            "preprocessing",
            "path",
        },
        RECOGNITION_TASK: {
            "file",
            "preprocessing",
            "preprocessing_version",
            "input_size",
            "embedding_dimension",
            "path",
        },
    }
    required_tasks = [DETECTION_TASK, VERIFICATION_TASK]
    if selective:
        required_tasks.append(RECOGNITION_TASK)
    for task in required_tasks:
        required = expected_task_keys[task]
        settings = models.get(task)
        if not isinstance(settings, dict):
            raise TypeError(f"models.{task} configuration is required")
        if not required <= set(settings) or set(settings) - (required | {"sha256"}):
            raise ValueError(
                f"effective models.{task} keys do not match the manifest contract"
            )
        declared_model_sha256(settings, field=f"models.{task}.sha256")
        normalize_preprocessing(
            settings["preprocessing"],
            f"models.{task}.preprocessing",
        )

    revalidation = config.get("revalidation")
    if not isinstance(revalidation, dict):
        raise TypeError("revalidation configuration is required")


def validate_current_config_contract(config: dict[str, Any]) -> None:
    """Reject settings whose runtime branches were removed from this release."""

    def reject(mapping: Any, prefix: str, removed: set[str]) -> None:
        if not isinstance(mapping, dict):
            return
        found = sorted(set(mapping) & removed)
        if found:
            raise ValueError(
                f"{prefix}.{found[0]} is no longer configurable"
            )

    scan = config.get("scan")
    tracking = config.get("tracking")
    revalidation = config.get("revalidation")
    render = config.get("render")
    reject(
        scan,
        "scan",
        {"pose", "scene_cut_mean_absdiff", "scene_cut_histogram_correlation"},
    )
    reject(
        tracking,
        "tracking",
        {
            "backend",
            "detector_competition_iou",
            "max_area_ratio",
            "max_center_distance",
            "max_internal_gap",
            "min_area_ratio",
        },
    )
    reject(
        tracking.get("kalman_optical_flow")
        if isinstance(tracking, dict)
        else None,
        "tracking.kalman_optical_flow",
        {
            "adaptive_edge_levels",
            "adaptive_edge_min_detector_frames",
            "padding_pixels",
            "validation",
            "validation_max_aspect_ratio",
            "workers",
        },
    )
    reject(revalidation, "revalidation", {"pose", "workers"})
    reject(
        revalidation.get("edge_fallback")
        if isinstance(revalidation, dict)
        else None,
        "revalidation.edge_fallback",
        {"trigger"},
    )
    reject(
        render,
        "render",
        {"blur_expansion_ratio", "blur_kernel_ratio", "min_blur_kernel"},
    )
    reject(
        render.get("box_stabilization") if isinstance(render, dict) else None,
        "render.box_stabilization",
        {"mode"},
    )


_ConfigPathToken = str | int


def _merge_config(
    target: dict[str, Any],
    update: dict[str, Any],
    path: tuple[_ConfigPathToken, ...] = (),
) -> None:
    for key, value in update.items():
        child_path = (*path, key)
        if child_path[-1] == "rate_control":
            target[key] = deepcopy(value)
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_config(target[key], value, child_path)
        else:
            target[key] = deepcopy(value)


def _public_config_override_schema() -> dict[str, Any]:
    schema = _current_config_schema()
    # These fields select/describe the document contract; they are not runtime
    # knobs. ``base_config`` has already been resolved by the time dotted
    # overrides are applied.
    schema.pop("base_config", None)
    schema.pop("schema_version", None)
    return schema


def _parse_config_override_path(
    path: Any,
    schema: dict[str, Any],
) -> tuple[_ConfigPathToken, ...]:
    if not isinstance(path, str) or not path:
        raise TypeError("configuration override paths must be non-empty strings")
    if "." not in path:
        raise ValueError(
            f"configuration override path must use a dotted YAML field: {path}"
        )
    raw_parts = path.split(".")
    if any(not part for part in raw_parts):
        raise ValueError(f"invalid configuration override path: {path}")
    cursor: Any = schema
    tokens: list[_ConfigPathToken] = []
    for part in raw_parts:
        if isinstance(cursor, dict):
            if part not in cursor:
                raise ValueError(f"unknown configuration override path: {path}")
            tokens.append(part)
            cursor = cursor[part]
            continue
        if isinstance(cursor, list):
            if not part.isdigit() or (len(part) > 1 and part.startswith("0")):
                raise ValueError(
                    f"configuration override path requires a non-negative list index: {path}"
                )
            tokens.append(int(part))
            cursor = cursor[0] if cursor else None
            continue
        raise ValueError(f"unknown configuration override path: {path}")
    return tuple(tokens)


def _validate_config_override_value(
    value: Any,
    *,
    path: str,
    active_containers: set[int] | None = None,
) -> None:
    active = set() if active_containers is None else active_containers
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"configuration override {path} must be finite")
        return
    if isinstance(value, (list, dict)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"configuration override {path} must not be cyclic")
        active.add(identity)
        if isinstance(value, list):
            for item in value:
                _validate_config_override_value(
                    item,
                    path=path,
                    active_containers=active,
                )
        else:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"configuration override {path} mapping keys must be strings"
                    )
                _validate_config_override_value(
                    item,
                    path=path,
                    active_containers=active,
                )
        active.remove(identity)
        return
    raise TypeError(
        f"configuration override {path} must use JSON-compatible YAML types"
    )


def _prepare_config_override_items(
    items: Sequence[tuple[Any, Any]],
) -> list[tuple[str, tuple[_ConfigPathToken, ...], Any]]:
    schema = _public_config_override_schema()
    prepared: list[tuple[str, tuple[_ConfigPathToken, ...], Any]] = []
    for path, value in items:
        tokens = _parse_config_override_path(path, schema)
        _validate_config_override_value(value, path=path)
        prepared.append((path, tokens, value))
    for index, (left_path, left, _left_value) in enumerate(prepared):
        for right_path, right, _right_value in prepared[index + 1 :]:
            shortest = min(len(left), len(right))
            if left[:shortest] == right[:shortest]:
                raise ValueError(
                    "configuration override paths overlap: "
                    f"{left_path} and {right_path}"
                )
    return prepared


def validate_config_override_paths(paths: Sequence[str]) -> None:
    """Validate CLI dotted paths, including duplicates and parent overlaps."""

    _prepare_config_override_items([(path, None) for path in paths])


def _config_override_child(
    container: Any,
    token: _ConfigPathToken,
    child_schema: Any,
    *,
    path: str,
) -> Any:
    if isinstance(token, str):
        if not isinstance(container, dict):
            raise TypeError(f"cannot traverse configuration override path: {path}")
        if token not in container:
            if not isinstance(child_schema, dict):
                raise ValueError(f"configuration override path is absent: {path}")
            container[token] = {}
        return container[token]
    if not isinstance(container, list):
        raise TypeError(f"cannot traverse configuration override path: {path}")
    if token >= len(container):
        raise IndexError(f"configuration override list index is out of range: {path}")
    return container[token]


def _assign_config_override(
    container: Any,
    token: _ConfigPathToken,
    value: Any,
    *,
    path: str,
    merge_path: tuple[_ConfigPathToken, ...],
) -> None:
    if isinstance(token, str):
        if not isinstance(container, dict):
            raise TypeError(f"cannot assign configuration override path: {path}")
        previous = container.get(token)
        if token == "rate_control":
            container[token] = deepcopy(value)
        elif isinstance(previous, dict) and isinstance(value, dict):
            _merge_config(previous, value, merge_path)
        else:
            container[token] = deepcopy(value)
        return
    if not isinstance(container, list):
        raise TypeError(f"cannot assign configuration override path: {path}")
    if token >= len(container):
        raise IndexError(f"configuration override list index is out of range: {path}")
    previous = container[token]
    if isinstance(previous, dict) and isinstance(value, dict):
        _merge_config(previous, value, merge_path)
    else:
        container[token] = deepcopy(value)


def _apply_prepared_config_overrides(
    config: dict[str, Any],
    prepared: Sequence[tuple[str, tuple[_ConfigPathToken, ...], Any]],
) -> None:
    schema = _public_config_override_schema()
    for path, tokens, value in prepared:
        container: Any = config
        schema_cursor: Any = schema
        for token in tokens[:-1]:
            if isinstance(schema_cursor, dict):
                child_schema = schema_cursor[token]
            else:
                child_schema = schema_cursor[0]
            container = _config_override_child(
                container,
                token,
                child_schema,
                path=path,
            )
            schema_cursor = child_schema
        _assign_config_override(
            container,
            tokens[-1],
            value,
            path=path,
            merge_path=tokens,
        )


def apply_config_overrides(
    config: dict[str, Any],
    overrides: Mapping[str, Any],
) -> None:
    """Apply schema-approved dotted overrides to a merged raw configuration."""

    if not isinstance(overrides, Mapping):
        raise TypeError("config_overrides must be a dotted-path mapping")
    prepared = _prepare_config_override_items(list(overrides.items()))
    _apply_prepared_config_overrides(config, prepared)


def _prepared_overrides_reference_dir(
    prepared: Sequence[tuple[str, tuple[_ConfigPathToken, ...], Any]],
) -> bool:
    for _path, tokens, _value in prepared:
        if tokens == ("recognition", "reference_dir"):
            return True
    return False


_NUMBER_FIELDS_WITH_INTEGER_DEFAULTS = {"scan.max_analysis_fps"}
_INTEGER_OR_STRING_FIELDS = {"render.video_output.audio.bitrate"}


def _validate_config_value_types(
    value: Any,
    template: Any,
    path: tuple[str, ...] = (),
) -> None:
    field = ".".join(path) or "configuration"
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    if template is None:
        # Nullable/optional fields have dedicated semantic validators.
        return
    if isinstance(template, dict):
        if not isinstance(value, dict):
            raise TypeError(f"{field} must be a mapping")
        for key, item in value.items():
            if key in template:
                _validate_config_value_types(
                    item,
                    template[key],
                    (*path, str(key)),
                )
        return
    if isinstance(template, list):
        if not isinstance(value, list):
            raise TypeError(f"{field} must be a list")
        if not template:
            return
        for index, item in enumerate(value):
            item_template = template[min(index, len(template) - 1)]
            _validate_config_value_types(
                item,
                item_template,
                (*path, str(index)),
            )
        return
    if isinstance(template, bool):
        if not isinstance(value, bool):
            raise TypeError(f"{field} must be boolean")
        return
    if isinstance(template, int):
        if field in _NUMBER_FIELDS_WITH_INTEGER_DEFAULTS:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field} must be a number")
        elif isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field} must be an integer")
        return
    if isinstance(template, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be a number")
        return
    if isinstance(template, str):
        if field in _INTEGER_OR_STRING_FIELDS:
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise TypeError(f"{field} must be an integer or string")
        elif not isinstance(value, str):
            raise TypeError(f"{field} must be a string")


def _load_base_config(
    path: str | Path,
    *,
    derived_overrides: dict[str, Any] | None = None,
    derived_override_root: Path | None = None,
    dotted_overrides: Mapping[str, Any] | None = None,
    dotted_override_root: Path | None = None,
    materialize_models: bool = True,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if (
        not isinstance(config, dict)
        or type(config.get("schema_version")) is not int
        or config["schema_version"] != 1
    ):
        raise ValueError("ONNX config must be a schema_version: 1 mapping")
    config = deepcopy(config)
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as stream:
        type_template = yaml.safe_load(stream)
    if not isinstance(type_template, dict):
        raise TypeError("packaged Base configuration must be a mapping")
    root = source.parent
    recognition_root = root
    if derived_overrides is not None:
        if derived_override_root is None:
            raise ValueError(
                "derived_override_root is required with derived YAML overrides"
            )
        if not isinstance(derived_overrides, dict):
            raise TypeError("derived YAML overrides must be a mapping")
        recognition_override = derived_overrides.get("recognition")
        if isinstance(recognition_override, dict) and "reference_dir" in recognition_override:
            recognition_root = derived_override_root
        _merge_config(config, derived_overrides)
    if dotted_overrides is not None:
        if not isinstance(dotted_overrides, Mapping):
            raise TypeError("config_overrides must be a dotted-path mapping")
        prepared = _prepare_config_override_items(list(dotted_overrides.items()))
        if prepared:
            if dotted_override_root is None:
                raise ValueError("config_override_root is required with config_overrides")
            if _prepared_overrides_reference_dir(prepared):
                recognition_root = dotted_override_root
            _apply_prepared_config_overrides(config, prepared)
    validate_current_config_contract(config)
    validate_config_keys(config)
    _validate_config_value_types(config, type_template)
    validate_model_package_selection(config)
    if materialize_models:
        materialize_model_package(config)
    resolve_runtime_provider(config)
    if materialize_models:
        validate_model_package_contracts(config)
    validate_scan_passes(config)
    validate_endpoint_conflicts(config)
    validate_scene_cut_detector(config["scan"])
    validate_revalidation_passes(config)
    validate_bidirectional_fusion(config)
    edge_fallback = config["revalidation"].get("edge_fallback", {})
    if bool(edge_fallback.get("enabled", False)):
        shift_fraction = float(edge_fallback.get("shift_fraction", 0.0))
        if not 0.0 < shift_fraction <= 0.25:
            raise ValueError("revalidation.edge_fallback.shift_fraction must be in (0, 0.25]")
    validate_admission_policy(config)
    validate_recognition(config, recognition_root)
    return config


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "apply_config_overrides",
    "resolve_runtime_provider",
    "validate_admission_policy",
    "validate_bidirectional_fusion",
    "validate_config_keys",
    "validate_config_override_paths",
    "validate_current_config_contract",
    "validate_endpoint_conflicts",
    "validate_model_package_contracts",
    "validate_recognition",
    "validate_revalidation_passes",
    "validate_scan_passes",
    "validate_scene_cut_detector",
]
