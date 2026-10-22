"""Configuration validation for the streaming GOP/ROI profile."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .base_config import (
    DEFAULT_CONFIG_PATH,
    _load_base_config,
    validate_config_keys,
    validate_current_config_contract,
)


_MAX_KEYFRAME_INTERVAL = 2_147_483_647
_MAX_BITRATE_BPS = 9_223_372_036_854_775_807


def _reject_unknown_keys(settings: dict[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise ValueError(f"unknown {field} setting: {unknown[0]}")


def _validate_bitrate(value: Any, field: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a positive bitrate")
    try:
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str):
            text = value.strip().lower()
            if text and text[-1] in {"k", "m", "g"}:
                multiplier = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}[text[-1]]
                parsed = round(float(text[:-1]) * multiplier)
            else:
                parsed = int(text)
        else:
            raise TypeError
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            f"{field} must be a positive integer optionally suffixed by k, m, or g"
        ) from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be positive")
    if parsed > _MAX_BITRATE_BPS:
        raise ValueError(f"{field} must not exceed {_MAX_BITRATE_BPS} bits/s")


def validate_video_output(settings: dict[str, Any]) -> None:
    _reject_unknown_keys(
        settings,
        {
            "backend",
            "encoder",
            "pixel_format",
            "preset",
            "rate_control",
            "keyframe_interval",
            "faststart",
            "audio",
        },
        "render.video_output",
    )
    if settings.get("backend") not in {"pyav", "ffmpeg"}:
        raise ValueError("render.video_output.backend must be pyav or ffmpeg")
    if not str(settings.get("encoder", "")):
        raise ValueError("render.video_output.encoder is required")
    if not str(settings.get("pixel_format", "")):
        raise ValueError("render.video_output.pixel_format is required")
    rate = settings.get("rate_control")
    if not isinstance(rate, dict):
        raise TypeError("render.video_output.rate_control must be a mapping")
    _reject_unknown_keys(
        rate,
        {"mode", "quality", "bitrate", "max_bitrate", "buffer_size"},
        "render.video_output.rate_control",
    )
    mode = str(rate.get("mode", ""))
    if mode in {"crf", "cq"}:
        if set(rate) != {"mode", "quality"}:
            raise ValueError(f"{mode} rate control requires exactly mode and quality")
        quality = int(rate.get("quality", -1))
        if not 0 <= quality <= 51:
            raise ValueError("quality rate control requires quality in [0, 51]")
        if "bitrate" in rate or "max_bitrate" in rate:
            raise ValueError("quality rate control cannot also specify bitrate")
    elif mode in {"vbr", "cbr"}:
        allowed = {"mode", "bitrate", "max_bitrate"} if mode == "vbr" else {
            "mode",
            "bitrate",
            "buffer_size",
        }
        if "bitrate" not in rate or not set(rate) <= allowed:
            raise ValueError(f"{mode} rate control has invalid settings")
        if not str(rate.get("bitrate", "")):
            raise ValueError(f"{mode} rate control requires bitrate")
        for key in ("bitrate", "max_bitrate", "buffer_size"):
            if key in rate:
                _validate_bitrate(
                    rate[key],
                    f"render.video_output.rate_control.{key}",
                )
    else:
        raise ValueError("rate_control.mode must be crf, cq, vbr, or cbr")
    keyframe_interval = settings.get("keyframe_interval", 0)
    if isinstance(keyframe_interval, bool) or not isinstance(keyframe_interval, int):
        raise TypeError("render.video_output.keyframe_interval must be an integer")
    if not 0 <= keyframe_interval <= _MAX_KEYFRAME_INTERVAL:
        raise ValueError(
            "render.video_output.keyframe_interval must be in "
            f"[0, {_MAX_KEYFRAME_INTERVAL}]"
        )
    audio = settings.get("audio", {})
    if not isinstance(audio, dict):
        raise TypeError("render.video_output.audio must be a mapping")
    _reject_unknown_keys(
        audio,
        {"debug", "redacted", "bitrate"},
        "render.video_output.audio",
    )
    for key in ("debug", "redacted"):
        if audio.get(key, "none") not in {"none", "copy", "aac"}:
            raise ValueError(f"render.video_output.audio.{key} is invalid")
    if "bitrate" in audio:
        _validate_bitrate(
            audio["bitrate"],
            "render.video_output.audio.bitrate",
        )


def validate_redaction(settings: dict[str, Any]) -> None:
    _reject_unknown_keys(
        settings,
        {"method", "box_scale", "gaussian", "mosaic", "feather"},
        "render.redaction",
    )
    method = settings.get("method")
    if method not in {"gaussian", "mosaic"}:
        raise ValueError("render.redaction.method must be gaussian or mosaic")
    box_scale = float(settings.get("box_scale", 0.0))
    if not 0.1 <= box_scale <= 4.0:
        raise ValueError("render.redaction.box_scale must be in [0.1, 4.0]")
    if method == "gaussian":
        gaussian = settings.get("gaussian")
        if not isinstance(gaussian, dict):
            raise TypeError("render.redaction.gaussian must be a mapping")
        _reject_unknown_keys(
            gaussian,
            {"algorithm", "max_side", "kernel_ratio", "min_kernel", "sigma"},
            "render.redaction.gaussian",
        )
        algorithm = gaussian.get("algorithm", "exact")
        if not isinstance(algorithm, str) or algorithm not in {"exact", "pyramid"}:
            raise ValueError(
                "render.redaction.gaussian.algorithm must be exact or pyramid"
            )
        max_side = gaussian.get("max_side", 64)
        if type(max_side) is not int or max_side < 1:
            raise ValueError(
                "render.redaction.gaussian.max_side must be a positive integer"
            )
        if float(gaussian.get("kernel_ratio", 0.0)) <= 0.0:
            raise ValueError("render.redaction.gaussian.kernel_ratio must be positive")
        minimum_kernel = int(gaussian.get("min_kernel", 0))
        if minimum_kernel < 1 or minimum_kernel % 2 == 0:
            raise ValueError("render.redaction.gaussian.min_kernel must be a positive odd integer")
        if float(gaussian.get("sigma", 0.0)) < 0.0:
            raise ValueError("render.redaction.gaussian.sigma cannot be negative")
    if method == "mosaic":
        mosaic = settings.get("mosaic")
        if not isinstance(mosaic, dict):
            raise TypeError("render.redaction.mosaic must be a mapping")
        _reject_unknown_keys(
            mosaic,
            {"block_size_ratio", "min_block_size"},
            "render.redaction.mosaic",
        )
        if not 0.0 < float(mosaic.get("block_size_ratio", 0.0)) <= 1.0:
            raise ValueError("render.redaction.mosaic.block_size_ratio must be in (0, 1]")
        if int(mosaic.get("min_block_size", 0)) < 1:
            raise ValueError("render.redaction.mosaic.min_block_size must be positive")
    feather = settings.get("feather", {"enabled": False})
    if not isinstance(feather, dict) or not isinstance(feather.get("enabled", False), bool):
        raise TypeError("render.redaction.feather.enabled must be boolean")
    _reject_unknown_keys(
        feather,
        {"enabled", "ratio", "min_pixels"},
        "render.redaction.feather",
    )
    if bool(feather.get("enabled", False)):
        if not 0.0 <= float(feather.get("ratio", -1.0)) <= 0.5:
            raise ValueError("render.redaction.feather.ratio must be in [0, 0.5]")
        if int(feather.get("min_pixels", 0)) < 1:
            raise ValueError("render.redaction.feather.min_pixels must be positive")


def load_config(
    path: str | Path,
    *,
    config_overrides: Mapping[str, Any] | None = None,
    config_override_root: str | Path | None = None,
    materialize_models: bool = True,
) -> dict[str, Any]:
    """Load Base plus an optional overlay and validate the effective settings.

    ``materialize_models=False`` is the read-only diagnostics path.  It keeps
    the public ``models.name`` selector intact and never invokes ModelZoo's
    download resolver; normal analysis continues to materialize the manifest
    and task descriptors before any inference Session is constructed.
    """

    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(materialize_models, bool):
        raise TypeError("materialize_models must be boolean")
    if config_overrides is not None and not isinstance(config_overrides, Mapping):
        raise TypeError("config_overrides must be a dotted-path mapping")
    if config_overrides is None and config_override_root is not None:
        raise ValueError("config_override_root requires config_overrides")
    dotted_root = (
        Path(config_override_root).expanduser().resolve()
        if config_override_root is not None
        else Path.cwd().resolve()
    )
    if source == DEFAULT_CONFIG_PATH.resolve():
        config = _load_base_config(
            source,
            dotted_overrides=config_overrides,
            dotted_override_root=dotted_root,
            materialize_models=materialize_models,
        )
    else:
        if (
            not isinstance(raw, dict)
            or type(raw.get("schema_version")) is not int
            or raw["schema_version"] != 1
        ):
            raise ValueError("custom ONNX config must be a schema_version: 1 mapping")
        validate_current_config_contract(raw)
        validate_config_keys(raw, allow_base_config=True)
        if "base_config" in raw:
            base_value = raw["base_config"]
            if not isinstance(base_value, str) or not base_value.strip():
                raise TypeError("base_config must be a non-empty path string")
            base_candidate = Path(base_value).expanduser()
            if not base_candidate.is_absolute():
                base_candidate = source.parent / base_candidate
            base = base_candidate.resolve()
        else:
            base = DEFAULT_CONFIG_PATH.resolve()
        file_overrides = {
            key: value
            for key, value in raw.items()
            if key not in {"base_config", "schema_version"}
        }
        config = _load_base_config(
            base,
            derived_overrides=file_overrides,
            derived_override_root=source.parent,
            dotted_overrides=config_overrides,
            dotted_override_root=dotted_root,
            materialize_models=materialize_models,
        )
    output = config.setdefault("output", {})
    if not isinstance(output, dict):
        raise TypeError("output settings must be a mapping")
    if output.get("artifacts_level", "final") not in {
        "final",
        "audit",
        "debug",
    }:
        raise ValueError("output.artifacts_level must be final, audit, or debug")
    output.setdefault("artifacts_level", "final")
    scan = config.get("scan")
    if not isinstance(scan, dict):
        raise TypeError("scan settings are required")
    session_sharing = scan.get("session_sharing", "single_session_parallel")
    if not isinstance(session_sharing, str):
        raise TypeError("scan.session_sharing must be a string")
    if session_sharing not in {
        "single_session_parallel",
        "single_session_serial",
    }:
        raise ValueError(
            "scan.session_sharing must be single_session_parallel or "
            "single_session_serial"
        )
    scan.setdefault("session_sharing", "single_session_parallel")
    streaming = config.get("streaming")
    if not isinstance(streaming, dict):
        raise TypeError("streaming settings are required")
    if float(streaming.get("max_missed_seconds", 0.0)) <= 0.0:
        raise ValueError("streaming.max_missed_seconds must be positive")
    if float(streaming.get("max_retroactive_seconds", 0.0)) < float(streaming["max_missed_seconds"]):
        raise ValueError("max_retroactive_seconds must cover max_missed_seconds")
    if int(streaming.get("max_corridor_side_pixels", 0)) < 384:
        raise ValueError("streaming.max_corridor_side_pixels must be at least 384")
    streaming.setdefault("recent_frame_cache_frames", None)
    streaming.setdefault("recent_frame_cache_max_bytes", 256 * 1024 * 1024)
    streaming.setdefault("pre_roll_decode_chunk_frames", 32)
    recent_frame_target = streaming["recent_frame_cache_frames"]
    if recent_frame_target is not None:
        if isinstance(recent_frame_target, bool) or not isinstance(
            recent_frame_target,
            int,
        ):
            raise TypeError(
                "streaming.recent_frame_cache_frames must be an integer or null"
            )
        if recent_frame_target < 0:
            raise ValueError("streaming.recent_frame_cache_frames cannot be negative")
    if int(streaming["recent_frame_cache_max_bytes"]) < 0:
        raise ValueError("streaming.recent_frame_cache_max_bytes cannot be negative")
    if int(streaming["pre_roll_decode_chunk_frames"]) < 1:
        raise ValueError("streaming.pre_roll_decode_chunk_frames must be positive")
    flow = config["tracking"]["kalman_optical_flow"]
    if int(flow.get("roi_size", 0)) <= 0 or float(flow.get("roi_expansion", 0.0)) <= 1.0:
        raise ValueError("ROI flow requires roi_size and roi_expansion > 1")
    if int(flow.get("max_source_canvas_side_pixels", 0)) < int(flow["roi_size"]):
        raise ValueError("max_source_canvas_side_pixels must cover roi_size")
    if int(streaming["pre_roll_decode_chunk_frames"]) <= int(flow.get("max_coast_frames", 0)):
        raise ValueError("pre_roll_decode_chunk_frames must exceed max_coast_frames")
    require_cycle = flow.get("require_cycle_consistency_after_coast", False)
    if not isinstance(require_cycle, bool):
        raise TypeError("tracking.kalman_optical_flow.require_cycle_consistency_after_coast must be boolean")
    cycle_minimum_iou = float(flow.get("recovery_cycle_min_iou", 0.10))
    if not 0.0 <= cycle_minimum_iou <= 1.0:
        raise ValueError("recovery_cycle_min_iou must be between 0 and 1")
    endpoint_affine = flow.get("endpoint_affine_repair", {})
    if not isinstance(endpoint_affine, dict) or not isinstance(endpoint_affine.get("enabled", False), bool):
        raise TypeError("endpoint_affine_repair.enabled must be boolean")
    if bool(endpoint_affine.get("enabled", False)) and not 1 <= int(
        endpoint_affine.get("max_frames", 0)
    ) <= int(config["tracking"]["endpoint_extension"]):
        raise ValueError("endpoint_affine_repair.max_frames must be within the short endpoint extension")
    tracking = config["tracking"]
    between_scan_frames = tracking.get("between_scan_frames", "interpolate")
    if not isinstance(between_scan_frames, str):
        raise TypeError("tracking.between_scan_frames must be a string")
    if between_scan_frames not in {"visual", "interpolate"}:
        raise ValueError(
            "tracking.between_scan_frames must be visual or interpolate"
        )
    tracking.setdefault("between_scan_frames", "interpolate")
    association_scan_gap = tracking.get("association_max_scan_gap")
    if (
        isinstance(association_scan_gap, bool)
        or not isinstance(association_scan_gap, int)
        or association_scan_gap < 1
    ):
        raise ValueError("tracking.association_max_scan_gap must be a positive integer")
    association_gap_seconds = tracking.get("association_max_gap_seconds")
    if (
        isinstance(association_gap_seconds, bool)
        or not isinstance(association_gap_seconds, (int, float))
        or float(association_gap_seconds) <= 0.0
    ):
        raise ValueError("tracking.association_max_gap_seconds must be positive")
    strict_geometry_seconds = tracking.get(
        "association_strict_geometry_after_seconds"
    )
    if (
        isinstance(strict_geometry_seconds, bool)
        or not isinstance(strict_geometry_seconds, (int, float))
        or not 0.0 < float(strict_geometry_seconds) <= float(association_gap_seconds)
    ):
        raise ValueError(
            "tracking.association_strict_geometry_after_seconds must be positive "
            "and cannot exceed association_max_gap_seconds"
        )
    if float(tracking.get("long_gap_min_iou", 0.0)) < 0.0:
        raise ValueError("tracking.long_gap_min_iou cannot be negative")
    if float(tracking.get("long_gap_max_center_distance", 0.0)) <= 0.0:
        raise ValueError("tracking.long_gap_max_center_distance must be positive")
    if (
        int(
            tracking.get(
                "reliable_pre_roll_extension",
                tracking.get("reliable_endpoint_extension", 0),
            )
        )
        < 1
    ):
        raise ValueError("tracking.reliable_pre_roll_extension must be positive")
    stitching = tracking.get("fragment_stitching", {})
    if not isinstance(stitching, dict) or not isinstance(stitching.get("enabled", False), bool):
        raise TypeError("tracking.fragment_stitching.enabled must be boolean")
    if bool(stitching.get("enabled", False)):
        if not isinstance(
            stitching.get("resolve_duplicate_candidates_before_stabilization", True),
            bool,
        ):
            raise TypeError(
                "tracking.fragment_stitching."
                "resolve_duplicate_candidates_before_stabilization must be boolean"
            )
        if int(stitching.get("min_overlap_frames", 0)) < 2:
            raise ValueError("fragment_stitching.min_overlap_frames must be at least 2")
        if int(stitching.get("min_agreement_frames", 0)) < 2:
            raise ValueError("fragment_stitching.min_agreement_frames must be at least 2")
        if int(stitching.get("max_interval_gap_frames", -1)) < 0:
            raise ValueError("fragment_stitching.max_interval_gap_frames cannot be negative")
        for key in (
            "min_local_confidence",
            "min_local_iou",
            "min_agreement_fraction",
        ):
            value = float(stitching.get(key, -1.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"fragment_stitching.{key} must be between 0 and 1")
    refinement = config["revalidation"].get("geometry_refinement")
    if not isinstance(refinement, dict):
        raise TypeError("revalidation.geometry_refinement settings are required")
    if float(refinement.get("min_local_confidence", -1.0)) < 0.0:
        raise ValueError("geometry_refinement.min_local_confidence cannot be negative")
    if float(refinement.get("max_area_ratio", 0.0)) < 1.0:
        raise ValueError("geometry_refinement.max_area_ratio must be at least 1")
    if float(refinement.get("max_center_distance", 0.0)) <= 0.0:
        raise ValueError("geometry_refinement.max_center_distance must be positive")
    measurement = refinement.get("measurement_filter", {})
    if not isinstance(measurement, dict) or not isinstance(measurement.get("enabled", False), bool):
        raise TypeError("geometry_refinement.measurement_filter.enabled must be boolean")
    if measurement.get("scope", "all") not in {"all", "tracking_only"}:
        raise ValueError("geometry_refinement.measurement_filter.scope must be all or tracking_only")
    confidence_low = float(measurement.get("confidence_low", -1.0))
    confidence_high = float(measurement.get("confidence_high", -1.0))
    if not 0.0 <= confidence_low < confidence_high <= 1.0:
        raise ValueError("measurement_filter confidence range must increase within [0, 1]")
    for key in (
        "center_gain_low",
        "center_gain_high",
        "recovery_center_gain",
        "size_gain_low",
        "size_gain_high",
    ):
        value = float(measurement.get(key, -1.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"measurement_filter.{key} must be between 0 and 1")
    if float(measurement.get("center_gain_high", 0.0)) < float(measurement.get("center_gain_low", 0.0)):
        raise ValueError("measurement_filter center gains must be nondecreasing")
    if float(measurement.get("size_gain_high", 0.0)) < float(measurement.get("size_gain_low", 0.0)):
        raise ValueError("measurement_filter size gains must be nondecreasing")
    if float(measurement.get("max_center_step", 0.0)) <= 0.0:
        raise ValueError("measurement_filter.max_center_step must be positive")
    if float(measurement.get("max_size_ratio_per_update", 0.0)) < 1.0:
        raise ValueError("measurement_filter.max_size_ratio_per_update must be at least 1")
    recovery = refinement.get("anchor_recovery", {})
    if not isinstance(recovery, dict) or not isinstance(recovery.get("enabled", False), bool):
        raise TypeError("geometry_refinement.anchor_recovery.enabled must be boolean")
    if recovery.get("candidate_selection") not in {
        "confidence",
        "target_geometry",
    }:
        raise ValueError("anchor_recovery.candidate_selection must be confidence or target_geometry")
    for key in (
        "min_local_confidence",
        "min_iou",
        "min_containment",
        "max_center_distance",
    ):
        value = float(recovery.get(key, -1.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"anchor_recovery.{key} must be between 0 and 1")
    stabilization = config["render"].get("box_stabilization", {})
    if not isinstance(stabilization, dict) or not isinstance(stabilization.get("enabled", False), bool):
        raise TypeError("render.box_stabilization.enabled must be boolean")
    window = int(stabilization.get("median_window", 0))
    if window < 3 or window % 2 == 0:
        raise ValueError("box_stabilization.median_window must be odd and at least 3")
    if int(stabilization.get("min_segment_frames", 0)) < window:
        raise ValueError("box_stabilization.min_segment_frames must cover median_window")
    if int(stabilization.get("reset_gap_frames", -1)) < 1:
        raise ValueError("box_stabilization.reset_gap_frames must be positive")
    for key in ("center_alpha", "size_alpha"):
        value = float(stabilization.get(key, -1.0))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"box_stabilization.{key} must be in (0, 1]")
    if float(stabilization.get("max_center_innovation", 0.0)) <= 0.0:
        raise ValueError("box_stabilization.max_center_innovation must be positive")
    if float(stabilization.get("max_size_ratio", 0.0)) < 1.0:
        raise ValueError("box_stabilization.max_size_ratio must be at least 1")
    anchor_strength = float(stabilization.get("detector_anchor_strength", 0.0))
    if not 0.0 <= anchor_strength <= 1.0:
        raise ValueError("box_stabilization.detector_anchor_strength must be between 0 and 1")
    minimum_anchor_gap = int(stabilization.get("detector_anchor_min_gap_frames", 1))
    maximum_anchor_gap = int(stabilization.get("detector_anchor_max_gap_frames", 1))
    if minimum_anchor_gap < 1 or maximum_anchor_gap < minimum_anchor_gap:
        raise ValueError("box_stabilization detector anchor gap range must be positive and increasing")
    change_point = stabilization.get("change_point_reset", {})
    if not isinstance(change_point, dict) or not isinstance(change_point.get("enabled", False), bool):
        raise TypeError("box_stabilization.change_point_reset.enabled must be boolean")
    if bool(change_point.get("enabled", False)):
        change_window = int(change_point.get("window_frames", 0))
        minimum_detector_frames = int(change_point.get("min_detector_frames_per_side", 0))
        if change_window < 2:
            raise ValueError("box_stabilization.change_point_reset.window_frames must be at least 2")
        if not 1 <= minimum_detector_frames <= change_window:
            raise ValueError("box_stabilization.change_point_reset detector frames must fit its window")
        for key in (
            "min_instantaneous_scale_ratio",
            "min_persistent_scale_ratio",
            "max_within_regime_ratio",
        ):
            if float(change_point.get(key, 0.0)) <= 1.0:
                raise ValueError(f"box_stabilization.change_point_reset.{key} must exceed 1")
        if float(change_point.get("min_scene_mean_absdiff", -1.0)) < 0.0:
            raise ValueError(
                "box_stabilization.change_point_reset.min_scene_mean_absdiff must be non-negative"
            )
    render = config["render"]
    redaction = render.get("redaction")
    if not isinstance(redaction, dict):
        raise TypeError("render.redaction settings are required")
    validate_redaction(redaction)
    video_output = render.get("video_output")
    if not isinstance(video_output, dict):
        raise TypeError("render.video_output settings are required")
    validate_video_output(video_output)
    return config


__all__ = [
    "load_config",
    "validate_redaction",
    "validate_video_output",
]
