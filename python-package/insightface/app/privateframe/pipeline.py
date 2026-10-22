"""Artifact-first application entry points for analysis and video rendering."""

from __future__ import annotations

import gc
import json
import math
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .artifact_render import RenderTarget, render_artifacts, _identity_should_blur
from .artifacts import (
    git_version,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from .base_config import apply_config_overrides, validate_config_keys
from .config import load_config, validate_redaction, validate_video_output
from .model_catalog import (
    RECOGNITION_TASK,
    VERIFICATION_TASK,
    verify_model_file,
)
from .models import (
    active_face_detector,
    active_face_verifier,
    packaged_face_recognizer,
)
from .streaming import run_stream
from .recognition import resolve_unknown_action
from .video import paths_are_distinct

RESULT_FILENAME = "result.privateframe.json"
DEVELOPMENT_REPORT_FILENAME = "result.privateframe.dev.json"
ARTIFACT_LEVELS = {"final", "audit", "debug"}
_GENERATED_FILENAMES = {
    RESULT_FILENAME,
    DEVELOPMENT_REPORT_FILENAME,
    "detections.streaming-onnx.jsonl",
    "tracking.streaming-onnx.jsonl",
    "bidirectional-fusion.streaming-onnx.jsonl",
    "revalidation.streaming-onnx.jsonl",
    "observations.streaming-onnx.jsonl",
    "tracks.streaming-onnx.json",
    "effective-config.streaming-onnx.json",
    "manifest.streaming-onnx.json",
    "summary.streaming-onnx.json",
    "render-summary.streaming-onnx.json",
}
_RUNTIME_CACHE_FILENAMES = {
    "encoded-packets.sqlite",
    "encoded-packets.sqlite-wal",
    "encoded-packets.sqlite-shm",
}
_RESERVED_WORK_FILENAMES = _GENERATED_FILENAMES | _RUNTIME_CACHE_FILENAMES


def _raise_if_cancelled(is_cancelled: Callable[[], bool] | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise InterruptedError("PrivateFrame operation was cancelled")


def _merge(
    target: dict[str, Any],
    update: dict[str, Any],
    path: tuple[str, ...] = (),
) -> None:
    for key, value in update.items():
        child_path = (*path, key)
        if child_path[-1] == "rate_control":
            target[key] = value
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value, child_path)
        else:
            target[key] = value


def _reject_reserved_work_path(
    path: Path,
    workdir: Path,
    *,
    label: str,
    allowed: set[Path] | None = None,
) -> None:
    if path.parent != workdir or path.name not in _RESERVED_WORK_FILENAMES:
        return
    if allowed is not None and path in allowed:
        return
    raise ValueError(f"{label} path conflicts with a reserved work artifact: {path}")


def _reject_reserved_result_name(path: Path) -> None:
    if path.name in _RESERVED_WORK_FILENAMES and path.name != RESULT_FILENAME:
        raise ValueError(f"result JSON uses a reserved artifact name: {path.name}")


def _model_fingerprints(config: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    detector_id, detector = active_face_detector(config)
    models = {
        detector_id: detector,
        VERIFICATION_TASK: active_face_verifier(config),
    }
    if str(config.get("recognition", {}).get("mode", "all")) != "all":
        recognizer = packaged_face_recognizer(config)
        models[RECOGNITION_TASK] = recognizer
    for model_id, model in sorted(models.items()):
        if "path" not in model:
            raise TypeError(f"active model {model_id} has no path mapping")
        path = verify_model_file(model)
        declared_sha256 = model.get("sha256")
        fingerprint = {
            "bytes": path.stat().st_size,
            "file": str(model["file"]),
            "preprocessing": (
                dict(model["preprocessing"])
                if isinstance(model["preprocessing"], Mapping)
                else str(model["preprocessing"])
            ),
        }
        if declared_sha256 is not None:
            fingerprint["sha256"] = declared_sha256
        for key in (
            "preprocessing_version",
            "input_size",
            "embedding_dimension",
            "expansion",
        ):
            if key in model:
                value = model[key]
                fingerprint[key] = list(value) if isinstance(value, tuple) else value
        values[str(model_id)] = fingerprint
    return values


def _model_package_fingerprint(
    config: dict[str, Any],
) -> dict[str, Any]:
    models = config["models"]
    manifest_path = Path(str(models["manifest_path"]))
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Model package manifest does not exist: {manifest_path}"
        )
    return {
        "name": str(models["name"]),
        "manifest": {
            "file": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
        },
    }


def _result_path(workdir: Path | None, result_path: str | Path | None) -> Path:
    if result_path is not None:
        return Path(result_path).expanduser().resolve()
    if workdir is None:
        raise ValueError("--result or --workdir is required")
    return (workdir / RESULT_FILENAME).resolve()


def _development_report_path(result_path: Path) -> Path:
    """Return the developer-report path paired with a user result document."""

    return result_path.with_name(f"{result_path.stem}.dev.json")


def _owned_development_report_exists(
    path: Path, result_path: Path, *, reject_unowned: bool
) -> bool:
    """Recognize an old report, rejecting other files only when replacing it."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        if not reject_unowned:
            return False
        raise FileExistsError(
            f"refusing to replace unrecognized development report: {path}"
        ) from error
    if (
        not isinstance(value, dict)
        or value.get("format") != "privateframe-development-report"
        or value.get("result_file") != result_path.name
    ):
        if not reject_unowned:
            return False
        raise FileExistsError(
            f"refusing to replace unrecognized development report: {path}"
        )
    return True


def _public_observation_source(item: Mapping[str, Any]) -> str:
    """Collapse internal geometry mechanisms into stable user-facing terms."""

    raw_source = str(item.get("source", ""))
    if raw_source == "detector":
        return "detector"
    if item.get("endpoint_repair") is not None or item.get(
        "endpoint_repair_reason"
    ) is not None:
        return "repaired"
    if (
        item.get("geometry_source") == "detector_anchor_interpolation"
        or item.get("local_review_reason") == "detector_anchor_interpolation"
    ):
        return "interpolated"
    return "tracked"


def _export_observations(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Export only the stable geometry and privacy semantics used by renderers."""

    output: list[dict[str, Any]] = []
    for item in values:
        exported = {
            "frame_idx": int(item["frame_idx"]),
            "track_id": str(item["track_id"]),
            "box": list(item["box"]),
            "source": _public_observation_source(item),
        }
        if bool(item.get("reduced_assurance", False)):
            exported["reduced_assurance"] = True
        if bool(item.get("identity_unconfirmed", False)) or (
            item.get("endpoint_repair_reason")
            == "interpolate_unanchored_endpoint"
        ):
            # Geometry review cannot establish continuity of reference matching.
            exported["identity_unconfirmed"] = True
        output.append(exported)
    return output


def _export_recognition(value: Any) -> dict[str, Any]:
    """Export photo acceptance and reference-match decisions for later renders."""

    if not isinstance(value, dict) or value.get("enabled") is not True:
        return {"enabled": False}
    raw_tracks = value.get("tracks", {})
    tracks: dict[str, dict[str, Any]] = {}
    if isinstance(raw_tracks, dict):
        for track_id, record in raw_tracks.items():
            if not isinstance(record, dict):
                continue
            tracks[str(track_id)] = {
                "status": str(record.get("status", "UNKNOWN")),
                "matched_reference_files": list(record.get("matched_reference_files", [])),
                "similarity": record.get("similarity"),
                "reason": str(record.get("reason", "")),
            }
    return {
        "enabled": True,
        "references": deepcopy(value.get("references", {})),
        "tracks": tracks,
    }


def validate_result_document(value: Any) -> dict[str, Any]:
    """Validate the stable structure required to render a result document."""

    if not isinstance(value, dict):
        raise TypeError("PrivateFrame result root must be an object")
    if value.get("format") != "privateframe-result":
        raise ValueError("PrivateFrame result format must be privateframe-result")
    schema_version = value.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise TypeError("PrivateFrame result schema_version must be an integer")
    if schema_version != 1:
        raise ValueError("PrivateFrame result schema_version must be 1")

    source_video = value.get("source_video")
    if not isinstance(source_video, dict):
        raise TypeError("PrivateFrame result source_video must be an object")
    file_name = source_video.get("file_name")
    if file_name is not None and (
        not isinstance(file_name, str) or not file_name.strip()
    ):
        raise TypeError("PrivateFrame result source_video.file_name must be a string")
    for field, expected in (
        ("coordinate_system", "pixel_xyxy"),
        ("timing_contract", "cfr_frame_index"),
        ("frame_index_origin", 0),
    ):
        if field in source_video and source_video[field] != expected:
            raise ValueError(
                f"PrivateFrame result source_video.{field} must be {expected!r}"
            )
    metadata = source_video.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("PrivateFrame result source_video.metadata must be an object")
    for field in ("width", "height", "frame_count"):
        item = metadata.get(field)
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(
                f"PrivateFrame result source_video.metadata.{field} "
                "must be a positive integer"
            )
        if item <= 0:
            raise ValueError(
                f"PrivateFrame result source_video.metadata.{field} must be positive"
            )
    fps = metadata.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
    ):
        raise TypeError(
            "PrivateFrame result source_video.metadata.fps must be a finite number"
        )
    if float(fps) <= 0.0:
        raise ValueError(
            "PrivateFrame result source_video.metadata.fps must be positive"
        )
    duration = metadata.get("duration")
    if duration is not None and (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) <= 0.0
    ):
        raise ValueError(
            "PrivateFrame result source_video.metadata.duration must be "
            "positive and finite"
        )

    if not isinstance(value.get("render_defaults"), dict):
        raise TypeError("PrivateFrame result render_defaults must be an object")
    if not isinstance(value.get("recognition"), dict):
        raise TypeError("PrivateFrame result recognition must be an object")
    recognition_enabled = value["recognition"].get("enabled")
    if not isinstance(recognition_enabled, bool):
        raise TypeError("PrivateFrame result recognition.enabled must be boolean")

    observations = value.get("observations")
    if not isinstance(observations, list):
        raise TypeError("PrivateFrame result observations must be an array")
    for index, observation in enumerate(observations):
        prefix = f"PrivateFrame result observations[{index}]"
        if not isinstance(observation, dict):
            raise TypeError(f"{prefix} must be an object")
        frame_idx = observation.get("frame_idx")
        if (
            isinstance(frame_idx, bool)
            or not isinstance(frame_idx, int)
            or frame_idx < 0
        ):
            raise ValueError(f"{prefix}.frame_idx must be a non-negative integer")
        if frame_idx >= int(metadata["frame_count"]):
            raise ValueError(f"{prefix}.frame_idx exceeds the source frame count")
        track_id = observation.get("track_id")
        if not isinstance(track_id, str) or not track_id:
            raise TypeError(f"{prefix}.track_id must be a non-empty string")
        source = observation.get("source")
        if not isinstance(source, str) or not source:
            raise TypeError(f"{prefix}.source must be a non-empty string")
        for flag in ("reduced_assurance", "identity_unconfirmed"):
            if flag in observation and not isinstance(observation[flag], bool):
                raise TypeError(f"{prefix}.{flag} must be boolean")
        box = observation.get("box")
        if not isinstance(box, list) or len(box) != 4:
            raise TypeError(f"{prefix}.box must be an array of four finite numbers")
        if any(
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or (isinstance(coordinate, float) and not math.isfinite(coordinate))
            for coordinate in box
        ):
            raise ValueError(f"{prefix}.box must contain four finite numbers")
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"{prefix}.box must satisfy x2 > x1 and y2 > y1")
    return value


def _load_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return validate_result_document(value)


def _render_settings(
    result: dict[str, Any],
    render_config: str | Path | None,
    config_overrides: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    settings = deepcopy(result["render_defaults"])
    if render_config is not None:
        source = Path(render_config).expanduser().resolve()
        override = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(override, dict):
            raise TypeError("render config must be a mapping")
        if "render" in override:
            if set(override) != {"render"} or not isinstance(override["render"], dict):
                raise ValueError(
                    "a nested render config must contain only the render mapping"
                )
            override = override["render"]
        unknown = set(override) - set(settings)
        if unknown:
            raise ValueError("unknown render settings: " + ", ".join(sorted(unknown)))
        validate_config_keys(
            {
                "render": {
                    key: value
                    for key, value in override.items()
                    if key != "recognition_policy"
                }
            }
        )
        _merge(settings, override)
    if config_overrides is not None and not isinstance(config_overrides, Mapping):
        raise TypeError("config_overrides must be a dotted-path mapping")
    if config_overrides:
        if any(not isinstance(path, str) for path in config_overrides):
            raise TypeError("configuration override paths must be strings")
        invalid = sorted(
            path for path in config_overrides if not path.startswith("render.")
        )
        if invalid:
            raise ValueError(
                "render only accepts render.* configuration overrides: " + invalid[0]
            )
        # Command-line/Python dotted overrides are the final configuration
        # layer, above both the analyzed defaults and --render-config YAML.
        wrapped = {"render": settings}
        apply_config_overrides(wrapped, config_overrides)
        settings = wrapped["render"]
        validate_config_keys(
            {
                "render": {
                    key: value
                    for key, value in settings.items()
                    if key != "recognition_policy"
                }
            }
        )
    validate_redaction(settings["redaction"])
    validate_video_output(settings["video_output"])
    _validate_recognition_render_policy(settings, result.get("recognition"))
    if settings["recognition_policy"]["mode"] != "all":
        for observation in result.get("observations", []):
            _identity_should_blur(observation, settings["recognition_policy"], result["recognition"])
    return settings, sha256_json(settings)


def _validate_recognition_render_policy(
    settings: dict[str, Any], recognition: Any
) -> None:
    policy = settings.get("recognition_policy")
    if not isinstance(policy, dict):
        raise TypeError(
            "render.recognition_policy must be a mapping in the analysis result"
        )
    unknown = set(policy) - {"mode", "unknown_action"}
    if unknown:
        raise ValueError(
            "unknown render recognition policy settings: " + ", ".join(sorted(unknown))
        )
    mode = str(policy.get("mode", "all"))
    if mode not in {"all", "blur_only", "exempt"}:
        raise ValueError(
            "render.recognition_policy.mode must be all, blur_only, or exempt"
        )
    policy["mode"] = mode
    policy["unknown_action"] = resolve_unknown_action(mode, policy.get("unknown_action", "auto"))
    if mode == "all":
        return
    if not isinstance(recognition, dict) or recognition.get("enabled") is not True:
        raise ValueError(
            "selective rendering requires a result produced with recognition enabled"
        )
    references = recognition.get("references")
    files = references.get("files") if isinstance(references, dict) else None
    if not isinstance(files, list) or not files:
        raise ValueError("selective rendering requires usable reference photos in the analysis result")
    if any(not isinstance(item, dict) or not isinstance(item.get("file"), str) or not item["file"] for item in files):
        raise ValueError("recognition reference photo records must name each accepted file")
    tracks = recognition.get("tracks")
    if not isinstance(tracks, dict):
        raise ValueError("selective rendering requires reference-match track records")
    for track_id in tracks:
        _identity_should_blur({"track_id": track_id}, policy, recognition)


def _remove_runtime_cache(workdir: Path) -> None:
    for filename in _RUNTIME_CACHE_FILENAMES:
        path = workdir / filename
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _remove_unrequested_artifacts(workdir: Path, retained: set[str]) -> None:
    for filename in _GENERATED_FILENAMES - retained:
        path = workdir / filename
        if path.exists():
            path.unlink()


def _analyze_streaming_pipeline_impl(
    *,
    config_path: str | Path,
    input_path: str | Path,
    workdir: str | Path,
    result_path: str | Path | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    config_override_root: str | Path | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    _raise_if_cancelled(is_cancelled)
    source = Path(input_path).expanduser().resolve()
    work = Path(workdir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    destination = _result_path(work, result_path)
    if not paths_are_distinct([source, destination]):
        raise ValueError("input video and result JSON paths must be distinct")
    _reject_reserved_result_name(destination)
    _reject_reserved_work_path(source, work, label="input video")
    _reject_reserved_work_path(
        destination,
        work,
        label="result JSON",
        allowed={work / RESULT_FILENAME},
    )
    config = load_config(
        config_path,
        config_overrides=config_overrides,
        config_override_root=config_override_root,
    )
    _raise_if_cancelled(is_cancelled)
    artifacts_level = str(config.get("output", {}).get("artifacts_level", "final"))
    if artifacts_level not in ARTIFACT_LEVELS:
        raise ValueError("artifacts_level must be final, audit, or debug")
    development_destination = _development_report_path(destination)
    development_enabled = artifacts_level in {"audit", "debug"}
    # Fingerprints and repository state are diagnostic data. Avoid their I/O
    # and serialization entirely for the ordinary user-facing result.
    model_fingerprints = _model_fingerprints(config) if development_enabled else None
    model_package = (
        _model_package_fingerprint(config) if development_enabled else None
    )
    _raise_if_cancelled(is_cancelled)
    result = run_stream(
        source,
        work,
        config,
        progress=progress,
        is_cancelled=is_cancelled,
    )
    _raise_if_cancelled(is_cancelled)
    # ``run_stream`` constructs the detector, recognizer and references before
    # entering StreamingEngine.run(). Measure the whole analysis phase here so
    # selective setup cannot be mislabeled as artifact-writing time.
    analysis_seconds = time.perf_counter() - started
    artifact_started = time.perf_counter()
    if development_enabled:
        timestamps = {
            int(item["frame_idx"]): float(item["time_seconds"])
            for item in result["scan"]["frames"]
        }
        for values in (
            result["scan"]["detections"],
            result["tracking"]["observations"],
            result["review"]["evidence"],
            result["review"]["observations"],
        ):
            for item in values:
                item.setdefault("time_seconds", timestamps[int(item["frame_idx"])])
    observations = _export_observations(result["review"]["observations"])
    scene_cut_candidates = sum(
        bool(item.get("scene_cut_candidate", False))
        for item in result["scan"]["frames"]
    )
    scene_cut_flow_confirmed = sum(
        bool(item.get("scene_cut_flow_confirmed", False))
        for item in result["scan"]["frames"]
    )
    scene_cut_appearance_confirmed = sum(
        bool(item.get("scene_cut_appearance_confirmed", False))
        for item in result["scan"]["frames"]
    )
    scene_cut_confirmed = sum(
        bool(item.get("scene_cut_confirmed", False))
        for item in result["scan"]["frames"]
    )
    scene_cut_flash_suppressed = sum(
        bool(item.get("scene_cut_flash_suppressed", False))
        for item in result["scan"]["frames"]
    )
    scene_cuts = sum(
        bool(item.get("scene_cut_from_previous", False))
        for item in result["scan"]["frames"]
    )
    raw_metadata = result["scan"]["metadata"]
    source_document = {
        "file_name": source.name,
        "metadata": {
            "width": int(raw_metadata["width"]),
            "height": int(raw_metadata["height"]),
            "fps": float(raw_metadata["fps"]),
            "frame_count": int(raw_metadata["frame_count"]),
            "duration": float(raw_metadata["duration"]),
        },
        "timing_contract": "cfr_frame_index",
        "coordinate_system": "pixel_xyxy",
        "frame_index_origin": 0,
    }
    analysis_statistics = {
        "frame_count": int(result["scan"]["frame_count"]),
        "detections": len(result["scan"]["detections"]),
        "tracks": len(result["tracks"]),
        "accepted_tracks": int(result["review"]["accepted_tracks"]),
        "observations": len(observations),
        "reverse_jobs": int(result["reverse_jobs"]),
        "reverse_frames": int(result["reverse_frames"]),
        "long_gap_reanchors": int(result["long_gap_reanchors"]),
        "discarded_unanchored_tail_frames": int(
            result["discarded_unanchored_tail_frames"]
        ),
        "endpoint_affine_jobs": int(result["endpoint_affine_jobs"]),
        "endpoint_affine_frames": int(result["endpoint_affine_frames"]),
        "endpoint_affine_published_frames": int(
            result["endpoint_affine_published_frames"]
        ),
        "endpoint_verifier_checkpoints": int(
            result.get("endpoint_verifier_checkpoints", 0)
        ),
        "endpoint_verifier_refinement_frames": int(
            result.get("endpoint_verifier_refinement_frames", 0)
        ),
        "endpoint_local_review_frames": int(
            result.get("endpoint_local_review_frames", 0)
        ),
        "interpolate_endpoint_jobs": int(result["interpolate_endpoint_jobs"]),
        "interpolate_endpoint_frames": int(result["interpolate_endpoint_frames"]),
        "interpolate_endpoint_published_frames": int(
            result["interpolate_endpoint_published_frames"]
        ),
        "interpolate_endpoint_seconds": float(result["interpolate_endpoint_seconds"]),
        "interpolate_endpoint_reason_counts": dict(
            result["interpolate_endpoint_reason_counts"]
        ),
        "fragment_stitches": int(result["fragment_stitches"]),
        "scene_cut_candidates": scene_cut_candidates,
        "scene_cut_flow_confirmed": scene_cut_flow_confirmed,
        "scene_cut_appearance_confirmed": scene_cut_appearance_confirmed,
        "scene_cut_confirmed": scene_cut_confirmed,
        "scene_cut_flash_suppressed": scene_cut_flash_suppressed,
        "scene_cuts": scene_cuts,
        "detector_analyzed_frames": int(result["detector_sampling"]["analyzed_frames"]),
        "detector_regular_scan_frames": int(
            result["detector_sampling"]["regular_scan_frames"]
        ),
        "detector_forced_scan_frames": int(
            result["detector_sampling"]["forced_scan_frames"]
        ),
        "detector_skipped_scan_frames": int(
            result["detector_sampling"]["skipped_scan_frames"]
        ),
        "local_review_attempts": int(result["local_review_sampling"]["attempts"]),
        "local_review_sampled_out": int(result["local_review_sampling"]["sampled_out"]),
        "local_review_forced_attempts": int(
            result["local_review_sampling"]["forced_attempts"]
        ),
        "verifier_review_calls": int(result["local_review_sampling"]["verifier_calls"]),
        "verifier_review_cache_hits": int(
            result["local_review_sampling"]["verifier_cache_hits"]
        ),
    }
    analysis_statistics.update(
        {
            "bidirectional_gap_jobs": int(result["bidirectional_gap_jobs"]),
            "bidirectional_gap_frames": int(result["bidirectional_gap_frames"]),
            "bidirectional_accepted_frames": int(
                result["bidirectional_accepted_frames"]
            ),
            "bidirectional_rejected_frames": int(
                result["bidirectional_rejected_frames"]
            ),
            "bidirectional_review_resolutions": int(
                result["bidirectional_review_resolutions"]
            ),
            "bidirectional_skipped_jobs": int(result["bidirectional_skipped_jobs"]),
            "bidirectional_association_attempts": int(
                result["bidirectional_association_attempts"]
            ),
            "bidirectional_association_rescues": int(
                result["bidirectional_association_rescues"]
            ),
        }
    )
    recognition_diagnostics = result.get(
        "recognition", {"enabled": False, "reason": "policy_all"}
    )
    recognition = _export_recognition(recognition_diagnostics)
    render_defaults = deepcopy(config["render"])
    render_defaults.pop("box_stabilization", None)
    render_defaults.pop("debug_line_thickness", None)
    video_output_defaults = render_defaults.get("video_output")
    if isinstance(video_output_defaults, dict):
        audio_defaults = video_output_defaults.get("audio")
        if isinstance(audio_defaults, dict):
            audio_defaults.pop("debug", None)
    recognition_settings = config.get(
        "recognition", {"mode": "all", "unknown_action": "auto"}
    )
    recognition_mode = str(recognition_settings.get("mode", "all"))
    render_defaults["recognition_policy"] = {
        "mode": recognition_mode,
        "unknown_action": resolve_unknown_action(
            recognition_mode, recognition_settings.get("unknown_action", "auto")
        ),
    }
    final_result = {
        "format": "privateframe-result",
        "schema_version": 1,
        "source_video": source_document,
        "render_defaults": render_defaults,
        "recognition": recognition,
        "observations": observations,
    }
    summary = {
        "backend": "onnxruntime-streaming-gop-roi",
        "phase": "analysis",
        "provider": config["runtime"]["resolved_provider"],
        "input": str(source),
        "workdir": str(work),
        "result": str(destination),
        "artifacts_level": artifacts_level,
        "frame_count": int(result["scan"]["frame_count"]),
        "detections": len(result["scan"]["detections"]),
        "tracks": len(result["tracks"]),
        "accepted_tracks": int(result["review"]["accepted_tracks"]),
        "observations": len(result["review"]["observations"]),
        "recognition": recognition_diagnostics,
        "reverse_jobs": int(result["reverse_jobs"]),
        "reverse_frames": int(result["reverse_frames"]),
        "long_gap_reanchors": int(result["long_gap_reanchors"]),
        "discarded_unanchored_tail_frames": int(
            result["discarded_unanchored_tail_frames"]
        ),
        "endpoint_affine_jobs": int(result["endpoint_affine_jobs"]),
        "endpoint_affine_frames": int(result["endpoint_affine_frames"]),
        "endpoint_affine_published_frames": int(
            result["endpoint_affine_published_frames"]
        ),
        "endpoint_verifier_checkpoints": int(
            result.get("endpoint_verifier_checkpoints", 0)
        ),
        "endpoint_verifier_refinement_frames": int(
            result.get("endpoint_verifier_refinement_frames", 0)
        ),
        "endpoint_local_review_frames": int(
            result.get("endpoint_local_review_frames", 0)
        ),
        "interpolate_endpoint_jobs": int(result["interpolate_endpoint_jobs"]),
        "interpolate_endpoint_frames": int(result["interpolate_endpoint_frames"]),
        "interpolate_endpoint_published_frames": int(
            result["interpolate_endpoint_published_frames"]
        ),
        "interpolate_endpoint_seconds": float(result["interpolate_endpoint_seconds"]),
        "interpolate_endpoint_reason_counts": dict(
            result["interpolate_endpoint_reason_counts"]
        ),
        "fragment_stitches": int(result["fragment_stitches"]),
        "scene_cut_candidates": scene_cut_candidates,
        "scene_cut_flow_confirmed": scene_cut_flow_confirmed,
        "scene_cut_appearance_confirmed": scene_cut_appearance_confirmed,
        "scene_cut_confirmed": scene_cut_confirmed,
        "scene_cut_flash_suppressed": scene_cut_flash_suppressed,
        "scene_cuts": scene_cuts,
        "detector_sampling": result["detector_sampling"],
        "local_review_sampling": result["local_review_sampling"],
        "endpoint_conflicts": result.get("endpoint_conflicts", {"enabled": False}),
        "cache": result["cache"],
        "timings": {
            "analysis_seconds": analysis_seconds,
            "artifact_seconds": 0.0,
            "total_seconds": analysis_seconds,
        },
        "profile": {
            "scene_cut_detector": "adaptive",
            "max_analysis_fps": float(result["detector_sampling"]["max_analysis_fps"]),
            "detector_frame_stride": int(
                result["detector_sampling"]["effective_frame_stride"]
            ),
            "max_missed_seconds": float(config["streaming"]["max_missed_seconds"]),
            "max_retroactive_seconds": float(
                config["streaming"]["max_retroactive_seconds"]
            ),
            "pre_roll_decode_chunk_frames": int(
                config["streaming"]["pre_roll_decode_chunk_frames"]
            ),
            "recent_frame_cache_frames": int(
                result["cache"]["recent_frame_target_frames"]
            ),
            "roi_size": int(config["tracking"]["kalman_optical_flow"]["roi_size"]),
            "roi_expansion": float(
                config["tracking"]["kalman_optical_flow"]["roi_expansion"]
            ),
            "bidirectional_fusion_mode": "symmetric_local_soft",
            "measurement_filter": bool(
                config["revalidation"]["geometry_refinement"]["measurement_filter"][
                    "enabled"
                ]
            ),
            "box_stabilization": bool(config["render"]["box_stabilization"]["enabled"]),
        },
    }
    summary.update(
        {
            "bidirectional_gap_jobs": int(result["bidirectional_gap_jobs"]),
            "bidirectional_gap_frames": int(result["bidirectional_gap_frames"]),
            "bidirectional_accepted_frames": int(
                result["bidirectional_accepted_frames"]
            ),
            "bidirectional_rejected_frames": int(
                result["bidirectional_rejected_frames"]
            ),
            "bidirectional_review_resolutions": int(
                result["bidirectional_review_resolutions"]
            ),
            "bidirectional_skipped_jobs": int(result["bidirectional_skipped_jobs"]),
            "bidirectional_association_attempts": int(
                result["bidirectional_association_attempts"]
            ),
            "bidirectional_association_rescues": int(
                result["bidirectional_association_rescues"]
            ),
        }
    )
    # The paired report has its own ownership-aware cleanup below. Keep it out
    # of generic workdir cleanup, including an unrelated file in final mode.
    retained = (
        {destination.name, development_destination.name}
        if destination.parent == work
        else set()
    )
    if development_enabled:
        write_json(work / "tracks.streaming-onnx.json", result["tracks"])
        write_json(work / "effective-config.streaming-onnx.json", config)
        retained.update(
            {
                "tracks.streaming-onnx.json",
                "effective-config.streaming-onnx.json",
                "summary.streaming-onnx.json",
                "manifest.streaming-onnx.json",
            }
        )
        write_jsonl(
            work / "bidirectional-fusion.streaming-onnx.jsonl",
            result["bidirectional_audits"],
        )
        retained.add("bidirectional-fusion.streaming-onnx.jsonl")
    if artifacts_level == "debug":
        write_jsonl(
            work / "detections.streaming-onnx.jsonl",
            result["scan"]["detections"],
        )
        write_jsonl(
            work / "tracking.streaming-onnx.jsonl",
            result["tracking"]["observations"],
        )
        write_jsonl(
            work / "revalidation.streaming-onnx.jsonl",
            result["review"]["evidence"],
        )
        write_jsonl(
            work / "observations.streaming-onnx.jsonl",
            result["review"]["observations"],
        )
        retained.update(
            {
                "detections.streaming-onnx.jsonl",
                "tracking.streaming-onnx.jsonl",
                "revalidation.streaming-onnx.jsonl",
                "observations.streaming-onnx.jsonl",
            }
        )

    # Only diagnostic runs need to claim the developer filename. Final runs
    # leave unrelated or unreadable files alone. Keep an owned report intact
    # until the atomic result write succeeds.
    had_development_report = _owned_development_report_exists(
        development_destination,
        destination,
        reject_unowned=development_enabled,
    )

    # The production document is intentionally compact and contains only what
    # a user, editor, or renderer needs. Diagnostic evidence is written to the
    # paired developer report below only when explicitly requested.
    write_json(destination, final_result, indent=None)
    # The new result is durable. Remove only its deterministic paired filename
    # before optionally replacing it, so write failures do not destroy the
    # prior result/report pair and final runs cannot retain a stale report.
    if had_development_report:
        development_destination.unlink()
    summary["timings"]["artifact_seconds"] = time.perf_counter() - artifact_started
    summary["timings"]["total_seconds"] = time.perf_counter() - started
    if development_enabled:
        summary["development_report"] = str(development_destination)
        detector_id, _detector = active_face_detector(config)
        repository = Path(__file__).resolve().parents[2]
        development_source_document = {
            "path": str(source),
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
            "metadata": raw_metadata,
        }
        development_report = {
            "format": "privateframe-development-report",
            "schema_version": 1,
            "artifacts_level": artifacts_level,
            "result_file": destination.name,
            "source_video": development_source_document,
            "analysis": {
                "backend": "onnxruntime-streaming-gop-roi",
                "provider": config["runtime"]["resolved_provider"],
                "active_face_detector": detector_id,
                "effective_config_sha256": sha256_json(config),
                "model_package": model_package,
                "models": model_fingerprints,
                "git": git_version(repository),
                "statistics": analysis_statistics,
                "timings": deepcopy(summary["timings"]),
                "detector_sampling": result["detector_sampling"],
                "local_review_sampling": result["local_review_sampling"],
                "endpoint_conflicts": result.get("endpoint_conflicts", {"enabled": False}),
                "cache": result["cache"],
            },
            "effective_config": config,
            "recognition_diagnostics": recognition_diagnostics,
            "observation_diagnostics": result["review"]["observations"],
            "endpoint_conflict_decisions": result.get("endpoint_conflict_decisions", []),
        }
        write_json(development_destination, development_report)
        write_json(work / "summary.streaming-onnx.json", summary)
        named_artifacts = {
            filename: work / filename
            for filename in retained
            if filename != "manifest.streaming-onnx.json"
            and (work / filename).exists()
        }
        if destination.parent != work:
            named_artifacts[destination.name] = destination
        if development_destination.parent != work:
            named_artifacts[development_destination.name] = development_destination
        manifest = {
            "schema_version": 1,
            "artifacts_level": artifacts_level,
            "source_video_sha256": development_source_document["sha256"],
            "artifacts": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in sorted(named_artifacts.items())
            },
        }
        write_json(work / "manifest.streaming-onnx.json", manifest)
    _remove_unrequested_artifacts(work, retained)
    return summary


def analyze_streaming_pipeline(
    *,
    config_path: str | Path,
    input_path: str | Path,
    workdir: str | Path,
    result_path: str | Path | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    config_override_root: str | Path | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Analyze one video while treating the encoded packet DB as runtime-only."""

    work = Path(workdir).expanduser().resolve()
    # A prior process may have terminated before it could run its finalizer.
    # The cache is never a resumable artifact, so every analysis starts fresh.
    _remove_runtime_cache(work)
    try:
        return _analyze_streaming_pipeline_impl(
            config_path=config_path,
            input_path=input_path,
            workdir=work,
            result_path=result_path,
            config_overrides=config_overrides,
            config_override_root=config_override_root,
            progress=progress,
            is_cancelled=is_cancelled,
        )
    finally:
        # ``run_stream`` closes the SQLite connection before returning or
        # raising. Remove only its three fixed runtime files; result, audit,
        # and user-selected output files remain untouched.
        _remove_runtime_cache(work)


def render_streaming_artifacts(
    *,
    input_path: str | Path,
    workdir: str | Path | None = None,
    result_path: str | Path | None = None,
    debug_path: str | Path | None = None,
    redacted_path: str | Path | None = None,
    render_config: str | Path | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    _raise_if_cancelled(is_cancelled)
    work = Path(workdir).expanduser().resolve() if workdir is not None else None
    destination = _result_path(work, result_path)
    analysis_result = _load_result(destination)
    render_settings, render_settings_sha256 = _render_settings(
        analysis_result,
        render_config,
        config_overrides,
    )
    targets: list[RenderTarget] = []
    if debug_path is not None:
        targets.append(RenderTarget("debug", Path(debug_path).expanduser().resolve()))
    if redacted_path is not None:
        targets.append(
            RenderTarget("redacted", Path(redacted_path).expanduser().resolve())
        )
    render_paths = [
        Path(input_path).expanduser().resolve(),
        destination,
        *(target.path for target in targets),
    ]
    if not paths_are_distinct(render_paths):
        raise ValueError("input, result JSON, and render output paths must be distinct")
    if work is not None:
        _reject_reserved_work_path(render_paths[0], work, label="input video")
        for target in targets:
            _reject_reserved_work_path(target.path, work, label=f"{target.mode} video")
    effective_render_settings = {
        **render_settings,
        **render_settings["video_output"],
    }
    result = render_artifacts(
        source=input_path,
        targets=targets,
        settings=effective_render_settings,
        analysis_result=analysis_result,
        progress=progress,
        is_cancelled=is_cancelled,
    )
    if str(effective_render_settings.get("backend", "pyav")) == "pyav":
        # ``render_artifacts`` has now returned and released its Python frame.
        # Collect once more at the public API boundary so any PyAV cycles that
        # depended on frame locals cannot leak into interpreter shutdown.
        gc.collect()
    result.update(
        {
            "phase": "render",
            "result": str(destination),
            "render_settings_sha256": render_settings_sha256,
            "recognition_policy": deepcopy(render_settings["recognition_policy"]),
        }
    )
    return result


def run_streaming_pipeline(
    *,
    config_path: str | Path,
    input_path: str | Path,
    debug_path: str | Path | None,
    workdir: str | Path,
    redacted_path: str | Path | None = None,
    result_path: str | Path | None = None,
    render_config: str | Path | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    config_override_root: str | Path | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    _raise_if_cancelled(is_cancelled)
    source = Path(input_path).expanduser().resolve()
    if debug_path is None and redacted_path is None:
        raise ValueError("process requires a debug and/or redacted video output")
    outputs = [source, _result_path(Path(workdir).expanduser().resolve(), result_path)]
    if debug_path is not None:
        outputs.append(Path(debug_path).expanduser().resolve())
    if redacted_path is not None:
        outputs.append(Path(redacted_path).expanduser().resolve())
    if not paths_are_distinct(outputs):
        raise ValueError("input and output paths must be distinct")
    work = Path(workdir).expanduser().resolve()
    _reject_reserved_result_name(outputs[1])
    _reject_reserved_work_path(source, work, label="input video")
    _reject_reserved_work_path(
        outputs[1],
        work,
        label="result JSON",
        allowed={work / RESULT_FILENAME},
    )
    for output in outputs[2:]:
        _reject_reserved_work_path(output, work, label="render video")

    analysis_total = 0

    def report_analysis(current: int, total: int, _message: str) -> None:
        nonlocal analysis_total
        analysis_total = total
        if progress is not None:
            progress(current, total * 2, "analysis")

    def report_render(current: int, total: int, _message: str) -> None:
        if progress is not None:
            progress(analysis_total + current, analysis_total + total, "render")

    analysis = analyze_streaming_pipeline(
        config_path=config_path,
        input_path=source,
        workdir=workdir,
        result_path=result_path,
        config_overrides=config_overrides,
        config_override_root=config_override_root,
        progress=report_analysis if progress is not None else None,
        is_cancelled=is_cancelled,
    )
    _raise_if_cancelled(is_cancelled)
    rendered = render_streaming_artifacts(
        input_path=source,
        workdir=workdir,
        result_path=result_path,
        debug_path=debug_path,
        redacted_path=redacted_path,
        render_config=render_config,
        config_overrides=(
            {
                path: value
                for path, value in config_overrides.items()
                if path.startswith("render.")
            }
            if config_overrides
            else None
        ),
        progress=report_render if progress is not None else None,
        is_cancelled=is_cancelled,
    )
    return {"analysis": analysis, "render": rendered}


__all__ = [
    "analyze_streaming_pipeline",
    "render_streaming_artifacts",
    "run_streaming_pipeline",
    "validate_result_document",
]
