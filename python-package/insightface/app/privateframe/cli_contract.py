"""Machine-readable description of the PrivateFrame command-line contract."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ... import __version__ as _INSIGHTFACE_VERSION
from .base_config import DEFAULT_CONFIG_PATH, _current_config_schema
from .config_options import DESCRIBE_OPTION_GROUPS, DESCRIBE_OPTION_METADATA

_TOOL_NAME = "insightface-privateframe"
_MISSING = object()
_COMMAND_DISCOVERY: dict[str, dict[str, Any]] = {
    "analyze": {
        "operation": "analyze",
        "when_to_use": (
            "Choose this when the user wants reusable face-analysis JSON only, "
            "or wants to inspect or edit the analysis before rendering."
        ),
        "reads": ["source_video"],
        "optional_reads": ["analysis_config"],
        "outputs": ["result_json"],
    },
    "render": {
        "operation": "render",
        "when_to_use": (
            "Choose this when a compatible result_json already exists and the user "
            "wants to render or re-render the privacy-redacted video without model "
            "inference."
        ),
        "reads": ["source_video", "result_json"],
        "optional_reads": ["render_config"],
        "outputs": ["result_video"],
    },
    "process": {
        "operation": "analyze_and_render",
        "when_to_use": (
            "Default choice when the user asks to blur, pixelate, mosaic, redact, "
            "or anonymize faces in a video now."
        ),
        "reads": ["source_video"],
        "optional_reads": ["analysis_config", "render_config"],
        "outputs": ["result_json", "result_video"],
    },
    "describe": {
        "operation": "describe_contract",
        "when_to_use": (
            "Choose this first when a human or automation client needs to discover "
            "the tool, its data flow, commands, options, or output contract."
        ),
        "reads": [],
        "optional_reads": [],
        "outputs": [],
    },
    "doctor": {
        "operation": "check_readiness",
        "when_to_use": (
            "Choose this for read-only environment diagnostics; optionally provide "
            "the intended source video and output directory for a more specific check."
        ),
        "reads": [],
        "optional_reads": ["source_video", "analysis_config", "output_directory"],
        "outputs": [],
    },
}
_DOTTED_METADATA: dict[str, dict[str, Any]] = {
    "output.artifacts_level": {
        "enum": ["final", "audit", "debug"],
        "description": "Retention level for internal analysis artifacts.",
    },
    "models.name": {
        "enum": ["raccoon_s", "raccoon_l"],
        "description": "Manifest-backed InsightFace ModelZoo package name.",
    },
    "models.root": {
        "format": "local_directory_path",
        "description": (
            "InsightFace root containing models/<package>; the selected root is "
            "authoritative and no alternate root is searched."
        ),
    },
    "models.detection.nms_iou_threshold": {
        "minimum": 0.0,
        "maximum": 1.0,
    },
    "models.detection.max_detections": {"minimum": 1},
    "runtime.provider": {
        "special_values": ["auto"],
        "dynamic_values": "onnxruntime.get_available_providers()",
        "description": "auto selects CoreML, then CUDA, then CPU when available.",
    },
    "recognition.mode": {"enum": ["all", "blur_only", "exempt"]},
    "recognition.unknown_action": {"enum": ["auto", "blur", "keep"]},
    "recognition.profile": {"enum": ["fast", "balanced", "accurate"]},
    "recognition.similarity_threshold": {"minimum": 0.0, "maximum": 1.0},
    "recognition.max_frames_per_track": {
        "type": ["integer", "null"],
        "minimum": 1,
        "maximum": 32,
    },
    "recognition.reference_dir": {"type": ["string", "null"]},
    "revalidation.passes": {
        "type": ["array", "null"],
        "min_items": 1,
        "description": "Optional local-review cascade; null uses the shared revalidation settings.",
        "items": {
            "type": "object",
            "required": ["name", "input_size", "crop_expansion"],
            "name": {
                "description": "Converted to text; the resulting name must be nonempty and unique within the cascade.",
            },
            "input_size": {"type": "integer", "minimum": 32, "multiple_of": 32},
            "crop_expansion": {
                "type": "number",
                "exclusive_minimum": 0,
                "finite": True,
            },
        },
    },
    "scan.max_analysis_fps": {
        "type": "number",
        "exclusive_minimum": 0.0,
        "unit": "frames_per_second_of_input_video",
        "description": (
            "Approximate ceiling on regular full-frame face detection sampling, "
            "default 15 (Fast mode). This controls sampling along the input video timeline; "
            "it is not a wall-clock processing speed target and does not change "
            "the output video's frame count or frame rate. A uniform integer "
            "stride with 5% rate tolerance is derived per video: at 30, 25/30 FPS "
            "input is scanned every frame and 60 FPS every 2 frames. At 15, 30 FPS "
            "input is scanned every 2 frames (15/s), 60 FPS every 4 (15/s), and "
            "25 FPS every 2 (12.5/s). Input at or below the setting is scanned "
            "every frame. Scene boundaries, endpoints, and new-track bursts can "
            "add scans above this soft ceiling. Sampled-out frames are still "
            "decoded and rendered; by default their face regions are interpolated."
        ),
        "tuning_guidance": {
            "keep_default": (
                "Keep the default Fast mode (15) to reduce detector work for faster processing."
            ),
            "increase": (
                "Use 30 or raise toward the source FPS for "
                "fast motion, briefly visible faces, or greater detection coverage. "
                "Setting the source FPS scans every frame. "
                "Higher sampling costs more compute and cannot guarantee detection."
            ),
            "decrease": (
                "Lower below 15 (e.g. to 10) when faster processing and less detector "
                "work take priority, especially in limited-motion scenes. "
                "Wider gaps increase the risk of missing faces "
                "that appear only between samples; review representative output."
            ),
        },
    },
    "scan.session_sharing": {
        "enum": ["single_session_parallel", "single_session_serial"],
        "description": "Whether scan workers may call the shared detector Session concurrently.",
    },
    "tracking.between_scan_frames": {"enum": ["interpolate", "visual"]},
    "render.redaction.method": {"enum": ["gaussian", "mosaic"]},
    "render.redaction.box_scale": {"minimum": 0.1, "maximum": 4.0},
    "render.redaction.gaussian.algorithm": {"enum": ["exact", "pyramid"]},
    "render.video_output.backend": {"enum": ["pyav", "ffmpeg"]},
    "render.video_output.preset": {
        "description": "Encoder-specific preset; support is finalized by the selected encoder.",
    },
    "render.video_output.rate_control.mode": {"enum": ["crf", "cq", "vbr", "cbr"]},
    "render.video_output.rate_control.quality": {"minimum": 0, "maximum": 51},
    "render.video_output.keyframe_interval": {
        "minimum": 0,
        "maximum": 2_147_483_647,
        "description": "0 disables an explicit GOP interval.",
    },
    "render.video_output.rate_control.bitrate": {
        "type": ["integer", "string"],
        "format": "positive_integer_or_k_m_g_suffix",
        "maximum_bps": 9_223_372_036_854_775_807,
    },
    "render.video_output.rate_control.max_bitrate": {
        "type": ["integer", "string"],
        "format": "positive_integer_or_k_m_g_suffix",
        "maximum_bps": 9_223_372_036_854_775_807,
    },
    "render.video_output.rate_control.buffer_size": {
        "type": ["integer", "string"],
        "format": "positive_integer_or_k_m_g_suffix",
        "maximum_bits": 9_223_372_036_854_775_807,
    },
    "render.video_output.audio.debug": {"enum": ["none", "copy", "aac"]},
    "render.video_output.audio.redacted": {"enum": ["none", "copy", "aac"]},
    "render.video_output.audio.bitrate": {
        "type": ["integer", "string"],
        "format": "positive_integer_or_k_m_g_suffix",
        "maximum_bps": 9_223_372_036_854_775_807,
    },
}
_CANDIDATE_FILTER_METADATA: dict[str, dict[str, Any]] = {
    "enabled": {"type": "boolean"},
    "min_box_area_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "min_height_width_ratio": {"type": "number", "exclusive_minimum": 0.0},
    "max_height_width_ratio": {
        "type": ["number", "null"],
        "description": "null omits the maximum; otherwise it must be at least min_height_width_ratio.",
    },
    "aspect_ratio_exempt_min_area_fraction": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "description": "Must be at least min_box_area_fraction when the filter is enabled.",
    },
}

for _path, _guidance in DESCRIBE_OPTION_METADATA.items():
    _DOTTED_METADATA.setdefault(_path, {}).update(deepcopy(_guidance))


def _json_compatible(value: Any) -> Any:
    """Return an argparse/default value in a JSON-compatible representation."""

    if value == argparse.SUPPRESS:
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    return str(value)


def _action_type(action: argparse.Action) -> str:
    if isinstance(
        action,
        (
            argparse._StoreTrueAction,
            argparse._StoreFalseAction,
            argparse._HelpAction,
            argparse._VersionAction,
        ),
    ):
        return "boolean"
    if action.nargs in {"*", "+"} or (
        isinstance(action.nargs, int) and action.nargs != 1
    ):
        return "array"
    if action.type is int:
        return "integer"
    if action.type is float:
        return "number"
    return "string"


def _canonical_option(action: argparse.Action) -> tuple[str, list[str]]:
    if not action.option_strings:
        return action.dest, []
    long_options = [
        option for option in action.option_strings if option.startswith("--")
    ]
    name = long_options[0] if long_options else action.option_strings[0]
    aliases = [option for option in action.option_strings if option != name]
    return name, aliases


def _parameter_spec(action: argparse.Action) -> dict[str, Any] | None:
    if action.help == argparse.SUPPRESS:
        return None
    name, aliases = _canonical_option(action)
    default = None if action.default == argparse.SUPPRESS else action.default
    spec: dict[str, Any] = {
        "name": name,
        "dest": action.dest,
        "required": bool(action.required),
        "type": _action_type(action),
        "default": _json_compatible(default),
        "aliases": aliases,
        "help": str(action.help or ""),
    }
    if action.metavar is not None:
        spec["metavar"] = _json_compatible(action.metavar)
    if action.choices is not None:
        spec["choices"] = [_json_compatible(choice) for choice in action.choices]
    return spec


def _visible_parameters(parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            continue
        spec = _parameter_spec(action)
        if spec is not None:
            values.append(spec)
    return values


def _command_specs(
    parser: argparse.ArgumentParser,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    global_parameters = _visible_parameters(parser)
    commands: dict[str, dict[str, Any]] = {}
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        help_by_name = {
            item.dest: str(item.help or "") for item in action._choices_actions
        }
        for name, command_parser in action.choices.items():
            if name in commands:
                continue
            dotted_scope: str | None
            if name in {"analyze", "process", "doctor"}:
                dotted_scope = "all_public_fields"
            elif name == "render":
                dotted_scope = "render_fields_only"
            else:
                dotted_scope = None
            command_spec = {
                "help": help_by_name.get(name, ""),
                "description": str(command_parser.description or ""),
                "parameters": _visible_parameters(command_parser),
                "dotted_config_overrides": dotted_scope,
            }
            command_spec.update(deepcopy(_COMMAND_DISCOVERY[name]))
            if name == "analyze":
                command_spec.update(
                    {
                        "writes": ["result_json", "work_directory"],
                        "may_download_models": True,
                        "constraints": [
                            "--output-dir or --workdir is required",
                            "existing result JSON requires --overwrite",
                        ],
                    }
                )
            elif name == "render":
                command_spec.update(
                    {
                        "writes": ["result_video", "work_directory"],
                        "conditional_writes": {
                            "work_directory": (
                                "render diagnostics may be written when a paired "
                                "development report exists"
                            )
                        },
                        "may_download_models": False,
                        "constraints": [
                            "analysis JSON is selected by --output-dir, --result, or --workdir",
                            "a video output is required",
                            "existing result video requires --overwrite",
                        ],
                    }
                )
            elif name == "process":
                command_spec.update(
                    {
                        "writes": [
                            "result_json",
                            "result_video",
                            "work_directory",
                        ],
                        "may_download_models": True,
                        "constraints": [
                            "--output-dir or --workdir is required",
                            "a video output is required",
                            "existing public artifacts require --overwrite",
                        ],
                    }
                )
            else:
                command_spec.update({"writes": [], "may_download_models": False})
            commands[name] = command_spec
        break
    return global_parameters, commands


def _is_internal_config_path(parts: tuple[str, ...]) -> bool:
    if not parts:
        return False
    if parts[0] in {"schema_version", "base_config"}:
        return True
    if parts[:2] in {
        ("runtime", "providers"),
        ("runtime", "resolved_provider"),
        ("models", "manifest_path"),
    }:
        return True
    if parts in {
        ("render", "debug_line_thickness"),
        ("render", "video_output", "audio", "debug"),
    }:
        return True
    if parts[0] == "models" and parts[-1] in {
        "file",
        "path",
        "preprocessing",
    }:
        return True
    return False


def _filtered_config_tree(
    value: Any,
    parts: tuple[str, ...] = (),
) -> Any:
    if _is_internal_config_path(parts):
        return _MISSING
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            child = _filtered_config_tree(item, (*parts, str(key)))
            if child is not _MISSING:
                result[str(key)] = child
        return result
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            child = _filtered_config_tree(item, (*parts, str(index)))
            if child is not _MISSING:
                result.append(child)
        return result
    return deepcopy(value)


def _yaml_type(value: Any, *, has_default: bool) -> str:
    if not has_default or value is None:
        return "yaml"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "yaml"


def _add_dotted_option(
    output: dict[str, dict[str, Any]],
    parts: tuple[str, ...],
    default: Any,
    *,
    has_default: bool,
) -> None:
    if len(parts) < 2 or _is_internal_config_path(parts):
        return
    path = ".".join(parts)
    spec = {
        "option": f"--{path}",
        "type": _yaml_type(default, has_default=has_default),
        "has_default": has_default,
        "default": _json_compatible(default) if has_default else None,
        "nullable": bool(has_default and default is None),
    }
    spec.update(deepcopy(_DOTTED_METADATA.get(path, {})))
    if (
        len(parts) == 5
        and parts[:2] == ("scan", "passes")
        and parts[2].isdigit()
        and parts[3] == "candidate_filter"
    ):
        spec.update(deepcopy(_CANDIDATE_FILTER_METADATA.get(parts[4], {})))
    declared_type = spec.get("type")
    if isinstance(declared_type, list) and "null" in declared_type:
        spec["nullable"] = True
    spec["authoritative_validation"] = "command --dry-run"
    output[path] = spec


def _flatten_dotted_options(
    schema: Any,
    default: Any,
    parts: tuple[str, ...],
    output: dict[str, dict[str, Any]],
) -> None:
    has_default = default is not _MISSING
    if isinstance(schema, dict):
        default_mapping = default if isinstance(default, dict) else {}
        for key, child_schema in schema.items():
            child_default = default_mapping.get(key, _MISSING)
            _flatten_dotted_options(
                child_schema,
                child_default,
                (*parts, str(key)),
                output,
            )
        return
    if isinstance(schema, list):
        visible_default = default if isinstance(default, list) else []
        _add_dotted_option(
            output,
            parts,
            visible_default,
            has_default=has_default,
        )
        if schema:
            item_schema = schema[0]
            for index, item_default in enumerate(visible_default):
                _flatten_dotted_options(
                    item_schema,
                    item_default,
                    (*parts, str(index)),
                    output,
                )
        return
    _add_dotted_option(
        output,
        parts,
        None if default is _MISSING else default,
        has_default=has_default,
    )


def _full_config_contract() -> dict[str, Any]:
    raw_defaults = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw_defaults, dict):
        raise TypeError(f"invalid built-in config: {DEFAULT_CONFIG_PATH}")
    schema = _filtered_config_tree(_current_config_schema())
    defaults = _filtered_config_tree(raw_defaults)
    if not isinstance(schema, dict) or not isinstance(defaults, dict):
        raise TypeError("public configuration contract must be a mapping")
    dotted_options: dict[str, dict[str, Any]] = {}
    _flatten_dotted_options(schema, defaults, (), dotted_options)
    return {
        "schema_version": int(raw_defaults.get("schema_version", 0)),
        "default_path": str(DEFAULT_CONFIG_PATH.resolve()),
        "layer_precedence": [
            "bundled_base_yaml",
            "custom_yaml_overlay",
            "cli_dotted_overrides",
        ],
        "render_layer_precedence": [
            "analysis_result_render_defaults",
            "render_config_yaml",
            "cli_render_dotted_overrides",
        ],
        "dotted_syntax": [
            "--section.field VALUE",
            "--section.field=VALUE",
        ],
        "schema": schema,
        "defaults": defaults,
        "dotted_options": dotted_options,
        "validation": {
            "leaf_types": "validated against the packaged Base configuration",
            "semantic_and_cross_field_constraints": "authoritative command --dry-run",
            "note": (
                "enum/range metadata is included where stable; dry-run remains "
                "authoritative for encoder-specific and cross-field rules"
            ),
        },
    }


def _config_contract() -> dict[str, Any]:
    full = _full_config_contract()
    selected = {
        path: deepcopy(full["dotted_options"][path])
        for group in DESCRIBE_OPTION_GROUPS
        for path in group["fields"]
    }
    for option in selected.values():
        option.pop("authoritative_validation", None)
    return {
        "schema_version": full["schema_version"],
        "scope": "common_and_intermediate",
        "default_path": full["default_path"],
        "layer_precedence": full["layer_precedence"],
        "render_layer_precedence": full["render_layer_precedence"],
        "dotted_syntax": full["dotted_syntax"],
        "groups": deepcopy(DESCRIBE_OPTION_GROUPS),
        "dotted_options": selected,
        "unlisted_options_supported": True,
        "full_reference": {
            "format": "markdown",
            "path": str((Path(__file__).parent / "docs" / "configuration.md").resolve()),
            "scope": "all_supported_configuration",
            "when_to_read": "Read for advanced fields or complete constraints; all documented YAML and CLI overrides remain supported.",
        },
        "validation": full["validation"],
    }


def _artifact_contract() -> dict[str, Any]:
    return {
        "source_video": {
            "kind": "video_file",
            "role": "primary_input",
            "locator": "local_path",
            "option": "--input",
            "modified": False,
            "uploaded_by_privateframe": False,
            "read_by": ["analyze", "render", "process"],
            "optionally_read_by": ["doctor"],
        },
        "analysis_config": {
            "kind": "yaml_file",
            "role": "optional_analysis_configuration",
            "option": "--config",
            "default": "bundled configs/base.yaml",
            "read_by": ["analyze", "process", "doctor"],
        },
        "render_config": {
            "kind": "yaml_file",
            "role": "optional_render_configuration",
            "option": "--render-config",
            "read_by": ["render", "process"],
        },
        "output_directory": {
            "kind": "directory",
            "role": "public_artifact_destination",
            "option": "--output-dir",
            "used_by": ["analyze", "render", "process"],
            "optionally_used_by": ["doctor"],
        },
        "result_json": {
            "kind": "json_file",
            "role": "reusable_face_analysis_output",
            "format": "privateframe-result",
            "schema_version": 1,
            "purpose": "reusable analysis and editing artifact",
            "produced_by": ["analyze", "process"],
            "consumed_by": ["render"],
            "distinct_from_stdout_status": True,
            "required_fields": [
                "format",
                "schema_version",
                "source_video",
                "render_defaults",
                "recognition",
                "observations",
            ],
            "stable_render_input_schema": {
                "source_video": {
                    "type": "object",
                    "required": [
                        "file_name",
                        "metadata",
                        "coordinate_system",
                        "frame_index_origin",
                        "timing_contract",
                    ],
                    "file_name": {"type": "string", "min_length": 1},
                    "coordinate_system": {"const": "pixel_xyxy"},
                    "frame_index_origin": {"const": 0},
                    "timing_contract": {"const": "cfr_frame_index"},
                    "metadata": {
                        "type": "object",
                        "required": ["width", "height", "fps", "frame_count"],
                        "width": {"type": "integer", "minimum": 1},
                        "height": {"type": "integer", "minimum": 1},
                        "fps": {
                            "type": "number",
                            "exclusive_minimum": 0,
                            "finite": True,
                        },
                        "frame_count": {"type": "integer", "minimum": 1},
                        "duration": {
                            "type": "number",
                            "exclusive_minimum": 0,
                            "finite": True,
                        },
                    },
                },
                "render_defaults": {
                    "type": "object",
                    "recognition_policy": {
                        "type": "object",
                        "required": ["mode", "unknown_action"],
                        "mode": {"enum": ["all", "blur_only", "exempt"]},
                        "unknown_action": {"enum": ["blur", "keep"]},
                        "description": (
                            "Resolved photo policy from analysis, reused by render. "
                            "auto is resolved before saving; all always blurs."
                        ),
                    },
                },
                "recognition": {
                    "type": "object",
                    "required": ["enabled"],
                    "enabled": {"type": "boolean"},
                    "fields_when_enabled": {
                        "references": {
                            "type": "object",
                            "accepted_images": {"type": "integer", "minimum": 1},
                            "skipped_images": {"type": "integer", "minimum": 0},
                            "files": {
                                "type": "array",
                                "items": {
                                    "file": {"type": "string"},
                                    "detected_face_count": {"type": "integer", "minimum": 1},
                                    "selected_box": {"type": "array", "length": 4},
                                },
                            },
                            "skipped": {
                                "type": "array",
                                "items": {
                                    "file": {"type": "string"},
                                    "reason": {"type": "string"},
                                },
                            },
                            "fingerprint": {"type": "string"},
                        },
                        "tracks": {
                            "type": "object",
                            "keys": "track_id",
                            "values": {
                                "status": {"enum": ["CONFIRMED", "UNKNOWN", "CONFLICT"]},
                                "matched_reference_files": {"type": "array", "items": {"type": "string"}},
                                "similarity": {"type": ["number", "null"]},
                                "reason": {"type": "string"},
                            },
                        },
                    },
                },
                "observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["frame_idx", "track_id", "box", "source"],
                        "frame_idx": {
                            "type": "integer",
                            "minimum": 0,
                            "constraint": "frame_idx < source_video.metadata.frame_count",
                        },
                        "track_id": {"type": "string", "min_length": 1},
                        "source": {
                            "enum": [
                                "detector",
                                "tracked",
                                "interpolated",
                                "repaired",
                                "manual",
                            ]
                        },
                        "box": {
                            "type": "array",
                            "length": 4,
                            "items": {"type": "number", "finite": True},
                            "constraint": "x2 > x1 and y2 > y1",
                        },
                        "reduced_assurance": {"type": "boolean"},
                        "identity_unconfirmed": {
                            "type": "boolean",
                            "description": "True makes this region follow unknown_action even if its track has a confirmed match.",
                        },
                        "additional_fields_allowed": True,
                    },
                },
            },
            "schema_semantics": {
                "stable_render_input_schema": (
                    "Canonical structure emitted by current analysis and recommended "
                    "for new or edited documents. The renderer also accepts the "
                    "legacy variations listed in accepted_input_compatibility."
                ),
                "authoritative_validation": "render --dry-run",
            },
            "accepted_input_compatibility": {
                "source_video_optional_fields": [
                    "file_name",
                    "coordinate_system",
                    "frame_index_origin",
                    "timing_contract",
                ],
                "source_video_file_name_nullable": True,
                "source_video_labels": (
                    "When supplied, coordinate_system, frame_index_origin, and "
                    "timing_contract must match their canonical values."
                ),
                "observation_source": {
                    "type": "string",
                    "min_length": 1,
                    "accepts_unlisted_values": True,
                },
            },
            "source_compatibility": {
                "checks": ["width", "height", "fps", "decoded_frame_count"],
                "content_hash_required": False,
            },
        },
        "result_video": {
            "kind": "video_file",
            "role": "privacy_redacted_output",
            "produced_by": ["render", "process"],
            "derived_from": ["source_video", "result_json"],
            "redacted_regions": "face_regions",
            "styles": ["gaussian", "mosaic"],
        },
        "work_directory": {
            "kind": "directory",
            "role": "private_runtime_workspace",
            "written_by": ["analyze", "process"],
            "conditionally_written_by": ["render"],
            "purpose": "private runtime and optional audit workspace",
            "contains": [
                "temporary encoded-packet SQLite cache",
                "audit/debug artifacts when requested",
            ],
            "sqlite_cache_removed_after_analysis": True,
            "default_final_mode_may_leave_empty_directory": True,
        },
        "output_directory_option": "--output-dir",
        "input_stem_placeholder": "<input_stem>",
        "path_precedence": [
            "explicit artifact path option",
            "--output-dir stable-name defaults",
            "--workdir result JSON fallback",
        ],
        "output_dir_defaults": {
            "result_json": "<input_stem>_privateframe.json",
            "result_video": "<input_stem>_privateframe.mp4",
            "work_directory": ".<input_stem>_privateframe_work",
        },
        "workdir_fallbacks": {
            "result_json": "<workdir>/result.privateframe.json",
            "result_video": None,
        },
        "commands": {
            "analyze": ["result_json"],
            "render": ["result_video"],
            "process": ["result_json", "result_video"],
        },
        "explicit_path_options": {
            "result_json": ["--result", "--json-output"],
            "result_video": ["--redacted", "--video-output"],
            "work_directory": ["--workdir"],
        },
        "overwrite": {
            "default": False,
            "enable_option": "--overwrite",
            "policy": "existing public output artifacts are rejected unless enabled",
            "concurrency": (
                "cooperating PrivateFrame CLI writers exclusively claim every "
                "public output and shared work directory; locks owned by dead "
                "processes are reclaimed"
            ),
        },
    }


def _discovery_contract() -> dict[str, Any]:
    return {
        "audience": [
            "humans",
            "shell_automation",
            "vendor_neutral_ai_coding_agents",
        ],
        "machine_discovery_argv": [_TOOL_NAME, "describe"],
        "use_when_user_intent_matches": [
            "blur faces in a video",
            "blur only people shown in reference photos",
            "keep people shown in reference photos visible and blur everyone else",
            "pixelate or mosaic faces in a video",
            "redact faces for privacy",
            "anonymize faces in local video",
            "detect and track face regions in video",
            "produce editable face-analysis JSON and render it later",
        ],
        "scope": "face regions in local video",
        "not_for": [
            "full-body anonymization",
            "license-plate redaction",
            "arbitrary-object redaction",
        ],
        "default_command_for_face_redaction": "process",
        "command_selection": {
            "redact_video_now": "process",
            "analysis_json_only": "analyze",
            "render_existing_or_edited_result_json": "render",
            "inspect_environment_readiness": "doctor",
            "inspect_machine_contract": "describe",
        },
        "required_user_values_for_default_workflow": [
            "source_video local path",
            "output directory local path",
        ],
        "defaults": {
            "redaction_style": "gaussian",
            "mosaic_override": ["--render.redaction.method", "mosaic"],
        },
        "safe_automation": {
            "preflight": "run the selected execution command with --dry-run",
            "continue_when": {
                "stdout.ok": True,
                "stdout.ready": True,
            },
            "execution": "repeat the same argv without --dry-run",
            "overwrite": (
                "do not add --overwrite unless replacement of existing public "
                "artifacts is explicitly intended"
            ),
            "progress": "use --progress jsonl or none; progress is written to stderr",
            "final_status": "parse the single JSON object written to stdout",
        },
        "assurance_note": (
            "Face analysis is model-based; review the result when missed redaction "
            "would create a significant privacy risk."
        ),
    }


def _primary_io_contract() -> dict[str, Any]:
    return {
        "primary_input": "source_video",
        "primary_file_outputs": ["result_json", "result_video"],
        "command_outputs_field_semantics": "file_artifacts_only",
        "status_output": {
            "channel": "stdout",
            "artifact": False,
            "purpose": "one final machine-readable command status object",
            "not_the_analysis_result_file": True,
        },
        "progress_output": {
            "channel": "stderr",
            "artifact": False,
            "purpose": "diagnostics and optional progress events",
        },
    }


def _recommended_workflows() -> list[dict[str, Any]]:
    return [
        {
            "id": "redact_video_now",
            "default": True,
            "description": (
                "Analyze a source video and immediately produce both reusable JSON "
                "and a privacy-redacted video."
            ),
            "commands": ["process"],
            "inputs": ["source_video", "output_directory"],
            "outputs": ["result_json", "result_video"],
            "preflight_argv": [
                _TOOL_NAME,
                "process",
                "--input",
                "<source_video_path>",
                "--output-dir",
                "<output_directory_path>",
                "--dry-run",
            ],
            "execute_argv": [
                _TOOL_NAME,
                "process",
                "--input",
                "<source_video_path>",
                "--output-dir",
                "<output_directory_path>",
            ],
        },
        {
            "id": "analysis_json_only",
            "default": False,
            "description": "Analyze face regions without rendering a video.",
            "commands": ["analyze"],
            "inputs": ["source_video", "output_directory"],
            "outputs": ["result_json"],
            "preflight_argv": [
                _TOOL_NAME,
                "analyze",
                "--input",
                "<source_video_path>",
                "--output-dir",
                "<output_directory_path>",
                "--dry-run",
            ],
            "execute_argv": [
                _TOOL_NAME,
                "analyze",
                "--input",
                "<source_video_path>",
                "--output-dir",
                "<output_directory_path>",
            ],
        },
        {
            "id": "analyze_edit_then_render",
            "default": False,
            "description": (
                "Analyze first, optionally edit the reusable result JSON, then render "
                "without rerunning model inference."
            ),
            "commands": ["analyze", "render"],
            "inputs": ["source_video", "output_directory"],
            "intermediate": "result_json",
            "intermediate_may_be_edited": True,
            "outputs": ["result_json", "result_video"],
            "steps": [
                {
                    "command": "analyze",
                    "preflight_argv": [
                        _TOOL_NAME,
                        "analyze",
                        "--input",
                        "<source_video_path>",
                        "--output-dir",
                        "<output_directory_path>",
                        "--dry-run",
                    ],
                    "argv": [
                        _TOOL_NAME,
                        "analyze",
                        "--input",
                        "<source_video_path>",
                        "--output-dir",
                        "<output_directory_path>",
                    ],
                },
                {
                    "operation": "optionally_edit_result_json",
                    "artifact": "result_json",
                    "constraint": "preserve the declared result_json schema",
                },
                {
                    "command": "render",
                    "preflight_argv": [
                        _TOOL_NAME,
                        "render",
                        "--input",
                        "<source_video_path>",
                        "--output-dir",
                        "<output_directory_path>",
                        "--dry-run",
                    ],
                    "argv": [
                        _TOOL_NAME,
                        "render",
                        "--input",
                        "<source_video_path>",
                        "--output-dir",
                        "<output_directory_path>",
                    ],
                },
            ],
        },
    ]


def _status_output_contract() -> dict[str, Any]:
    return {
        "required_envelope_fields": [
            "status_schema_version",
            "ok",
            "command",
        ],
        "stdout": {
            "format": "json",
            "framing": "exactly_one_compact_json_object",
            "contains": "final success or error status",
            "text_exceptions": ["--help", "--version"],
        },
        "stderr": {
            "contains": "Application progress and reference-photo diagnostics are JSONL in jsonl mode, otherwise text; third-party runtime diagnostics may still be plain text.",
            "progress_option": "--progress",
            "progress_choices": ["auto", "text", "jsonl", "none"],
            "jsonl_progress_stream": "stderr",
            "jsonl_log_record": {
                "log_schema_version": 1,
                "event": "log",
                "level": "logging severity, such as info or warning",
                "stage": "recognition",
                "message": "reference-photo selection, skipped-photo reason, or import summary",
            },
            "none_semantics": "Suppresses progress events; reference-photo diagnostics still appear as text on stderr.",
            "auto": {
                "interactive": "text",
                "non_interactive": "jsonl",
            },
        },
        "error_envelope": {
            "field": "error",
            "fields": [
                "code",
                "type",
                "stage",
                "message",
                "retryable",
                "hints",
            ],
            "codes": [
                "invalid_arguments",
                "output_exists",
                "output_busy",
                "missing_dependency",
                "dependency_import_failed",
                "file_not_found",
                "permission_denied",
                "cancelled",
                "provider_unavailable",
                "invalid_config",
                "media_error",
                "operation_failed",
            ],
        },
        "execution_success": {
            "required_fields": ["artifacts", "runtime", "timings", "summary"],
            "artifact_paths_field": "artifacts",
            "artifact_path_keys": ["result_json", "result_video"],
            "semantics": (
                "resolved absolute public artifact paths used or produced by the "
                "invocation; null means that path is not applicable"
            ),
            "summary_semantics": (
                "stable user-facing result counts only; detailed algorithm, model, "
                "cache, recognition, and tracking diagnostics are excluded"
            ),
            "summary_field_semantics": {
                "frame_count": "decoded video frames processed or rendered",
                "face_tracks": (
                    "accepted face trajectories; this is a trajectory count, not a "
                    "guaranteed count of distinct people"
                ),
                "face_regions": (
                    "frame-local face regions; repeated appearances across frames are "
                    "counted separately"
                ),
                "redacted_face_regions": "face regions blurred or mosaicked",
                "kept_face_regions": (
                    "face regions intentionally left visible by the recognition policy"
                ),
            },
            "summary_fields_by_command": {
                "analyze": [
                    "frame_count",
                    "face_tracks",
                    "face_regions",
                ],
                "render": [
                    "frame_count",
                    "face_regions",
                    "redacted_face_regions",
                    "kept_face_regions",
                ],
                "process": [
                    "frame_count",
                    "face_tracks",
                    "face_regions",
                    "redacted_face_regions",
                    "kept_face_regions",
                ],
            },
            "timing_fields_by_command": {
                "analyze": ["total_seconds"],
                "render": ["total_seconds"],
                "process": ["total_seconds"],
            },
            "timing_semantics": (
                "total_seconds is the reported pipeline time: analysis for analyze, "
                "rendering for render, and their sum for process. It is not the "
                "complete CLI process wall time and excludes CLI startup and orchestration."
            ),
            "runtime_provider_semantics": (
                "resolved inference provider for analyze/process; null for render, "
                "which does not run ONNX inference"
            ),
        },
        "dry_run": {
            "option": "--dry-run",
            "executes_inference": False,
            "writes_public_artifacts": False,
            "downloads_models": False,
            "creates_onnx_sessions": False,
            "codec_probe": (
                "may open an in-memory codec context or encode one synthetic "
                "frame to a null sink"
            ),
            "returns": ["ready", "plan", "checks", "diagnostics"],
            "artifact_paths_field": "plan.artifacts",
            "artifact_path_keys": ["result_json", "result_video"],
        },
        "readiness": {
            "commands": [
                "doctor",
                "analyze --dry-run",
                "render --dry-run",
                "process --dry-run",
            ],
            "field": "ready",
            "semantics": "false means the inspected operation is not currently runnable",
            "exit_code": (
                "readiness reports still exit 0 when diagnostics complete; "
                "inspect the ready field"
            ),
            "diagnostic_failure": (
                "Internal diagnostic failures return ok=false, ready=false, "
                "an error envelope, and exit code 1."
            ),
        },
    }


def _examples(commands: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    if "process" in commands:
        process = [
            _TOOL_NAME,
            "process",
            "--input",
            "/data/video.mp4",
            "--output-dir",
            "/data/output",
        ]
        parameter_names = {item["name"] for item in commands["process"]["parameters"]}
        if "--progress" in parameter_names:
            process.extend(["--progress", "jsonl"])
        values.append(
            {
                "name": "analyze_and_render",
                "description": (
                    "Default end-to-end choice for a request to blur or redact "
                    "faces in a video."
                ),
                "argv": process,
                "reads": ["source_video"],
                "artifacts": [
                    "/data/output/video_privateframe.json",
                    "/data/output/video_privateframe.mp4",
                ],
                "stdout": "one final status JSON object",
                "stderr": "Application progress and reference-photo diagnostics are JSONL because --progress jsonl is selected; third-party runtime diagnostics may still be plain text.",
            }
        )
        values.append(
            {
                "name": "fast_analysis_cap",
                "description": (
                    "Explicitly select the default Fast mode (15 analysis FPS); "
                    "wider sampling gaps can miss brief faces."
                ),
                "argv": [*process, "--scan.max_analysis_fps", "15"],
            }
        )
        values.append(
            {
                "name": "mosaic_faces",
                "description": "Use mosaic/pixelation instead of Gaussian blur.",
                "argv": [
                    *process,
                    "--render.redaction.method",
                    "mosaic",
                ],
            }
        )
        for mode, name, description in (
            (
                "blur_only", "blur_reference_people",
                "Blur people matched to photos in one folder; unmatched or uncertain people remain visible by default.",
            ),
            (
                "exempt", "keep_reference_people_visible",
                "Keep people matched to photos in one folder visible; blur unmatched or uncertain people by default.",
            ),
        ):
            values.append({
                "name": name,
                "description": description,
                "argv": [
                    *process, "--recognition.mode", mode,
                    "--recognition.reference_dir", "/data/reference_photos",
                ],
            })
    if "analyze" in commands:
        values.append(
            {
                "name": "analysis_json_only",
                "description": (
                    "Produce reusable face-analysis JSON without rendering video."
                ),
                "argv": [
                    _TOOL_NAME,
                    "analyze",
                    "--input",
                    "/data/video.mp4",
                    "--output-dir",
                    "/data/output",
                ],
            }
        )
    if "render" in commands:
        values.append(
            {
                "name": "render_existing_analysis",
                "description": (
                    "Render the stable result JSON in the output directory together "
                    "with its original source video."
                ),
                "argv": [
                    _TOOL_NAME,
                    "render",
                    "--input",
                    "/data/video.mp4",
                    "--output-dir",
                    "/data/output",
                ],
            }
        )
    if "doctor" in commands:
        values.append(
            {
                "name": "read_only_environment_check",
                "description": (
                    "Inspect runtime readiness without model downloads, inference, or "
                    "output creation."
                ),
                "argv": [_TOOL_NAME, "doctor"],
            }
        )
    if "describe" in commands:
        values.append(
            {
                "name": "machine_readable_contract",
                "description": (
                    "Discover the tool's purpose, data flow, workflows, options, and "
                    "status contract."
                ),
                "argv": [_TOOL_NAME, "describe"],
            }
        )
    return values


def build_describe_payload(
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    """Build the side-effect-free, JSON-compatible CLI description payload."""

    global_parameters, commands = _command_specs(parser)
    return {
        "contract_schema_version": 2,
        "status_schema_version": 1,
        "ok": True,
        "command": "describe",
        "tool": {
            "name": _TOOL_NAME,
            "version": str(_INSIGHTFACE_VERSION),
            "purpose_id": "video_face_privacy_redaction",
            "summary": str(parser.description or ""),
            "primary_input": "source_video",
            "primary_file_outputs": ["result_json", "result_video"],
            "capabilities": [
                "face_detection_and_tracking",
                "gaussian_face_blur",
                "mosaic_face_redaction",
                "reusable_json_analysis",
                "render_from_edited_analysis",
                "optional_identity_aware_redaction_policy",
            ],
            "execution_scope": {
                "source_video": "local_file",
                "source_video_modified": False,
                "source_video_uploaded_by_privateframe": False,
            },
            "installation": {
                "python_distribution": "insightface",
                "required_extra": "privateframe",
                "pip_argv": [
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "insightface[privateframe]",
                ],
            },
            "entry_points": [
                _TOOL_NAME,
                "python -m insightface_privateframe_bootstrap",
            ],
            "global_parameters": global_parameters,
        },
        "discovery": _discovery_contract(),
        "primary_io": _primary_io_contract(),
        "recommended_workflows": _recommended_workflows(),
        "commands": commands,
        "config": _config_contract(),
        "artifacts": _artifact_contract(),
        "status_output": _status_output_contract(),
        "exit_codes": {
            "0": {
                "ok": True,
                "meaning": "command_completed; inspect ready for doctor and dry-run",
            },
            "1": {"ok": False, "meaning": "operation_or_diagnostic_failure"},
            "2": {"ok": False, "meaning": "command_usage_error"},
            "130": {"ok": False, "meaning": "interrupted_by_user"},
        },
        "examples": _examples(commands),
    }


__all__ = ["build_describe_payload"]
