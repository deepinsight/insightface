"""Read-only PrivateFrame environment and job diagnostics.

The doctor's own checks deliberately stop before PrivateFrame operations that
mutate local or remote state. In particular they never ask ModelZoo to
materialize a package, construct no ONNX Runtime Session, and decode at most
the first video frame of an optional input. Dependency imports occur before
these checks and remain subject to those dependencies' initialization behavior.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import io
import math
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import onnxruntime as ort
import yaml

from ... import __version__ as INSIGHTFACE_VERSION
from ...model_zoo.onnxruntime_utils import (
    DEFAULT_PROVIDER_PRIORITY,
    get_default_providers,
)
from ...model_zoo.package_manifest import (
    DETECTION_TASK,
    MODEL_PACKAGE_MANIFEST,
    MODEL_PACKAGE_TASKS,
    RECOGNITION_TASK,
    SUPPORTED_MANIFEST_PACKAGES,
    VERIFICATION_TASK,
    load_model_package,
)
from .artifacts import sha256_file
from .base_config import DEFAULT_CONFIG_PATH
from .config import load_config
from .model_catalog import DEFAULT_INSIGHTFACE_ROOT
from .output_paths import default_output_paths
from .video import paths_are_distinct, paths_refer_to_same_location, probe_video


_PYAV_MINIMUM_VERSION = (12, 0, 0)
_ANALYSIS_FPS_TOLERANCE_MULTIPLIER = 1.05


def _json_safe(value: Any) -> Any:
    """Return only values accepted by ``json.dumps(..., allow_nan=False)``."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


def _error_details(error: BaseException) -> dict[str, str]:
    return {
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    ok: bool,
    message: str,
    *,
    severity: str | None = None,
    details: Mapping[str, Any] | None = None,
    skipped: bool = False,
) -> None:
    if severity is None:
        severity = "info" if ok else "error"
    record_details = dict(details or {})
    if skipped:
        record_details["skipped"] = True
    checks.append(
        {
            "name": str(name),
            "ok": bool(ok),
            "severity": str(severity),
            "message": str(message),
            "details": _json_safe(record_details),
        }
    )


def _skipped_check(
    checks: list[dict[str, Any]],
    name: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> None:
    _add_check(
        checks,
        name,
        True,
        message,
        severity="info",
        details=details,
        skipped=True,
    )


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _numeric_version(value: str) -> tuple[int, ...]:
    output: list[int] = []
    for component in str(value).split("."):
        digits = "".join(character for character in component if character.isdigit())
        if not digits:
            break
        output.append(int(digits))
    return tuple(output)


def _version_at_least(value: str, minimum: tuple[int, ...]) -> bool:
    current = _numeric_version(value)
    width = max(len(current), len(minimum))
    return (*current, *(0 for _ in range(width - len(current)))) >= (
        *minimum,
        *(0 for _ in range(width - len(minimum))),
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return value


def _peek_config_field(
    path: Path,
    keys: Sequence[str],
    *,
    seen: set[Path] | None = None,
) -> Any:
    """Best-effort inheritance lookup used only after full validation fails."""

    resolved = path.expanduser().resolve()
    visited = set() if seen is None else seen
    if resolved in visited:
        return None
    visited.add(resolved)
    raw = _load_yaml_mapping(resolved)
    cursor: Any = raw
    for key in keys:
        if not isinstance(cursor, Mapping) or key not in cursor:
            cursor = None
            break
        cursor = cursor[key]
    if cursor is not None:
        return cursor
    if resolved == DEFAULT_CONFIG_PATH.resolve():
        return None
    base_value = raw.get("base_config")
    if base_value is None:
        base = DEFAULT_CONFIG_PATH.resolve()
    elif isinstance(base_value, str) and base_value.strip():
        candidate = Path(base_value).expanduser()
        base = (
            (resolved.parent / candidate).resolve()
            if not candidate.is_absolute()
            else candidate.resolve()
        )
    else:
        return None
    return _peek_config_field(base, keys, seen=visited)


def _resolve_analysis_stride(source_fps: float, maximum_fps: float) -> int:
    source = float(source_fps)
    maximum = float(maximum_fps)
    if not math.isfinite(source) or source <= 0.0:
        raise ValueError("source FPS must be finite and positive")
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("scan.max_analysis_fps must be finite and positive")
    return max(
        1,
        math.ceil(source / (maximum * _ANALYSIS_FPS_TOLERANCE_MULTIPLIER)),
    )


def _environment_report(checks: list[dict[str, Any]]) -> dict[str, Any]:
    python = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }
    system = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "mac_ver": platform.mac_ver()[0] or None,
    }
    _add_check(
        checks,
        "environment.python",
        True,
        f"Python {python['version']} ({python['implementation']})",
        details=python,
    )
    _add_check(
        checks,
        "environment.insightface",
        True,
        f"InsightFace {INSIGHTFACE_VERSION}",
        details={
            "version": INSIGHTFACE_VERSION,
            "distribution_version": _distribution_version("insightface"),
        },
    )
    return {
        "python": python,
        "platform": system,
        "insightface": {
            "version": INSIGHTFACE_VERSION,
            "distribution_version": _distribution_version("insightface"),
        },
    }


def _runtime_report(
    checks: list[dict[str, Any]],
    effective_config: Mapping[str, Any] | None,
    *,
    requested_fallback: Any = None,
) -> dict[str, Any]:
    distributions = {
        name: version
        for name in ("onnxruntime", "onnxruntime-gpu")
        if (version := _distribution_version(name)) is not None
    }
    report: dict[str, Any] = {
        "onnxruntime_version": getattr(ort, "__version__", None),
        "distributions": distributions,
        "provider_priority": list(DEFAULT_PROVIDER_PRIORITY),
        "available_providers": [],
        "default_providers": [],
        "requested_provider": None,
        "resolved_provider": None,
        "resolved_providers": [],
        "scrfd_static_shape_sessions": None,
    }
    try:
        available = [str(value) for value in ort.get_available_providers()]
        defaults = get_default_providers(available)
        report["available_providers"] = available
        report["default_providers"] = defaults
        _add_check(
            checks,
            "runtime.onnxruntime",
            True,
            f"ONNX Runtime {report['onnxruntime_version']} reports {len(available)} provider(s)",
            details={
                "version": report["onnxruntime_version"],
                "distributions": distributions,
                "available_providers": available,
            },
        )
        if {"onnxruntime", "onnxruntime-gpu"} <= set(distributions):
            _add_check(
                checks,
                "runtime.distribution_conflict",
                False,
                "Both onnxruntime and onnxruntime-gpu distributions are installed",
                severity="warning",
                details={"distributions": distributions},
            )
        else:
            _add_check(
                checks,
                "runtime.distribution_conflict",
                True,
                "No conflicting CPU/GPU ONNX Runtime distributions were found",
                details={"distributions": distributions},
            )
    except Exception as error:
        _add_check(
            checks,
            "runtime.onnxruntime",
            False,
            "Could not inspect ONNX Runtime providers",
            details=_error_details(error),
        )
        available = []
        defaults = []

    runtime = (
        effective_config.get("runtime", {})
        if isinstance(effective_config, Mapping)
        else {}
    )
    requested = runtime.get("provider", requested_fallback)
    resolved = runtime.get("resolved_provider")
    providers = runtime.get("providers")
    if not isinstance(providers, list):
        providers = []
    if requested is None:
        resolved = defaults[0] if defaults else None
        providers = list(defaults)
    elif requested == "auto" and resolved is None:
        resolved = defaults[0] if defaults else None
        providers = list(defaults)
    elif resolved is None and isinstance(requested, str):
        resolved = requested
        providers = [requested]
        if requested != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
    report.update(
        {
            "requested_provider": requested,
            "resolved_provider": resolved,
            "resolved_providers": [str(value) for value in providers],
            "scrfd_static_shape_sessions": runtime.get("scrfd_static_shape_sessions"),
        }
    )
    if requested is None:
        _skipped_check(
            checks,
            "runtime.configured_provider",
            "No configuration was supplied; reporting the automatic default only",
            details={
                "default_providers": defaults,
                "resolved_provider": resolved,
            },
        )
    else:
        supported = isinstance(resolved, str) and resolved in available
        _add_check(
            checks,
            "runtime.configured_provider",
            supported,
            (
                f"Resolved Provider {resolved} is available"
                if supported
                else f"Resolved Provider {resolved!r} is unavailable"
            ),
            details={
                "requested_provider": requested,
                "resolved_provider": resolved,
                "resolved_providers": providers,
                "available_providers": available,
            },
        )
    return report


def _codec_capability(av_module: Any, name: str, mode: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": str(name),
        "mode": str(mode),
        "available": False,
    }
    try:
        codec = av_module.Codec(str(name), str(mode))
        result.update(
            {
                "available": True,
                "canonical_name": str(codec.name),
                "long_name": str(codec.long_name),
                "type": str(codec.type),
                "is_decoder": bool(codec.is_decoder),
                "is_encoder": bool(codec.is_encoder),
            }
        )
        video_formats = getattr(codec, "video_formats", None)
        if video_formats:
            result["supported_pixel_formats"] = [
                str(getattr(value, "name", value)) for value in video_formats
            ]
    except Exception as error:
        result.update(_error_details(error))
    return result


def _ffmpeg_help_capability(
    executable: str,
    *,
    kind: str,
    name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": kind,
        "name": name,
        "available": False,
        "command": [executable, "-hide_banner", "-h", f"{kind}={name}"],
    }
    try:
        completed = subprocess.run(
            result["command"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        output = f"{completed.stdout}\n{completed.stderr}".strip()
        recognized = (
            completed.returncode == 0
            and "is not recognized by FFmpeg" not in output
            and "Unknown" not in output
        )
        result.update(
            {
                "available": recognized,
                "returncode": completed.returncode,
                "output_excerpt": output[:2000],
            }
        )
        if kind == "encoder" and recognized:
            prefix = "Supported pixel formats:"
            for line in output.splitlines():
                stripped = line.strip()
                if stripped.startswith(prefix):
                    result["supported_pixel_formats"] = (
                        stripped[len(prefix) :].strip().split()
                    )
                    break
    except Exception as error:
        result.update(_error_details(error))
    return result


def _bitrate_value(value: Any) -> int:
    text = str(value).strip().lower()
    multipliers = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    if text and text[-1] in multipliers:
        return round(float(text[:-1]) * multipliers[text[-1]])
    return int(text)


def _pyav_encoder_settings_capability(
    av_module: Any,
    video_output: Mapping[str, Any],
) -> dict[str, Any]:
    encoder = str(video_output.get("encoder", ""))
    result: dict[str, Any] = {
        "backend": "pyav",
        "encoder": encoder,
        "available": False,
        "probe": "in_memory_codec_context_open",
    }
    context = None
    try:
        context = av_module.CodecContext.create(encoder, "w")
        context.width = 64
        context.height = 64
        context.pix_fmt = str(video_output["pixel_format"])
        context.time_base = Fraction(1, 30)
        context.framerate = Fraction(30, 1)
        options: dict[str, str] = {}
        if video_output.get("preset"):
            options["preset"] = str(video_output["preset"])
        rate = video_output.get("rate_control", {})
        mode = str(rate.get("mode", "")) if isinstance(rate, Mapping) else ""
        if mode in {"crf", "cq"}:
            quality = str(int(rate["quality"]))
            if "nvenc" in encoder:
                options.update({"rc": "vbr", "cq": quality, "b": "0"})
            else:
                options["crf"] = quality
        elif mode == "cbr":
            options["minrate"] = str(rate["bitrate"])
            options["maxrate"] = str(rate["bitrate"])
            if rate.get("buffer_size"):
                options["bufsize"] = str(rate["buffer_size"])
        elif mode == "vbr" and rate.get("max_bitrate"):
            options["maxrate"] = str(rate["max_bitrate"])
        context.options = options
        if mode in {"vbr", "cbr"}:
            context.bit_rate = _bitrate_value(rate["bitrate"])
        if int(video_output.get("keyframe_interval", 0)) > 0:
            context.gop_size = int(video_output["keyframe_interval"])
        context.open()
        unconsumed = dict(context.options or {})
        result.update(
            {
                "available": not unconsumed,
                "options": options,
                "unconsumed_options": unconsumed,
            }
        )
    except Exception as error:
        result.update(_error_details(error))
    finally:
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
    return result


def _ffmpeg_encoder_settings_capability(
    executable: str,
    video_output: Mapping[str, Any],
) -> dict[str, Any]:
    encoder = str(video_output.get("encoder", ""))
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-video_size",
        "64x64",
        "-framerate",
        "30",
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        "-c:v",
        encoder,
    ]
    if video_output.get("preset"):
        command.extend(["-preset", str(video_output["preset"])])
    rate = video_output.get("rate_control", {})
    mode = str(rate.get("mode", "")) if isinstance(rate, Mapping) else ""
    if mode in {"crf", "cq"}:
        quality = str(int(rate["quality"]))
        if "nvenc" in encoder:
            command.extend(["-rc:v", "vbr", "-cq:v", quality, "-b:v", "0"])
        else:
            command.extend(["-crf", quality])
    elif mode == "vbr":
        command.extend(["-b:v", str(rate["bitrate"])])
        if rate.get("max_bitrate"):
            command.extend(["-maxrate", str(rate["max_bitrate"])])
    elif mode == "cbr":
        bitrate = str(rate["bitrate"])
        command.extend(["-b:v", bitrate, "-minrate", bitrate, "-maxrate", bitrate])
        if rate.get("buffer_size"):
            command.extend(["-bufsize", str(rate["buffer_size"])])
    if int(video_output.get("keyframe_interval", 0)) > 0:
        command.extend(["-g", str(int(video_output["keyframe_interval"]))])
    command.extend(
        ["-pix_fmt", str(video_output["pixel_format"]), "-an", "-f", "null", "-"]
    )
    result: dict[str, Any] = {
        "backend": "ffmpeg",
        "encoder": encoder,
        "available": False,
        "probe": "one_in_memory_synthetic_frame",
        "command": command,
    }
    try:
        completed = subprocess.run(
            command,
            input=b"\0" * (64 * 64 * 3),
            capture_output=True,
            check=False,
            timeout=30,
        )
        result.update(
            {
                "available": completed.returncode == 0,
                "returncode": completed.returncode,
                "stderr_excerpt": completed.stderr.decode("utf-8", errors="replace")[
                    :2000
                ],
            }
        )
    except Exception as error:
        result.update(_error_details(error))
    return result


def _ffmpeg_aac_settings_capability(
    executable: str,
    bitrate: Any,
) -> dict[str, Any]:
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-i",
        "pipe:0",
        "-frames:a",
        "1",
        "-c:a",
        "aac",
        "-b:a",
        str(bitrate),
        "-f",
        "null",
        "-",
    ]
    result: dict[str, Any] = {
        "backend": "ffmpeg",
        "encoder": "aac",
        "bitrate": bitrate,
        "available": False,
        "probe": "one_in_memory_synthetic_audio_frame",
        "command": command,
    }
    try:
        completed = subprocess.run(
            command,
            input=b"\0" * (1024 * 2 * 2),
            capture_output=True,
            check=False,
            timeout=30,
        )
        result.update(
            {
                "available": completed.returncode == 0,
                "returncode": completed.returncode,
                "stderr_excerpt": completed.stderr.decode("utf-8", errors="replace")[
                    :2000
                ],
            }
        )
    except Exception as error:
        result.update(_error_details(error))
    return result


def _mp4_audio_copy_capability(
    av_module: Any,
    input_path: str | Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "probe": "in_memory_mp4_stream_template",
        "input": str(Path(input_path).expanduser().resolve()),
    }
    source = None
    output = None
    try:
        source = av_module.open(str(input_path), mode="r")
        audio_stream = next(iter(source.streams.audio), None)
        if audio_stream is None:
            result.update({"available": True, "input_codec": None})
            return result
        result["input_codec"] = str(audio_stream.codec_context.name)
        buffer = io.BytesIO()
        output = av_module.open(buffer, mode="w", format="mp4")
        method = getattr(output, "add_stream_from_template", None)
        if method is not None:
            method(audio_stream)
        else:
            output.add_stream(template=audio_stream)
        output.start_encoding()
        result["available"] = True
    except Exception as error:
        result.update(_error_details(error))
    finally:
        if output is not None:
            try:
                output.close()
            except Exception as error:
                if result.get("available"):
                    result["available"] = False
                    result.update(_error_details(error))
        if source is not None:
            source.close()
    return result


def _pyav_report(
    checks: list[dict[str, Any]],
    effective_config: Mapping[str, Any] | None,
    *,
    check_render_capabilities: bool = True,
    input_path: str | Path | None = None,
    input_audio_present: bool | None = None,
    target_modes: Sequence[str] = ("redacted",),
) -> tuple[dict[str, Any], Any | None]:
    report: dict[str, Any] = {
        "installed": False,
        "version": _distribution_version("av"),
        "minimum_version": ".".join(str(value) for value in _PYAV_MINIMUM_VERSION),
        "capabilities": {},
    }
    try:
        av_module = importlib.import_module("av")
        report["installed"] = True
        report["version"] = str(getattr(av_module, "__version__", report["version"]))
        supported_version = _version_at_least(
            str(report["version"]),
            _PYAV_MINIMUM_VERSION,
        )
        _add_check(
            checks,
            "runtime.pyav",
            supported_version,
            (
                f"PyAV {report['version']} is available"
                if supported_version
                else f"PyAV {report['version']} is older than 12.0.0"
            ),
            details={
                "version": report["version"],
                "minimum_version": report["minimum_version"],
            },
        )
    except Exception as error:
        _add_check(
            checks,
            "runtime.pyav",
            False,
            "PyAV is unavailable",
            details=_error_details(error),
        )
        return report, None

    if input_audio_present is None and input_path is not None:
        source_container = None
        try:
            source_container = av_module.open(
                str(Path(input_path).expanduser().resolve()),
                mode="r",
            )
            input_audio_present = (
                next(
                    iter(source_container.streams.audio),
                    None,
                )
                is not None
            )
        except Exception:
            # The dedicated media check reports an unreadable input. Keep the
            # audio requirement unknown here instead of duplicating that error.
            input_audio_present = None
        finally:
            if source_container is not None:
                source_container.close()

    if not check_render_capabilities:
        _skipped_check(
            checks,
            "runtime.video_encoder",
            "Video-render capability checks are not required for analysis-only operation",
        )
        return report, av_module

    video_output = (
        effective_config.get("render", {}).get("video_output", {})
        if isinstance(effective_config, Mapping)
        and isinstance(effective_config.get("render"), Mapping)
        else {}
    )
    if not isinstance(video_output, Mapping):
        video_output = {}
    backend = video_output.get("backend")
    encoder = video_output.get("encoder")
    pixel_format = video_output.get("pixel_format")
    if backend is None:
        _skipped_check(
            checks,
            "runtime.video_encoder",
            "No render configuration was supplied; encoder capability was not selected",
        )
    elif backend == "pyav" and isinstance(encoder, str) and encoder:
        capability = _codec_capability(av_module, encoder, "w")
        report["capabilities"]["configured_video_encoder"] = capability
        _add_check(
            checks,
            "runtime.video_encoder",
            bool(capability["available"] and capability.get("is_encoder")),
            (
                f"PyAV encoder {encoder} is available"
                if capability["available"]
                else f"PyAV encoder {encoder} is unavailable"
            ),
            details=capability,
        )
        supported_pixel_formats = capability.get("supported_pixel_formats")
        try:
            av_module.VideoFormat(str(pixel_format))
            globally_available = bool(pixel_format)
        except Exception as error:
            globally_available = False
            format_error = _error_details(error)
        else:
            format_error = {}
        pair_declared = bool(supported_pixel_formats)
        pixel_ok = bool(
            globally_available
            and (not pair_declared or str(pixel_format) in supported_pixel_formats)
        )
        pixel_details: dict[str, Any] = {
            "encoder": encoder,
            "pixel_format": pixel_format,
            "globally_available": globally_available,
            "encoder_formats_declared": pair_declared,
            "supported_pixel_formats": supported_pixel_formats,
            **format_error,
        }
        _add_check(
            checks,
            "runtime.pixel_format",
            pixel_ok,
            (
                f"PyAV encoder {encoder} supports pixel format {pixel_format}"
                if pixel_ok
                else f"PyAV encoder {encoder!r} does not support pixel format {pixel_format!r}"
            ),
            details=pixel_details,
        )
        try:
            av_module.format.ContainerFormat("mp4")
            muxer_ok = True
            muxer_details: dict[str, Any] = {"container": "mp4"}
        except Exception as error:
            muxer_ok = False
            muxer_details = {"container": "mp4", **_error_details(error)}
        _add_check(
            checks,
            "runtime.mp4_muxer",
            muxer_ok,
            (
                "PyAV MP4 container support is available"
                if muxer_ok
                else "PyAV MP4 container support is unavailable"
            ),
            details=muxer_details,
        )
        if (
            capability["available"]
            and capability.get("is_encoder")
            and pixel_ok
            and muxer_ok
        ):
            settings_capability = _pyav_encoder_settings_capability(
                av_module,
                video_output,
            )
            report["capabilities"]["configured_encoder_settings"] = settings_capability
            _add_check(
                checks,
                "runtime.encoder_options",
                bool(settings_capability["available"]),
                (
                    "PyAV encoder accepted the configured preset and rate control"
                    if settings_capability["available"]
                    else "PyAV encoder rejected the configured preset or rate control"
                ),
                details=settings_capability,
            )
        else:
            _skipped_check(
                checks,
                "runtime.encoder_options",
                "Encoder-option probe requires a usable encoder, pixel format, and MP4 muxer",
            )
    elif backend == "ffmpeg":
        executable = shutil.which("ffmpeg")
        encoder_capability = (
            _ffmpeg_help_capability(
                executable,
                kind="encoder",
                name=str(encoder),
            )
            if executable is not None and isinstance(encoder, str) and encoder
            else {
                "kind": "encoder",
                "name": encoder,
                "available": False,
            }
        )
        report["capabilities"]["ffmpeg"] = {
            "path": executable,
            "configured_video_encoder": encoder_capability,
        }
        _add_check(
            checks,
            "runtime.video_encoder",
            bool(executable is not None and encoder_capability["available"]),
            (
                f"ffmpeg encoder {encoder} is available"
                if executable is not None and encoder_capability["available"]
                else f"ffmpeg encoder {encoder!r} is unavailable"
            ),
            details={
                "backend": "ffmpeg",
                "path": executable,
                "configured_encoder": encoder,
                "capability": encoder_capability,
            },
        )
        supported_pixel_formats = encoder_capability.get(
            "supported_pixel_formats",
            [],
        )
        pixel_ok = bool(
            encoder_capability.get("available")
            and isinstance(pixel_format, str)
            and pixel_format
            and pixel_format in supported_pixel_formats
        )
        _add_check(
            checks,
            "runtime.pixel_format",
            pixel_ok,
            (
                f"ffmpeg encoder {encoder} supports pixel format {pixel_format}"
                if pixel_ok
                else f"ffmpeg encoder {encoder!r} does not declare pixel format {pixel_format!r}"
            ),
            details={
                "encoder": encoder,
                "pixel_format": pixel_format,
                "supported_pixel_formats": supported_pixel_formats,
            },
        )
        muxer_capability = (
            _ffmpeg_help_capability(executable, kind="muxer", name="mp4")
            if executable is not None
            else {"kind": "muxer", "name": "mp4", "available": False}
        )
        _add_check(
            checks,
            "runtime.mp4_muxer",
            bool(muxer_capability["available"]),
            (
                "ffmpeg MP4 muxer is available"
                if muxer_capability["available"]
                else "ffmpeg MP4 muxer is unavailable"
            ),
            details=muxer_capability,
        )
        if (
            executable is not None
            and encoder_capability.get("available")
            and pixel_ok
            and muxer_capability.get("available")
        ):
            settings_capability = _ffmpeg_encoder_settings_capability(
                executable,
                video_output,
            )
            report["capabilities"]["ffmpeg"][
                "configured_encoder_settings"
            ] = settings_capability
            _add_check(
                checks,
                "runtime.encoder_options",
                bool(settings_capability["available"]),
                (
                    "ffmpeg encoded a synthetic frame with the configured options"
                    if settings_capability["available"]
                    else "ffmpeg rejected the configured preset or rate control"
                ),
                details=settings_capability,
            )
        else:
            _skipped_check(
                checks,
                "runtime.encoder_options",
                "Encoder-option probe requires ffmpeg, its encoder, pixel format, and MP4 muxer",
            )
        audio_settings = video_output.get("audio", {})
        configured_aac = bool(
            isinstance(audio_settings, Mapping)
            and any(audio_settings.get(str(key)) == "aac" for key in target_modes)
        )
        needs_aac = configured_aac and input_audio_present is not False
        if needs_aac:
            aac_capability = (
                _ffmpeg_help_capability(executable, kind="encoder", name="aac")
                if executable is not None
                else {"kind": "encoder", "name": "aac", "available": False}
            )
            audio_bitrate = audio_settings.get("bitrate", "192k")
            if executable is not None and aac_capability.get("available"):
                aac_settings_capability = _ffmpeg_aac_settings_capability(
                    executable,
                    audio_bitrate,
                )
            else:
                aac_settings_capability = {
                    "backend": "ffmpeg",
                    "encoder": "aac",
                    "bitrate": audio_bitrate,
                    "available": False,
                    "probe": "skipped_encoder_unavailable",
                }
            report["capabilities"]["ffmpeg"][
                "configured_audio_encoder_settings"
            ] = aac_settings_capability
            audio_encoder_ok = bool(
                aac_capability.get("available")
                and aac_settings_capability.get("available")
            )
            _add_check(
                checks,
                "runtime.audio_encoder",
                audio_encoder_ok,
                (
                    "ffmpeg AAC encoder accepted the configured bitrate"
                    if audio_encoder_ok
                    else "ffmpeg AAC encoder rejected the configured bitrate"
                ),
                details={
                    "encoder_capability": aac_capability,
                    "configured_settings": aac_settings_capability,
                },
            )
        elif configured_aac and input_audio_present is False:
            _skipped_check(
                checks,
                "runtime.audio_encoder",
                "Source has no audio stream, so configured AAC settings are unused",
                details={
                    "backend": "ffmpeg",
                    "configured_bitrate": audio_settings.get("bitrate", "192k"),
                    "input_audio_present": False,
                    "target_modes": list(target_modes),
                },
            )
    else:
        _add_check(
            checks,
            "runtime.video_encoder",
            False,
            "Render backend or encoder is invalid",
            details={"backend": backend, "encoder": encoder},
        )
    return report, av_module


def _model_package_report(
    package_name: str,
    *,
    model_root: Path,
    selected: bool,
    required_tasks: set[str],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    package_path = model_root / "models" / package_name
    manifest_path = package_path / MODEL_PACKAGE_MANIFEST
    result: dict[str, Any] = {
        "name": package_name,
        "selected": selected,
        "path": str(package_path),
        "directory_exists": package_path.is_dir(),
        "manifest": {
            "path": str(manifest_path),
            "exists": manifest_path.is_file(),
            "valid": False,
            "manifest_version": None,
            "source_schema": None,
        },
        "tasks": {},
    }
    if not package_path.is_dir():
        if selected:
            _add_check(
                checks,
                "models.selected_directory",
                False,
                f"Selected model package {package_name} is not installed",
                details={"path": package_path},
            )
        return result
    if selected:
        _add_check(
            checks,
            "models.selected_directory",
            True,
            f"Selected model package directory exists: {package_name}",
            details={"path": package_path},
        )
    try:
        package = load_model_package(package_path)
        result["manifest"].update(
            {
                "valid": True,
                "path": str(package.manifest_path),
                "manifest_version": package.manifest_version,
                "source_schema": package.source_schema,
            }
        )
        if selected:
            _add_check(
                checks,
                "models.selected_manifest",
                True,
                f"Selected package manifest is valid: {package_name}",
                details=result["manifest"],
            )
    except Exception as error:
        result["manifest"].update(_error_details(error))
        if selected:
            _add_check(
                checks,
                "models.selected_manifest",
                False,
                f"Selected package manifest is invalid: {package_name}",
                details=result["manifest"],
            )
        return result

    for task in MODEL_PACKAGE_TASKS:
        descriptor = package.tasks.get(task)
        required = task in required_tasks
        if descriptor is None:
            task_report = {
                "task": task,
                "required": required,
                "declared": False,
                "exists": False,
                "bytes": None,
                "preprocessing": None,
                "sha256": {
                    "declared": None,
                    "actual": None,
                    "status": "not_declared",
                },
            }
            result["tasks"][task] = task_report
            if selected:
                _add_check(
                    checks,
                    f"models.selected.{task}",
                    not required,
                    (
                        f"Selected model package is missing required task: {task}"
                        if required
                        else f"Optional model task is not declared: {task}"
                    ),
                    severity="error" if required else "info",
                    details=task_report,
                )
            continue
        path = descriptor.path
        task_report: dict[str, Any] = {
            "task": task,
            "file": descriptor.file,
            "path": str(path),
            "required": required,
            "declared": True,
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "preprocessing": descriptor.metadata.get("preprocessing"),
            "sha256": {
                "declared": descriptor.sha256,
                "actual": None,
                "status": (
                    "not_declared"
                    if descriptor.sha256 is None
                    else "not_checked"
                ),
            },
        }
        if selected and path.is_file() and descriptor.sha256 is not None:
            try:
                actual_sha256 = sha256_file(path)
                task_report["sha256"].update(
                    {
                        "actual": actual_sha256,
                        "status": (
                            "verified"
                            if actual_sha256 == descriptor.sha256
                            else "mismatch"
                        ),
                    }
                )
            except Exception as error:
                task_report["sha256"].update(
                    {"status": "error", **_error_details(error)}
                )
        elif selected and not path.is_file() and descriptor.sha256 is not None:
            task_report["sha256"]["status"] = "file_missing"
        if task == VERIFICATION_TASK:
            task_report["expansion"] = descriptor.metadata.get("expansion")
        result["tasks"][task] = task_report
        if selected:
            digest_status = str(task_report["sha256"]["status"])
            task_ok = bool(task_report["exists"]) and digest_status not in {
                "error",
                "mismatch",
            }
            if not task_report["exists"]:
                message = f"Selected {task} model file is missing"
            elif digest_status == "mismatch":
                message = f"Selected {task} model SHA-256 does not match the manifest"
            elif digest_status == "error":
                message = f"Selected {task} model SHA-256 could not be checked"
            elif digest_status == "verified":
                message = f"Selected {task} model file exists and its SHA-256 is verified"
            else:
                message = f"Selected {task} model file exists; no SHA-256 is declared"
            _add_check(
                checks,
                f"models.selected.{task}",
                task_ok,
                message,
                severity=(
                    "error"
                    if required and not task_ok
                    else "warning" if not task_ok else "info"
                ),
                details=task_report,
            )
    return result


def _models_report(
    checks: list[dict[str, Any]],
    effective_config: Mapping[str, Any] | None,
    *,
    selected_fallback: Any = None,
    root_fallback: Any = DEFAULT_INSIGHTFACE_ROOT,
    config_supplied: bool,
    check_models: bool,
) -> dict[str, Any]:
    models = (
        effective_config.get("models", {})
        if isinstance(effective_config, Mapping)
        else {}
    )
    selected_value = (
        models.get("name", selected_fallback)
        if isinstance(models, Mapping)
        else selected_fallback
    )
    selected = str(selected_value).strip() if selected_value is not None else None
    root_value = (
        models.get("root", root_fallback)
        if isinstance(models, Mapping)
        else root_fallback
    )
    recognition = (
        effective_config.get("recognition", {})
        if isinstance(effective_config, Mapping)
        else {}
    )
    recognition_mode = (
        str(recognition.get("mode", "all"))
        if isinstance(recognition, Mapping)
        else "all"
    )
    required_tasks = {DETECTION_TASK, VERIFICATION_TASK}
    if recognition_mode != "all":
        required_tasks.add(RECOGNITION_TASK)
    try:
        if not isinstance(root_value, (str, os.PathLike)):
            raise TypeError("models.root must be a path string")
        root_text = os.fspath(root_value)
        if not root_text.strip():
            raise ValueError("models.root must be a non-empty path")
        model_root = Path(root_text).expanduser().resolve()
    except Exception as error:
        report = {
            "insightface_root": None,
            "root": None,
            "configured_root": _json_safe(root_value),
            "supported_packages": list(SUPPORTED_MANIFEST_PACKAGES),
            "selected_package": selected,
            "required_tasks": sorted(required_tasks),
            "packages": {},
            "checked": bool(check_models),
            "error": _error_details(error),
        }
        _add_check(
            checks,
            "models.root",
            False,
            "Configured models.root is not a usable path",
            details={"configured_root": root_value, **_error_details(error)},
        )
        return report
    report = {
        "insightface_root": str(model_root),
        "root": str(model_root / "models"),
        "supported_packages": list(SUPPORTED_MANIFEST_PACKAGES),
        "selected_package": selected,
        "required_tasks": sorted(required_tasks),
        "packages": {},
        "checked": bool(check_models),
    }
    if not check_models:
        _skipped_check(
            checks,
            "models.selected_package",
            "Model-package checks were disabled for this operation",
            details={
                "selected": selected,
                "supported": list(SUPPORTED_MANIFEST_PACKAGES),
            },
        )
        report["skipped"] = True
        return report
    for package_name in SUPPORTED_MANIFEST_PACKAGES:
        report["packages"][package_name] = _model_package_report(
            package_name,
            model_root=model_root,
            selected=package_name == selected,
            required_tasks=required_tasks,
            checks=checks,
        )
    if not config_supplied:
        _skipped_check(
            checks,
            "models.selected_package",
            "No configuration was supplied; supported local packages were inspected without selecting one",
        )
    elif selected not in SUPPORTED_MANIFEST_PACKAGES:
        _add_check(
            checks,
            "models.selected_package",
            False,
            f"Selected model package {selected!r} is unsupported",
            details={
                "selected": selected,
                "supported": list(SUPPORTED_MANIFEST_PACKAGES),
            },
        )
    else:
        _add_check(
            checks,
            "models.selected_package",
            True,
            f"Selected model package is supported: {selected}",
            details={"selected": selected},
        )
    return report


def _fraction(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _audio_output_checks(
    checks: list[dict[str, Any]],
    video_output: Mapping[str, Any] | None,
    *,
    input_audio_codec: str | None,
    input_audio_present: bool,
    target_modes: Sequence[str],
    name_root: str,
    copy_capability: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(video_output, Mapping):
        _skipped_check(
            checks,
            name_root,
            "No valid render configuration was available for audio compatibility",
        )
        return
    audio_settings = video_output.get("audio", {})
    if not isinstance(audio_settings, Mapping):
        _add_check(
            checks,
            name_root,
            False,
            "Effective render audio settings must be a mapping",
        )
        return
    unique_targets = list(dict.fromkeys(str(value) for value in target_modes))
    if not unique_targets:
        _skipped_check(
            checks,
            name_root,
            "No video artifact was selected for audio compatibility checking",
        )
        return
    backend = str(video_output.get("backend", ""))
    for target in unique_targets:
        check_name = name_root if len(unique_targets) == 1 else f"{name_root}.{target}"
        audio_mode = audio_settings.get(target)
        if audio_mode is None:
            _add_check(
                checks,
                check_name,
                False,
                f"No audio mode is configured for the {target} artifact",
                details={"target": target},
            )
            continue
        if not input_audio_present or audio_mode == "none":
            audio_ok = True
        elif audio_mode == "copy":
            audio_ok = bool(
                isinstance(copy_capability, Mapping)
                and copy_capability.get("available")
            )
        elif backend == "pyav" and audio_mode == "aac":
            # The current PyAV writer remuxes AAC; it does not transcode.
            audio_ok = input_audio_codec == "aac"
        else:
            audio_ok = audio_mode in {"copy", "aac"}
        if backend == "ffmpeg":
            implementation = "ffmpeg"
        elif audio_mode in {"copy", "aac"}:
            implementation = "packet_remux"
        else:
            implementation = "none"
        if audio_ok:
            message = f"Input audio is compatible with the {target} output mode"
        elif audio_mode == "copy":
            message = "Input audio codec cannot be copied into an MP4 container"
        else:
            message = "PyAV aac mode requires AAC source audio"
        _add_check(
            checks,
            check_name,
            audio_ok,
            message,
            details={
                "target": target,
                "backend": backend,
                "configured_mode": audio_mode,
                "input_audio_present": input_audio_present,
                "input_codec": input_audio_codec,
                "implementation": implementation,
                "copy_capability": copy_capability,
            },
        )


def _media_report(
    checks: list[dict[str, Any]],
    input_path: str | Path | None,
    effective_config: Mapping[str, Any] | None,
    av_module: Any | None,
    *,
    check_render_capabilities: bool = True,
) -> dict[str, Any]:
    if input_path is None:
        _skipped_check(
            checks,
            "media.input",
            "No input video was supplied",
        )
        return {"provided": False, "skipped": True}
    source = Path(input_path).expanduser().resolve()
    report: dict[str, Any] = {
        "provided": True,
        "path": str(source),
        "exists": source.is_file(),
        "probe_mode": "container_headers_and_first_video_frame",
        "first_frame_decoded": False,
        "container": None,
        "video_stream": None,
        "audio_stream": None,
        "analysis": None,
    }
    if not source.is_file():
        _add_check(
            checks,
            "media.input",
            False,
            "Input video does not exist",
            details={"path": source},
        )
        return report
    if av_module is None:
        _add_check(
            checks,
            "media.input",
            False,
            "Input video cannot be inspected because PyAV is unavailable",
            details={"path": source},
        )
        return report
    try:
        container = av_module.open(str(source), mode="r")
        try:
            container_duration = (
                float(container.duration) / float(av_module.time_base)
                if container.duration is not None
                else None
            )
            report["container"] = {
                "format": str(container.format.name),
                "long_name": str(container.format.long_name),
                "duration_seconds": container_duration,
                "stream_count": len(container.streams),
                "video_stream_count": len(container.streams.video),
                "audio_stream_count": len(container.streams.audio),
                "bytes": source.stat().st_size,
            }
            video_streams = list(container.streams.video)
            if not video_streams:
                raise RuntimeError("container has no video stream")
            stream = video_streams[0]
            fps = (
                _fraction(getattr(stream, "average_rate", None))
                or _fraction(getattr(stream, "guessed_rate", None))
                or _fraction(getattr(stream, "base_rate", None))
            )
            stream_duration = (
                float(stream.duration * stream.time_base)
                if stream.duration is not None and stream.time_base is not None
                else container_duration
            )
            declared_frames = int(stream.frames or 0)
            estimated_frames = (
                round(stream_duration * fps)
                if declared_frames <= 0
                and stream_duration is not None
                and fps is not None
                else None
            )
            codec_name = str(stream.codec_context.name)
            decoder = _codec_capability(av_module, codec_name, "r")
            report["video_stream"] = {
                "index": int(stream.index),
                "codec": codec_name,
                "profile": getattr(stream.codec_context, "profile", None),
                "width": int(stream.codec_context.width),
                "height": int(stream.codec_context.height),
                "pixel_format": getattr(stream.codec_context, "pix_fmt", None),
                "fps": fps,
                "frame_count": (
                    declared_frames if declared_frames > 0 else estimated_frames
                ),
                "declared_frame_count": (
                    declared_frames if declared_frames > 0 else None
                ),
                "estimated_frame_count": estimated_frames,
                "duration_seconds": stream_duration,
                "decoder": decoder,
            }
            audio_streams = list(container.streams.audio)
            if audio_streams:
                audio = audio_streams[0]
                audio_codec = str(audio.codec_context.name)
                report["audio_stream"] = {
                    "index": int(audio.index),
                    "codec": audio_codec,
                    "sample_rate": int(
                        getattr(audio.codec_context, "sample_rate", 0) or 0
                    )
                    or None,
                    "channels": int(getattr(audio.codec_context, "channels", 0) or 0)
                    or None,
                    "duration_seconds": (
                        float(audio.duration * audio.time_base)
                        if audio.duration is not None and audio.time_base is not None
                        else container_duration
                    ),
                }
            if fps is None:
                raise RuntimeError("video stream does not declare a usable frame rate")
            if (
                int(report["video_stream"]["width"]) <= 0
                or int(report["video_stream"]["height"]) <= 0
            ):
                raise RuntimeError("video stream does not declare usable dimensions")
            if not bool(decoder.get("available") and decoder.get("is_decoder")):
                raise RuntimeError(f"PyAV decoder {codec_name} is unavailable")
            first_frame = next(iter(container.decode(stream)), None)
            if first_frame is None:
                raise RuntimeError("video stream did not decode a first frame")
            frame_format = getattr(first_frame, "format", None)
            report["first_frame_decoded"] = True
            report["first_frame"] = {
                "width": int(getattr(first_frame, "width", 0) or 0),
                "height": int(getattr(first_frame, "height", 0) or 0),
                "pixel_format": (
                    str(getattr(frame_format, "name", frame_format))
                    if frame_format is not None
                    else None
                ),
            }
            if (
                report["first_frame"]["width"] <= 0
                or report["first_frame"]["height"] <= 0
            ):
                raise RuntimeError("decoded first frame has invalid dimensions")
            report["privateframe_metadata"] = probe_video(source).to_dict()
            scan = (
                effective_config.get("scan", {})
                if isinstance(effective_config, Mapping)
                else {}
            )
            maximum_fps = (
                scan.get("max_analysis_fps") if isinstance(scan, Mapping) else None
            )
            if maximum_fps is not None:
                stride = _resolve_analysis_stride(fps, float(maximum_fps))
                report["analysis"] = {
                    "max_analysis_fps": float(maximum_fps),
                    "tolerance_multiplier": _ANALYSIS_FPS_TOLERANCE_MULTIPLIER,
                    "effective_frame_stride": stride,
                    "nominal_regular_analysis_fps": fps / stride,
                    "between_scan_frames": (
                        effective_config.get("tracking", {}).get(
                            "between_scan_frames",
                            "interpolate",
                        )
                        if isinstance(effective_config.get("tracking"), Mapping)
                        else None
                    ),
                }
        finally:
            container.close()
        _add_check(
            checks,
            "media.input",
            True,
            "Input container and first video frame are readable",
            details={
                "path": source,
                "container": report["container"],
                "video_stream": report["video_stream"],
                "first_frame": report["first_frame"],
                "privateframe_metadata": report["privateframe_metadata"],
            },
        )
        if not check_render_capabilities:
            _skipped_check(
                checks,
                "media.audio_output",
                "Audio-output checks are not required for analysis-only operation",
            )
        else:
            video_output = (
                effective_config.get("render", {}).get("video_output", {})
                if isinstance(effective_config, Mapping)
                and isinstance(effective_config.get("render"), Mapping)
                else None
            )
            input_audio_codec = (
                report["audio_stream"].get("codec")
                if isinstance(report.get("audio_stream"), Mapping)
                else None
            )
            audio_settings = (
                video_output.get("audio", {})
                if isinstance(video_output, Mapping)
                else {}
            )
            copy_capability = (
                _mp4_audio_copy_capability(av_module, source)
                if isinstance(audio_settings, Mapping)
                and audio_settings.get("redacted") == "copy"
                and isinstance(report.get("audio_stream"), Mapping)
                else None
            )
            _audio_output_checks(
                checks,
                video_output,
                input_audio_codec=input_audio_codec,
                input_audio_present=isinstance(report.get("audio_stream"), Mapping),
                target_modes=("redacted",),
                name_root="media.audio_output",
                copy_capability=copy_capability,
            )
        if effective_config is None:
            _skipped_check(
                checks,
                "media.analysis_stride",
                "No valid configuration was available to derive the analysis stride",
            )
        else:
            _add_check(
                checks,
                "media.analysis_stride",
                report["analysis"] is not None,
                "Analysis stride was derived from the source and configured FPS ceiling",
                details=report["analysis"] or {},
            )
    except Exception as error:
        report.update(_error_details(error))
        _add_check(
            checks,
            "media.input",
            False,
            "Input container or first video frame could not be decoded",
            details={"path": source, **_error_details(error)},
        )
    return report


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current if current.exists() else None


def _output_report(
    checks: list[dict[str, Any]],
    output_dir: str | Path | None,
    input_path: str | Path | None,
) -> dict[str, Any]:
    if output_dir is None:
        _skipped_check(
            checks,
            "output.directory",
            "No output directory was supplied",
        )
        _skipped_check(
            checks,
            "output.targets",
            "Expected output targets require an explicit output directory",
        )
        return {"provided": False, "skipped": True}
    requested = Path(output_dir).expanduser()
    destination = requested.resolve()
    exists = destination.exists()
    is_directory = destination.is_dir()
    writable_base = (
        destination if is_directory else _nearest_existing_parent(destination.parent)
    )
    writable = bool(
        writable_base is not None
        and writable_base.is_dir()
        and os.access(writable_base, os.W_OK | os.X_OK)
    )
    creatable = bool(not exists and writable)
    report: dict[str, Any] = {
        "provided": True,
        "requested_path": str(requested),
        "path": str(destination),
        "exists": exists,
        "is_directory": is_directory,
        "writable": writable,
        "creatable": creatable,
        "writability_probe": "os.access_without_file_creation",
        "nearest_existing_parent": str(writable_base) if writable_base else None,
        "expected_targets": None,
        "conflicts": [],
    }
    directory_ok = bool((exists and is_directory and writable) or creatable)
    _add_check(
        checks,
        "output.directory",
        directory_ok,
        (
            "Output directory is writable"
            if exists and directory_ok
            else (
                "Output directory can be created by the current process"
                if directory_ok
                else "Output directory is not usable"
            )
        ),
        details=report,
    )
    if input_path is None:
        _skipped_check(
            checks,
            "output.targets",
            "Expected output targets require an input video name",
        )
        return report
    source = Path(input_path).expanduser().resolve()
    paths = default_output_paths(source, destination)
    targets = {
        "result_json": str(paths.result_json),
        "result_video": str(paths.result_video),
        "workdir": str(paths.workdir),
    }
    report["expected_targets"] = targets
    conflicts: list[dict[str, Any]] = []
    for name, text in targets.items():
        path = Path(text)
        if paths_refer_to_same_location(source, path):
            conflicts.append({"kind": "input_collision", "target": name, "path": text})
        if path.exists():
            conflicts.append(
                {
                    "kind": "already_exists",
                    "target": name,
                    "path": text,
                    "is_directory": path.is_dir(),
                }
            )
    target_paths = [Path(value) for value in targets.values()]
    if not paths_are_distinct(target_paths):
        conflicts.append({"kind": "target_collision", "paths": list(targets.values())})
    report["conflicts"] = conflicts
    hard_conflicts = [
        item
        for item in conflicts
        if item["kind"] in {"input_collision", "target_collision"}
        or (
            item["kind"] == "already_exists"
            and (
                (
                    item.get("target") == "workdir"
                    and not item.get("is_directory", False)
                )
                or (item.get("target") != "workdir" and item.get("is_directory", False))
            )
        )
    ]
    existing_only = [item for item in conflicts if item["kind"] == "already_exists"]
    _add_check(
        checks,
        "output.targets",
        not hard_conflicts,
        (
            "Expected output targets are distinct and do not collide with the input"
            if not hard_conflicts
            else "Expected output targets contain a path collision"
        ),
        details={
            "targets": targets,
            "conflicts": conflicts,
        },
    )
    if existing_only:
        _add_check(
            checks,
            "output.existing_targets",
            False,
            "One or more expected targets already exist",
            severity="warning",
            details={"existing": existing_only},
        )
    else:
        _add_check(
            checks,
            "output.existing_targets",
            True,
            "No expected output target currently exists",
        )
    return report


def _summary(checks: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "total": len(checks),
        "passed": sum(bool(item.get("ok")) for item in checks),
        "failed": sum(not bool(item.get("ok")) for item in checks),
        "errors": sum(
            not bool(item.get("ok")) and item.get("severity") == "error"
            for item in checks
        ),
        "warnings": sum(
            not bool(item.get("ok")) and item.get("severity") == "warning"
            for item in checks
        ),
        "skipped": sum(bool(item.get("details", {}).get("skipped")) for item in checks),
    }


def run_doctor(
    *,
    config_path: str | Path | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    config_override_root: str | Path | None = None,
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    check_models: bool = True,
    check_render_capabilities: bool = True,
) -> dict[str, Any]:
    """Return a complete, JSON-compatible, read-only diagnostic report.

    Configuration loading uses ``materialize_models=False`` so model names and
    roots are merged and validated without invoking ModelZoo download
    resolution. Model manifests and binaries are inspected only below the
    effective ``models.root``; no alternate root is searched.
    """

    checks: list[dict[str, Any]] = []
    if not isinstance(check_models, bool):
        check_models_error: dict[str, Any] | None = {
            "received_type": type(check_models).__name__,
        }
        check_models = True
    else:
        check_models_error = None
    if not isinstance(check_render_capabilities, bool):
        render_capabilities_error: dict[str, Any] | None = {
            "received_type": type(check_render_capabilities).__name__,
        }
        check_render_capabilities = True
    else:
        render_capabilities_error = None
    try:
        environment = _environment_report(checks)
    except Exception as error:
        environment = {"error": _error_details(error)}
        _add_check(
            checks,
            "doctor.internal.environment",
            False,
            "Environment diagnostics failed unexpectedly",
            details=_error_details(error),
        )
    if check_models_error is not None:
        _add_check(
            checks,
            "models.check_requested",
            False,
            "check_models must be boolean",
            details=check_models_error,
        )
    if render_capabilities_error is not None:
        _add_check(
            checks,
            "runtime.render_capabilities_requested",
            False,
            "check_render_capabilities must be boolean",
            details=render_capabilities_error,
        )
    effective_config: dict[str, Any] | None = None
    selected_fallback: Any = None
    model_root_fallback: Any = DEFAULT_INSIGHTFACE_ROOT
    requested_provider_fallback: Any = None
    config_report: dict[str, Any]
    if config_path is None:
        if config_overrides is not None or config_override_root is not None:
            _add_check(
                checks,
                "config.load",
                False,
                "Configuration overrides require config_path",
                details={
                    "overrides_provided": config_overrides is not None,
                    "override_root": config_override_root,
                },
            )
            config_report = {
                "provided": False,
                "valid": False,
                "error": "config_path is required with overrides",
            }
        else:
            _skipped_check(
                checks,
                "config.load",
                "No configuration was supplied",
            )
            config_report = {"provided": False, "skipped": True, "valid": None}
    else:
        source = Path(config_path).expanduser().resolve()
        config_report = {
            "provided": True,
            "path": str(source),
            "valid": False,
            "materialize_models": False,
        }
        try:
            kwargs: dict[str, Any] = {
                "config_overrides": config_overrides,
                "materialize_models": False,
            }
            if config_override_root is not None:
                kwargs["config_override_root"] = config_override_root
            effective_config = load_config(source, **kwargs)
            models = effective_config.get("models", {})
            runtime = effective_config.get("runtime", {})
            recognition = effective_config.get("recognition", {})
            scan = effective_config.get("scan", {})
            video_output = effective_config.get("render", {}).get(
                "video_output",
                {},
            )
            selected_fallback = (
                models.get("name") if isinstance(models, Mapping) else None
            )
            model_root_fallback = (
                models.get("root", DEFAULT_INSIGHTFACE_ROOT)
                if isinstance(models, Mapping)
                else DEFAULT_INSIGHTFACE_ROOT
            )
            requested_provider_fallback = (
                runtime.get("provider") if isinstance(runtime, Mapping) else None
            )
            config_report.update(
                {
                    "valid": True,
                    "model_package": selected_fallback,
                    "model_root": model_root_fallback,
                    "runtime_provider": requested_provider_fallback,
                    "resolved_provider": (
                        runtime.get("resolved_provider")
                        if isinstance(runtime, Mapping)
                        else None
                    ),
                    "max_analysis_fps": (
                        scan.get("max_analysis_fps")
                        if isinstance(scan, Mapping)
                        else None
                    ),
                    "recognition_mode": (
                        recognition.get("mode")
                        if isinstance(recognition, Mapping)
                        else None
                    ),
                    "video_output": (
                        {
                            "backend": video_output.get("backend"),
                            "encoder": video_output.get("encoder"),
                            "pixel_format": video_output.get("pixel_format"),
                        }
                        if isinstance(video_output, Mapping)
                        else None
                    ),
                }
            )
            _add_check(
                checks,
                "config.load",
                True,
                "Configuration merged and validated without materializing models",
                details=config_report,
            )
        except Exception as error:
            try:
                selected_fallback = _peek_config_field(source, ("models", "name"))
                model_root_fallback = _peek_config_field(
                    source,
                    ("models", "root"),
                ) or DEFAULT_INSIGHTFACE_ROOT
                if isinstance(config_overrides, Mapping) and "models.root" in config_overrides:
                    model_root_fallback = config_overrides["models.root"]
                requested_provider_fallback = _peek_config_field(
                    source,
                    ("runtime", "provider"),
                )
            except Exception:
                pass
            config_report.update(_error_details(error))
            config_report["model_package"] = selected_fallback
            config_report["model_root"] = model_root_fallback
            config_report["runtime_provider"] = requested_provider_fallback
            _add_check(
                checks,
                "config.load",
                False,
                "Configuration could not be merged and validated offline",
                details=config_report,
            )

    try:
        runtime = _runtime_report(
            checks,
            effective_config,
            requested_fallback=requested_provider_fallback,
        )
    except Exception as error:
        runtime = {"error": _error_details(error)}
        _add_check(
            checks,
            "doctor.internal.runtime",
            False,
            "Runtime diagnostics failed unexpectedly",
            details=_error_details(error),
        )
    try:
        pyav, av_module = _pyav_report(
            checks,
            effective_config,
            check_render_capabilities=check_render_capabilities,
            input_path=input_path,
            target_modes=("redacted",),
        )
    except Exception as error:
        pyav = {"installed": False, "error": _error_details(error)}
        av_module = None
        _add_check(
            checks,
            "doctor.internal.pyav",
            False,
            "PyAV diagnostics failed unexpectedly",
            details=_error_details(error),
        )
    runtime["pyav"] = pyav
    try:
        models = _models_report(
            checks,
            effective_config,
            selected_fallback=selected_fallback,
            root_fallback=model_root_fallback,
            config_supplied=config_path is not None,
            check_models=check_models,
        )
    except Exception as error:
        models = {
            "root": str(Path(str(model_root_fallback)).expanduser().resolve() / "models"),
            "selected_package": selected_fallback,
            "checked": bool(check_models),
            "error": _error_details(error),
        }
        _add_check(
            checks,
            "doctor.internal.models",
            False,
            "Model-package diagnostics failed unexpectedly",
            details=_error_details(error),
        )
    try:
        media = _media_report(
            checks,
            input_path,
            effective_config,
            av_module,
            check_render_capabilities=check_render_capabilities,
        )
    except Exception as error:
        media = {
            "provided": input_path is not None,
            "error": _error_details(error),
        }
        _add_check(
            checks,
            "doctor.internal.media",
            False,
            "Media diagnostics failed unexpectedly",
            details=_error_details(error),
        )
    try:
        output = _output_report(checks, output_dir, input_path)
    except Exception as error:
        output = {
            "provided": output_dir is not None,
            "error": _error_details(error),
        }
        _add_check(
            checks,
            "doctor.internal.output",
            False,
            "Output-path diagnostics failed unexpectedly",
            details=_error_details(error),
        )
    safety = {
        "read_only": True,
        "scope": "doctor checks after Python dependency imports",
        "model_downloads": False,
        "onnx_sessions_created": False,
        "coreml_compilation": False,
        "warmup": False,
        "files_or_directories_created": False,
        "dependency_import_side_effects_controlled": False,
        "input_frames_decoded": bool(media.get("first_frame_decoded", False)),
        "input_probe": (
            "container_headers_and_first_video_frame"
            if input_path is not None
            else "skipped"
        ),
        "output_writability_probe": (
            "os.access_only" if output_dir is not None else "skipped"
        ),
    }
    _add_check(
        checks,
        "doctor.read_only_contract",
        True,
        "Doctor checks completed without downloading models, creating Sessions, compiling, warming up, or writing PrivateFrame files",
        details=safety,
    )
    summary = _summary(checks)
    internal_failure = any(
        not bool(item.get("ok"))
        and str(item.get("name", "")).startswith("doctor.internal.")
        for item in checks
    )
    report = {
        "ok": not internal_failure,
        "ready": summary["errors"] == 0,
        "checks": checks,
        "summary": summary,
        "environment": environment,
        "config": config_report,
        "runtime": runtime,
        "models": models,
        "media": media,
        "output": output,
        "safety": safety,
    }
    return _json_safe(report)


def diagnose_render_settings(
    settings: Mapping[str, Any],
    *,
    input_audio_codec: str | None = None,
    input_audio_present: bool | None = None,
    input_path: str | Path | None = None,
    target_modes: Sequence[str] = ("redacted",),
) -> dict[str, Any]:
    """Check capabilities for already-merged render settings.

    ``input_audio_present=None`` means that no source was inspected, so source
    audio compatibility is reported as skipped rather than guessed.
    """

    checks: list[dict[str, Any]] = []
    if not isinstance(settings, Mapping):
        _add_check(
            checks,
            "runtime.render_settings",
            False,
            "Effective render settings must be a mapping",
            details={"received_type": type(settings).__name__},
        )
        pyav = {"installed": False, "capabilities": {}}
    else:
        try:
            pyav, _av_module = _pyav_report(
                checks,
                {"render": {"video_output": settings.get("video_output", {})}},
                input_path=input_path,
                input_audio_present=input_audio_present,
                target_modes=target_modes,
            )
            if input_audio_present is None:
                _skipped_check(
                    checks,
                    "runtime.audio_input",
                    "No source audio metadata was supplied for compatibility checking",
                )
            else:
                video_output = settings.get("video_output")
                audio_settings = (
                    video_output.get("audio", {})
                    if isinstance(video_output, Mapping)
                    else {}
                )
                copy_requested = bool(
                    isinstance(audio_settings, Mapping)
                    and any(
                        audio_settings.get(str(target)) == "copy"
                        for target in target_modes
                    )
                )
                if copy_requested and bool(input_audio_present):
                    if input_path is not None and _av_module is not None:
                        copy_capability: Mapping[str, Any] | None = (
                            _mp4_audio_copy_capability(_av_module, input_path)
                        )
                    else:
                        copy_capability = {
                            "available": False,
                            "error": "input_path and PyAV are required for MP4 copy probing",
                        }
                else:
                    copy_capability = None
                _audio_output_checks(
                    checks,
                    video_output,
                    input_audio_codec=input_audio_codec,
                    input_audio_present=bool(input_audio_present),
                    target_modes=target_modes,
                    name_root="runtime.audio_input",
                    copy_capability=copy_capability,
                )
        except Exception as error:
            pyav = {"installed": False, "error": _error_details(error)}
            _add_check(
                checks,
                "doctor.internal.effective_render",
                False,
                "Effective render capability diagnostics failed unexpectedly",
                details=_error_details(error),
            )
    summary = _summary(checks)
    return _json_safe(
        {
            "ok": summary["errors"] == 0,
            "ready": summary["errors"] == 0,
            "checks": checks,
            "summary": summary,
            "runtime": {"pyav": pyav},
        }
    )


diagnose_privateframe = run_doctor


__all__ = [
    "diagnose_privateframe",
    "diagnose_render_settings",
    "run_doctor",
]
