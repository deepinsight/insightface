"""CLI for artifact-first streaming analysis and deterministic rendering."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import math
import os
import sys
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ... import __version__
from .base_config import DEFAULT_CONFIG_PATH, validate_config_override_paths
from .cli_contract import build_describe_payload
from .config import load_config
from .output_paths import default_output_paths
from .video import paths_are_distinct

_DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_PATH
RESULT_FILENAME = "result.privateframe.json"
_DOTTED_CONFIG_HELP = (
    "Any public YAML setting may also be overridden as "
    "--section.field VALUE (for example, --scan.max_analysis_fps 15). "
    "Values use YAML/JSON types; existing list items use numeric segments. "
    "Use 'describe' for common options and the bundled full configuration reference."
)
_DOTTED_RENDER_HELP = (
    "Public render.* YAML settings may be overridden as dotted options. "
    "Dotted values are applied after --render-config."
)

_STATUS_SCHEMA_VERSION = 1
_PROGRESS_CHOICES = ("auto", "text", "jsonl", "none")


def analyze_streaming_pipeline(**kwargs: Any) -> dict[str, Any]:
    from .pipeline import analyze_streaming_pipeline as implementation

    return implementation(**kwargs)


def render_streaming_artifacts(**kwargs: Any) -> dict[str, Any]:
    from .pipeline import render_streaming_artifacts as implementation

    return implementation(**kwargs)


def run_streaming_pipeline(**kwargs: Any) -> dict[str, Any]:
    from .pipeline import run_streaming_pipeline as implementation

    return implementation(**kwargs)


def _render_settings(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], str]:
    from .pipeline import _render_settings as implementation

    return implementation(*args, **kwargs)


def validate_result_document(value: Any) -> dict[str, Any]:
    from .pipeline import validate_result_document as implementation

    return implementation(value)


@contextlib.contextmanager
def _recognition_logs(progress_format: str):
    """Expose reference import diagnostics without changing the host logger."""

    progress_format = _resolved_progress_mode(progress_format)
    class Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if progress_format == "jsonl":
                value = json.dumps(
                    {"log_schema_version": 1, "event": "log",
                     "level": record.levelname.lower(), "stage": "recognition",
                     "message": record.getMessage()},
                    ensure_ascii=False, separators=(",", ":"),
                )
            else:
                value = f"[{record.levelname.lower()}] {record.getMessage()}"
            print(value, file=sys.stderr, flush=True)

    handler = Handler()
    configured = []
    for name in ("recognition", "streaming"):
        logger = logging.getLogger(f"{__package__}.{name}")
        configured.append((logger, logger.level, logger.propagate))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    try:
        yield
    finally:
        for logger, level, propagate in configured:
            logger.removeHandler(handler)
            logger.setLevel(level)
            logger.propagate = propagate
        handler.close()


class _CLIUsageError(ValueError):
    """Argument error raised instead of letting argparse print unstructured text."""


class _OutputBusyError(RuntimeError):
    """Another cooperating PrivateFrame CLI invocation owns an output target."""


class _MachineArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _CLIUsageError(message)


def _execution_options(value: argparse.ArgumentParser) -> None:
    value.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the resolved plan without inference, downloads, or writes",
    )
    value.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of existing public output artifacts",
    )
    value.add_argument(
        "--progress",
        choices=_PROGRESS_CHOICES,
        default="auto",
        help=(
            "progress on stderr: auto selects text for a terminal and JSONL "
            "otherwise; stdout always remains one final JSON object"
        ),
    )


def _result(value: argparse.ArgumentParser) -> None:
    value.add_argument(
        "--result",
        "--json-output",
        dest="result",
        help="analysis result JSON path",
    )


def _input(value: argparse.ArgumentParser, *, required: bool) -> None:
    value.add_argument(
        "--input",
        required=required,
        help="source video path; the source is read but never modified",
    )


def _output_dir(
    value: argparse.ArgumentParser,
    *,
    diagnostic: bool = False,
) -> None:
    value.add_argument(
        "--output-dir",
        help=(
            "prospective output directory to inspect without creating it"
            if diagnostic
            else (
                "directory for stable <input>_privateframe.json/.mp4 output names; "
                "also supplies a private runtime work directory"
            )
        ),
    )


def _workdir(value: argparse.ArgumentParser) -> None:
    value.add_argument(
        "--workdir",
        help=(
            "private runtime/audit directory; its encoded-packet SQLite cache "
            "is removed after analysis (normally derived from --output-dir)"
        ),
    )


def _video_output(value: argparse.ArgumentParser) -> None:
    value.add_argument(
        "--redacted",
        "--video-output",
        dest="redacted",
        help="redacted result video path",
    )


def _render_config(value: argparse.ArgumentParser) -> None:
    value.add_argument(
        "--render-config",
        help=(
            "YAML overrides for redaction style and/or video encoding; "
            "dotted render.* options take precedence"
        ),
    )


def _analysis_config(value: argparse.ArgumentParser) -> None:
    value.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG_PATH),
        metavar="PATH",
        help=(
            "custom analysis YAML overlay (default: bundled configs/base.yaml; "
            "custom files inherit it unless base_config selects another parent)"
        ),
    )


def command_parser(*, machine_errors: bool = False) -> argparse.ArgumentParser:
    parser_class = _MachineArgumentParser if machine_errors else argparse.ArgumentParser
    value = parser_class(
        prog="insightface-privateframe",
        allow_abbrev=False,
        description=(
            "Automatically blur faces or add mosaic to a video before sharing or "
            "publishing it, to help protect the privacy of people who appear. "
            "Blur all detected faces, or use reference photos to set rules for "
            "specific people. Videos are processed locally, producing a new video "
            "while keeping the original unchanged."
        ),
    )
    value.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = value.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser(
        "analyze",
        allow_abbrev=False,
        help="detect and track face regions and write reusable analysis JSON only",
        description=(
            "Detect and track face regions in a source video and write reusable "
            "analysis JSON without rendering video."
        ),
        epilog=_DOTTED_CONFIG_HELP,
    )
    _analysis_config(analyze)
    _input(analyze, required=True)
    _workdir(analyze)
    _output_dir(analyze)
    _result(analyze)
    _execution_options(analyze)

    render = commands.add_parser(
        "render",
        allow_abbrev=False,
        help="render face blur or mosaic from existing analysis JSON",
        description=(
            "Render face regions with Gaussian blur or mosaic from an existing or "
            "edited PrivateFrame result JSON without rerunning models."
        ),
        epilog=_DOTTED_RENDER_HELP,
    )
    _input(render, required=True)
    _workdir(render)
    _output_dir(render)
    _result(render)
    render.add_argument("--debug", help=argparse.SUPPRESS)
    _video_output(render)
    _render_config(render)
    _execution_options(render)

    process = commands.add_parser(
        "process",
        allow_abbrev=False,
        help="detect and track faces, write JSON, and render blur or mosaic",
        description=(
            "Detect and track face regions, write reusable analysis JSON, then "
            "render the paired privacy-redacted video with Gaussian blur or mosaic."
        ),
        epilog=_DOTTED_CONFIG_HELP,
    )
    _analysis_config(process)
    _input(process, required=True)
    _workdir(process)
    _output_dir(process)
    process.add_argument("--debug", help=argparse.SUPPRESS)
    _video_output(process)
    _result(process)
    _render_config(process)
    _execution_options(process)

    commands.add_parser(
        "describe",
        allow_abbrev=False,
        help="describe workflows, common configuration, and machine output contracts",
        description=(
            "Explain what PrivateFrame does and print its recommended workflows, "
            "command selection, inputs, outputs, common and intermediate configuration, "
            "status, and error contracts without inspecting the runtime environment. "
            "The JSON links to the bundled full configuration reference for advanced settings."
        ),
    )

    doctor = commands.add_parser(
        "doctor",
        allow_abbrev=False,
        help="run read-only runtime, model, codec, input, and output checks",
        description=(
            "Inspect whether this environment is ready for PrivateFrame without "
            "downloading models, creating inference Sessions, compiling CoreML, "
            "warming models, or writing files."
        ),
        epilog=_DOTTED_CONFIG_HELP,
    )
    _analysis_config(doctor)
    _input(doctor, required=False)
    _output_dir(doctor, diagnostic=True)
    return value


def _reject_overlapping_config_paths(paths: list[str]) -> None:
    parts: list[tuple[str, tuple[str, ...]]] = []
    for path in paths:
        tokens = tuple(path.split("."))
        if not path or any(not token for token in tokens):
            raise ValueError(f"invalid configuration override path: {path}")
        parts.append((path, tokens))
    for index, (left_path, left) in enumerate(parts):
        for right_path, right in parts[index + 1 :]:
            shortest = min(len(left), len(right))
            if left[:shortest] == right[:shortest]:
                raise ValueError(
                    "configuration override paths overlap: "
                    f"{left_path} and {right_path}"
                )


def _validate_json_config_value(
    value: object,
    *,
    path: str,
    active_containers: set[int] | None = None,
) -> None:
    active = set() if active_containers is None else active_containers
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"configuration override --{path} must be finite")
        return
    if isinstance(value, (list, dict)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"configuration override --{path} must not be cyclic")
        active.add(identity)
        if isinstance(value, list):
            for item in value:
                _validate_json_config_value(
                    item,
                    path=path,
                    active_containers=active,
                )
        else:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"configuration override --{path} mapping keys must be strings"
                    )
                _validate_json_config_value(
                    item,
                    path=path,
                    active_containers=active,
                )
        active.remove(identity)
        return
    raise ValueError(
        f"configuration override --{path} must use JSON-compatible YAML types"
    )


def _parse_dotted_config_overrides(
    argv: list[str],
) -> tuple[list[str], dict[str, object]]:
    clean: list[str] = []
    parsed: list[tuple[str, object]] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        option, separator, attached = token.partition("=")
        if option.startswith("--") and "." in option[2:]:
            path = option[2:]
            if separator:
                raw_value = attached
            else:
                if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                    raise ValueError(
                        f"configuration override --{path} requires a value"
                    )
                index += 1
                raw_value = argv[index]
            try:
                value = yaml.safe_load(raw_value)
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"invalid YAML value for configuration override --{path}: {exc}"
                ) from exc
            _validate_json_config_value(value, path=path)
            parsed.append((path, value))
        else:
            clean.append(token)
        index += 1
    _reject_overlapping_config_paths([path for path, _value in parsed])
    validate_config_override_paths([path for path, _value in parsed])
    return clean, dict(parsed)


def _status_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _status_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_status_json_safe(item) for item in value]
    return str(value)


def _emit(value: Mapping[str, Any]) -> None:
    """Write the one-record machine status contract to stdout."""

    print(
        json.dumps(
            _status_json_safe(dict(value)),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )


def _command_from_argv(argv: list[str]) -> str:
    for token in argv:
        if token in {"analyze", "render", "process", "describe", "doctor"}:
            return token
        if token == "--version":
            return "version"
    return "unknown"


def _error_code(exc: BaseException, *, stage: str) -> str:
    if isinstance(exc, _CLIUsageError):
        return "invalid_arguments"
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        return "missing_dependency"
    if isinstance(exc, _OutputBusyError):
        return "output_busy"
    if isinstance(exc, FileExistsError):
        return "output_exists"
    if isinstance(exc, FileNotFoundError):
        return "file_not_found"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, (KeyboardInterrupt, InterruptedError)):
        return "cancelled"
    message = str(exc).casefold()
    if "provider" in message and ("unavailable" in message or "available=" in message):
        return "provider_unavailable"
    if any(
        token in message
        for token in ("video", "decode", "codec", "source_video", "result json")
    ):
        return "media_error"
    if (
        isinstance(exc, (TypeError, ValueError, yaml.YAMLError))
        or "config" in message
        or "yaml" in message
        or "setting" in message
    ):
        return "invalid_config"
    return "operation_failed"


def _error_status(
    command: str,
    exc: BaseException,
    *,
    stage: str,
) -> dict[str, Any]:
    code = _error_code(exc, stage=stage)
    message = (
        "operation interrupted by user"
        if isinstance(exc, KeyboardInterrupt)
        else str(exc)
    )
    hints: list[str] = []
    if code == "output_exists":
        hints.append(
            "Pass --overwrite only after confirming the existing artifact may be replaced."
        )
    elif code == "output_busy":
        hints.append(
            "Wait for the other PrivateFrame invocation to finish, then retry."
        )
    elif code == "missing_dependency":
        hints.append(
            'Install the optional runtime with: python -m pip install "insightface[privateframe]"'
        )
    elif code == "provider_unavailable":
        hints.append(
            "Run insightface-privateframe doctor to inspect available providers."
        )
    elif code == "file_not_found":
        hints.append(
            "Check the reported path or run doctor with the same input and configuration."
        )
    return {
        "status_schema_version": _STATUS_SCHEMA_VERSION,
        "ok": False,
        "command": command,
        "error": {
            "code": code,
            "stage": stage,
            "type": (
                "ArgumentError"
                if isinstance(exc, _CLIUsageError)
                else type(exc).__name__
            ),
            "message": message,
            "retryable": isinstance(
                exc,
                (_OutputBusyError, KeyboardInterrupt, InterruptedError, OSError),
            )
            and not isinstance(
                exc,
                (FileNotFoundError, FileExistsError, PermissionError),
            ),
            "hints": hints,
        },
    }


def _resolved_result_path(args: argparse.Namespace) -> Path | None:
    result = getattr(args, "result", None)
    if result is not None:
        return Path(result).expanduser().resolve()
    workdir = getattr(args, "workdir", None)
    if workdir is not None:
        return (Path(workdir).expanduser().resolve() / RESULT_FILENAME).resolve()
    return None


def _resolved_artifacts(args: argparse.Namespace) -> dict[str, str | None]:
    if args.command not in {"analyze", "render", "process"}:
        return {}
    result = _resolved_result_path(args)
    video = getattr(args, "redacted", None)
    return {
        "result_json": str(result) if result is not None else None,
        "result_video": (
            str(Path(video).expanduser().resolve()) if video is not None else None
        ),
    }


def _public_output_targets(args: argparse.Namespace) -> list[Path]:
    artifacts = _resolved_artifacts(args)
    values: list[Path] = []
    if args.command in {"analyze", "process"} and artifacts.get("result_json"):
        values.append(Path(str(artifacts["result_json"])))
    if args.command in {"render", "process"} and artifacts.get("result_video"):
        values.append(Path(str(artifacts["result_video"])))
    debug = getattr(args, "debug", None)
    if args.command in {"render", "process"} and debug is not None:
        values.append(Path(debug).expanduser().resolve())
    return values


def _protect_existing_outputs(args: argparse.Namespace) -> None:
    if bool(getattr(args, "overwrite", False)):
        return
    existing = [path for path in _public_output_targets(args) if path.exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to replace existing output artifact(s): {formatted}"
        )


def _output_lock_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.privateframe.lock")


def _workdir_lock_path(workdir: Path) -> Path:
    return workdir.parent / f".{workdir.name}.privateframe-work.lock"


def _lock_resources(args: argparse.Namespace) -> list[tuple[Path, Path, str]]:
    resources = [
        (target, _output_lock_path(target), "output")
        for target in _public_output_targets(args)
    ]
    workdir_value = getattr(args, "workdir", None)
    if workdir_value is not None:
        workdir = Path(workdir_value).expanduser().resolve()
        resources.append((workdir, _workdir_lock_path(workdir), "workdir"))
    unique: dict[Path, tuple[Path, Path, str]] = {}
    for resource in resources:
        unique.setdefault(resource[1].resolve(), resource)
    return sorted(unique.values(), key=lambda item: str(item[1]))


def _lock_owner_pid(lock: Path) -> int | None:
    try:
        first_line = lock.read_text(encoding="ascii", errors="ignore").splitlines()[0]
    except (OSError, IndexError):
        return None
    if not first_line.startswith("pid="):
        return None
    try:
        pid = int(first_line.removeprefix("pid="))
    except ValueError:
        return None
    return pid if pid > 0 else None


def _pid_is_running(pid: int) -> bool | None:
    if pid == os.getpid():
        return True
    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return None
        return True
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI/users
        try:
            import ctypes

            process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if process:
                ctypes.windll.kernel32.CloseHandle(process)
                return True
            return ctypes.get_last_error() == 5
        except Exception:
            return None
    return None


def _lock_state(lock: Path) -> dict[str, Any]:
    owner_pid = _lock_owner_pid(lock)
    running = _pid_is_running(owner_pid) if owner_pid is not None else None
    stale = running is False
    if owner_pid is None:
        try:
            stale = time.time() - lock.stat().st_mtime > 60.0
        except OSError:
            stale = False
    return {
        "path": str(lock),
        "owner_pid": owner_pid,
        "owner_running": running,
        "stale": stale,
    }


def _create_output_lock(resource: Path, lock: Path) -> int:
    for attempt in range(2):
        try:
            descriptor = os.open(
                lock,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as exc:
            state = _lock_state(lock)
            if attempt == 0 and state["stale"]:
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass
                continue
            raise _OutputBusyError(
                "resource is already claimed by another PrivateFrame "
                f"invocation: {resource} (lock: {lock}, "
                f"owner_pid: {state['owner_pid']})"
            ) from exc
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        except Exception:
            os.close(descriptor)
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
            raise
        return descriptor
    raise AssertionError("unreachable output-lock acquisition state")


@contextlib.contextmanager
def _claim_output_targets(args: argparse.Namespace):
    """Serialize cooperating writers for public targets and runtime workdir."""

    claimed: list[Path] = []
    created_directories: set[Path] = set()
    try:
        for resource, lock, _kind in _lock_resources(args):
            current = lock.parent
            while not current.exists() and current != current.parent:
                created_directories.add(current)
                current = current.parent
            lock.parent.mkdir(parents=True, exist_ok=True)
            descriptor = _create_output_lock(resource, lock)
            try:
                os.fsync(descriptor)
            except Exception:
                os.close(descriptor)
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass
                raise
            os.close(descriptor)
            claimed.append(lock)
        # Recheck after every lock is held so two PrivateFrame CLI writers
        # cannot both pass the no-overwrite existence check.
        _protect_existing_outputs(args)
        yield
    finally:
        for lock in reversed(claimed):
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
        for directory in sorted(
            created_directories,
            key=lambda value: len(value.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass


def _resolved_progress_mode(value: str) -> str:
    if value != "auto":
        return value
    return "text" if sys.stderr.isatty() else "jsonl"


def _progress_callback(
    selected: str,
) -> Callable[[int, int, str], None] | None:
    mode = _resolved_progress_mode(selected)
    if mode == "none":
        return None
    last: tuple[str, int, int] | None = None
    last_emitted_at = 0.0

    def report(current: int, total: int, phase: str) -> None:
        nonlocal last, last_emitted_at
        current_value = int(current)
        total_value = int(total)
        phase_value = str(phase)
        percentage = (
            max(0.0, min(100.0, current_value * 100.0 / total_value))
            if total_value > 0
            else None
        )
        bucket = int(percentage) if percentage is not None else current_value
        marker = (phase_value, bucket, total_value)
        now = time.monotonic()
        terminal = total_value > 0 and current_value >= total_value
        if (
            last is not None
            and marker == last
            and not terminal
            and now - last_emitted_at < 1.0
        ):
            return
        last = marker
        last_emitted_at = now
        if mode == "jsonl":
            event = {
                "progress_schema_version": 1,
                "event": "progress",
                "phase": phase_value,
                "current": current_value,
                "total": total_value,
                "percent": percentage,
            }
            print(
                json.dumps(event, ensure_ascii=False, separators=(",", ":")),
                file=sys.stderr,
                flush=True,
            )
            return
        suffix = f" {percentage:.1f}%" if percentage is not None else ""
        print(
            f"[privateframe/{phase_value}] {current_value}/{total_value}{suffix}",
            file=sys.stderr,
            flush=True,
        )

    return report


def _success_status(
    command: str,
    args: argparse.Namespace,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    def selected_fields(
        value: object,
        fields: tuple[tuple[str, str], ...],
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        return {
            public_name: value[source_name]
            for public_name, source_name in fields
            if source_name in value
        }

    def timing(value: object, *names: str) -> Any:
        if not isinstance(value, Mapping):
            return None
        for name in names:
            if name in value:
                return value[name]
        return None

    def present_timings(**values: Any) -> dict[str, Any]:
        return {name: value for name, value in values.items() if value is not None}

    if command == "analyze":
        provider = result.get("provider")
        raw_timings = result.get("timings", {})
        total_seconds = timing(
            raw_timings,
            "total_seconds",
            "seconds",
            "analysis_seconds",
        )
        timings = present_timings(total_seconds=total_seconds)
        summary = selected_fields(
            result,
            (
                ("frame_count", "frame_count"),
                ("face_tracks", "accepted_tracks"),
                ("face_regions", "observations"),
            ),
        )
    elif command == "process":
        analysis = result.get("analysis", {})
        render = result.get("render", {})
        provider = analysis.get("provider") if isinstance(analysis, Mapping) else None
        raw_analysis_timings = (
            analysis.get("timings", {}) if isinstance(analysis, Mapping) else {}
        )
        analysis_seconds = timing(
            raw_analysis_timings,
            "total_seconds",
            "analysis_seconds",
            "seconds",
        )
        render_seconds = timing(render, "seconds")
        total_seconds = (
            analysis_seconds + render_seconds
            if isinstance(analysis_seconds, (int, float))
            and not isinstance(analysis_seconds, bool)
            and isinstance(render_seconds, (int, float))
            and not isinstance(render_seconds, bool)
            else None
        )
        timings = present_timings(total_seconds=total_seconds)
        summary = selected_fields(
            analysis,
            (
                ("frame_count", "frame_count"),
                ("face_tracks", "accepted_tracks"),
                ("face_regions", "observations"),
            ),
        )
        if isinstance(render, Mapping):
            for public_name, source_name in (
                ("frame_count", "frame_count"),
                ("face_regions", "observations"),
            ):
                if public_name not in summary and source_name in render:
                    summary[public_name] = render[source_name]
            summary.update(
                selected_fields(
                    render,
                    (
                        ("redacted_face_regions", "blurred_observations"),
                        ("kept_face_regions", "kept_observations"),
                    ),
                )
            )
    else:
        provider = None
        render_seconds = result.get("seconds")
        timings = present_timings(total_seconds=render_seconds)
        summary = selected_fields(
            result,
            (
                ("frame_count", "frame_count"),
                ("face_regions", "observations"),
                ("redacted_face_regions", "blurred_observations"),
                ("kept_face_regions", "kept_observations"),
            ),
        )
    return {
        "status_schema_version": _STATUS_SCHEMA_VERSION,
        "ok": True,
        "command": command,
        "artifacts": _resolved_artifacts(args),
        "runtime": {"provider": provider},
        "timings": timings,
        "summary": summary,
    }


def _resolved_plan(
    args: argparse.Namespace,
    config_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    input_value = getattr(args, "input", None)
    workdir = getattr(args, "workdir", None)
    config = getattr(args, "config", None)
    render_config = getattr(args, "render_config", None)
    return {
        "config": (
            str(Path(config).expanduser().resolve()) if config is not None else None
        ),
        "input": (
            str(Path(input_value).expanduser().resolve())
            if input_value is not None
            else None
        ),
        "output_dir": getattr(args, "output_dir", None),
        "workdir": (
            str(Path(workdir).expanduser().resolve()) if workdir is not None else None
        ),
        "artifacts": _resolved_artifacts(args),
        "render_config": (
            str(Path(render_config).expanduser().resolve())
            if render_config is not None
            else None
        ),
        "config_overrides": dict(config_overrides),
        "overwrite": bool(getattr(args, "overwrite", False)),
        "progress": getattr(args, "progress", None),
    }


def _add_preflight_check(
    report: dict[str, Any],
    *,
    name: str,
    ok: bool,
    message: str,
    severity: str = "error",
    details: Mapping[str, Any] | None = None,
) -> None:
    checks = report.setdefault("checks", [])
    if not isinstance(checks, list):
        raise TypeError("doctor checks must be an array")
    checks.append(
        {
            "name": name,
            "ok": bool(ok),
            "severity": "info" if ok else severity,
            "message": message,
            "details": dict(details or {}),
        }
    )
    if not ok and severity == "error":
        report["ready"] = False


def _check_render_result(
    report: dict[str, Any],
    args: argparse.Namespace,
    config_overrides: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = _resolved_result_path(args)
    if path is None:
        _add_preflight_check(
            report,
            name="render_result",
            ok=False,
            message="render result JSON path is unresolved",
        )
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        validate_result_document(value)
        metadata = value["source_video"]["metadata"]
        media_report = report.get("media", {})
        inspected_metadata = (
            media_report.get("privateframe_metadata")
            if isinstance(media_report, Mapping)
            else None
        )
        if isinstance(inspected_metadata, Mapping):
            # Width/height/FPS come from the same lightweight probe used by
            # analysis. Frame count is intentionally excluded: analysis
            # replaces unreliable container counts with the decoded count.
            for field in ("width", "height"):
                if int(metadata[field]) != int(inspected_metadata[field]):
                    raise ValueError(
                        f"source_video.metadata.{field} does not match the input video"
                    )
            if not math.isclose(
                float(metadata["fps"]),
                float(inspected_metadata["fps"]),
                rel_tol=0.01,
                abs_tol=0.01,
            ):
                raise ValueError(
                    "source_video.metadata.fps does not match the input video"
                )
        settings, settings_sha256 = _render_settings(
            value,
            getattr(args, "render_config", None),
            config_overrides,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic records every failure
        _add_preflight_check(
            report,
            name="render_result",
            ok=False,
            message=f"result JSON is not renderable: {path}: {exc}",
            details={"path": str(path)},
        )
        return None
    _add_preflight_check(
        report,
        name="render_result",
        ok=True,
        message="result JSON and effective render settings are supported",
        details={"path": str(path), "render_settings_sha256": settings_sha256},
    )
    return settings


def _check_process_render_settings(
    report: dict[str, Any],
    args: argparse.Namespace,
    config_overrides: Mapping[str, Any],
) -> dict[str, Any] | None:
    render_overrides = {
        path: value
        for path, value in config_overrides.items()
        if path.startswith("render.")
    }
    analysis_overrides = {
        path: value
        for path, value in config_overrides.items()
        if not path.startswith("render.")
    }
    try:
        config_kwargs: dict[str, Any] = {"materialize_models": False}
        if analysis_overrides:
            config_kwargs.update(
                {
                    "config_overrides": analysis_overrides,
                    "config_override_root": Path.cwd(),
                }
            )
        config = load_config(args.config, **config_kwargs)
        render_defaults = deepcopy(config["render"])
        recognition = config.get("recognition", {})
        mode = str(recognition.get("mode", "all"))
        render_defaults["recognition_policy"] = {
            "mode": mode,
            "unknown_action": recognition.get("unknown_action", "auto"),
        }
        synthetic_result = {
            "render_defaults": render_defaults,
            "recognition": (
                {"enabled": False, "reason": "policy_all"}
                if mode == "all"
                else {
                    "enabled": True,
                    # Dry-run validates settings without decoding reference
                    # photos. Real execution validates the full reference set.
                    "references": {"files": [{"file": "dry-run-placeholder"}]},
                    "tracks": {},
                }
            ),
        }
        settings, settings_sha256 = _render_settings(
            synthetic_result,
            getattr(args, "render_config", None),
            render_overrides or None,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic records every failure
        _add_preflight_check(
            report,
            name="render_settings",
            ok=False,
            message=f"effective render settings are invalid: {exc}",
            details={
                "render_config": getattr(args, "render_config", None),
                "error_type": type(exc).__name__,
            },
        )
        return None
    _add_preflight_check(
        report,
        name="render_settings",
        ok=True,
        message="effective render settings are valid",
        details={"render_settings_sha256": settings_sha256},
    )
    return settings


def _append_effective_render_capabilities(
    report: dict[str, Any],
    settings: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    from .doctor import diagnose_render_settings

    media = report.get("media", {})
    audio_stream = media.get("audio_stream") if isinstance(media, Mapping) else None
    input_audio_present: bool | None = (
        isinstance(audio_stream, Mapping)
        if isinstance(media, Mapping) and media.get("first_frame_decoded") is True
        else None
    )
    input_audio_codec = (
        str(audio_stream.get("codec"))
        if isinstance(audio_stream, Mapping) and audio_stream.get("codec") is not None
        else None
    )
    target_modes = [
        name
        for name, path in (
            ("debug", getattr(args, "debug", None)),
            ("redacted", getattr(args, "redacted", None)),
        )
        if path is not None
    ]
    capability_report = diagnose_render_settings(
        settings,
        input_audio_codec=input_audio_codec,
        input_audio_present=input_audio_present,
        input_path=getattr(args, "input", None),
        target_modes=target_modes,
    )
    checks = capability_report.get("checks", [])
    if not isinstance(checks, list):
        raise TypeError("effective render capability checks must be an array")
    destination = report.setdefault("checks", [])
    if not isinstance(destination, list):
        raise TypeError("doctor checks must be an array")
    for check in checks:
        item = dict(check)
        item["name"] = f"effective_render.{item.get('name', 'unknown')}"
        destination.append(item)
    runtime = report.setdefault("runtime", {})
    if isinstance(runtime, dict):
        runtime["effective_render"] = capability_report.get("runtime", {})


def _apply_overwrite_readiness(
    report: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    existing = [path for path in _public_output_targets(args) if path.exists()]
    if not existing:
        _add_preflight_check(
            report,
            name="output_conflicts",
            ok=True,
            message="no public output artifact would be replaced",
        )
        return
    allowed = bool(getattr(args, "overwrite", False))
    paths = [str(path) for path in existing]
    _add_preflight_check(
        report,
        name="output_conflicts",
        ok=allowed,
        message=(
            "existing output artifacts may be replaced because --overwrite is enabled"
            if allowed
            else "existing output artifacts require --overwrite: " + ", ".join(paths)
        ),
        severity="error",
        details={"paths": paths, "overwrite": allowed},
    )


def _apply_lock_readiness(
    report: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    states: list[dict[str, Any]] = []
    for resource, lock, kind in _lock_resources(args):
        if not lock.exists():
            continue
        state = _lock_state(lock)
        state.update({"resource": str(resource), "kind": kind})
        states.append(state)
    active = [state for state in states if not state["stale"]]
    stale = [state for state in states if state["stale"]]
    _add_preflight_check(
        report,
        name="resource_locks",
        ok=not active,
        message=(
            "no output or workdir resource is owned by another invocation"
            if not active and not stale
            else (
                "only stale resource locks were found; execution will reclaim them"
                if not active
                else "one or more output/workdir resources are currently busy"
            )
        ),
        details={"active": active, "stale": stale},
    )


def _nearest_existing_path(path: Path) -> Path | None:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current if current.exists() else None


def _path_is_creatable(path: Path) -> tuple[bool, Path | None]:
    parent = path.parent
    existing = _nearest_existing_path(parent)
    usable = bool(
        existing is not None
        and existing.is_dir()
        and os.access(existing, os.W_OK | os.X_OK)
    )
    return usable, existing


def _check_planned_paths(report: dict[str, Any], args: argparse.Namespace) -> None:
    source = Path(args.input).expanduser().resolve()
    result = _resolved_result_path(args)
    video = getattr(args, "redacted", None)
    debug = getattr(args, "debug", None)
    workdir_value = getattr(args, "workdir", None)
    compared = [source]
    if result is not None:
        compared.append(result)
    if video is not None:
        compared.append(Path(video).expanduser().resolve())
    if debug is not None:
        compared.append(Path(debug).expanduser().resolve())
    if workdir_value is not None:
        compared.append(Path(workdir_value).expanduser().resolve())
    distinct = paths_are_distinct(compared)
    _add_preflight_check(
        report,
        name="paths.distinct",
        ok=distinct,
        message=(
            "input, result JSON, video, and workdir paths are distinct"
            if distinct
            else "input, result JSON, video, and workdir paths must be distinct"
        ),
        details={"paths": [str(path) for path in compared]},
    )
    targets = _public_output_targets(args)
    target_details: list[dict[str, Any]] = []
    targets_ok = True
    for target in targets:
        creatable, existing_parent = _path_is_creatable(target)
        target_is_directory = target.exists() and target.is_dir()
        valid = creatable and not target_is_directory
        targets_ok = targets_ok and valid
        target_details.append(
            {
                "path": str(target),
                "exists": target.exists(),
                "is_directory": target_is_directory,
                "nearest_existing_parent": (
                    str(existing_parent) if existing_parent is not None else None
                ),
                "parent_writable": creatable,
            }
        )
    _add_preflight_check(
        report,
        name="paths.output_parents",
        ok=targets_ok,
        message=(
            "all explicit output targets have writable parent paths"
            if targets_ok
            else "one or more explicit output targets cannot be created"
        ),
        details={"targets": target_details},
    )

    if workdir_value is None:
        return
    workdir = Path(workdir_value).expanduser().resolve()
    if workdir.exists():
        workdir_ok = workdir.is_dir() and os.access(workdir, os.W_OK | os.X_OK)
        existing_parent = workdir
    else:
        workdir_ok, existing_parent = _path_is_creatable(workdir / ".probe")
    _add_preflight_check(
        report,
        name="paths.workdir",
        ok=workdir_ok,
        message=(
            "workdir is usable"
            if workdir_ok
            else "workdir is not a writable directory and cannot be created"
        ),
        details={
            "path": str(workdir),
            "exists": workdir.exists(),
            "nearest_existing_parent": (
                str(existing_parent) if existing_parent is not None else None
            ),
        },
    )


def _refresh_diagnostic_summary(report: dict[str, Any]) -> None:
    checks = report.get("checks", [])
    if not isinstance(checks, list):
        raise TypeError("doctor checks must be an array")
    summary = {
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
    report["summary"] = summary
    report["ready"] = summary["errors"] == 0


def _dry_run_status(
    args: argparse.Namespace,
    config_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    from .doctor import run_doctor

    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        targets = _public_output_targets(args)
        if targets:
            output_dir = str(targets[0].parent)
    report = run_doctor(
        config_path=getattr(args, "config", _DEFAULT_CONFIG_PATH),
        config_overrides=config_overrides,
        config_override_root=Path.cwd(),
        input_path=getattr(args, "input", None),
        output_dir=output_dir,
        check_models=args.command != "render",
        # Render/process are checked again below using their final merged
        # --render-config and render.* dotted overrides. Analysis needs no
        # encoder at all.
        check_render_capabilities=False,
    )
    if not isinstance(report, dict):
        raise TypeError("doctor returned a non-object report")
    report = dict(report)
    report.setdefault("ready", bool(report.get("ok", False)))
    _check_planned_paths(report, args)
    effective_render_settings: dict[str, Any] | None = None
    if args.command == "render":
        effective_render_settings = _check_render_result(
            report,
            args,
            config_overrides,
        )
    elif args.command == "process":
        effective_render_settings = _check_process_render_settings(
            report,
            args,
            config_overrides,
        )
    if effective_render_settings is not None:
        _append_effective_render_capabilities(
            report,
            effective_render_settings,
            args,
        )
    _apply_overwrite_readiness(report, args)
    _apply_lock_readiness(report, args)
    _refresh_diagnostic_summary(report)
    diagnostic_ok = bool(report.get("ok", True))
    report["ready"] = diagnostic_ok and bool(report.get("ready", False))
    payload = {
        "status_schema_version": _STATUS_SCHEMA_VERSION,
        "ok": diagnostic_ok,
        "command": args.command,
        "dry_run": True,
        "ready": bool(report.get("ready", False)),
        "plan": _resolved_plan(args, config_overrides),
        "checks": report.get("checks", []),
        "diagnostics": report,
    }
    if not diagnostic_ok:
        payload["error"] = _error_status(
            args.command,
            RuntimeError("Preflight diagnostics failed internally; inspect checks for details"),
            stage="preflight",
        )["error"]
    return payload


def _apply_output_defaults(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Resolve the simplified output-directory form without changing legacy paths."""

    if args.command == "doctor":
        if args.output_dir is not None:
            args.output_dir = str(Path(args.output_dir).expanduser().resolve())
        return
    if args.output_dir is None:
        if args.command in {"analyze", "process"} and args.workdir is None:
            parser.error(
                f"{args.command} requires --workdir unless --output-dir is used"
            )
    else:
        paths = default_output_paths(args.input, args.output_dir)
        args.output_dir = str(paths.output_dir)
        if args.workdir is None:
            args.workdir = str(paths.workdir)
        if args.result is None:
            args.result = str(paths.result_json)
        if args.command in {"process", "render"} and args.redacted is None:
            args.redacted = str(paths.result_video)

    if args.command in {"render", "process"}:
        if args.debug is None and args.redacted is None:
            parser.error(f"{args.command} requires a video output")
    if args.command == "render" and args.result is None and args.workdir is None:
        parser.error(
            "render requires --result unless --output-dir or --workdir is used"
        )


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    command_hint = _command_from_argv(raw)
    parser = command_parser(machine_errors=True)
    try:
        clean, config_overrides = _parse_dotted_config_overrides(raw)
        args = parser.parse_args(clean)
        command_hint = args.command
        if config_overrides and args.command == "render":
            invalid = sorted(
                path for path in config_overrides if not path.startswith("render.")
            )
            if invalid:
                parser.error("render only accepts render.* configuration overrides")
        if config_overrides and args.command == "describe":
            parser.error("describe does not accept configuration overrides")
        if args.command in {"analyze", "render", "process", "doctor"}:
            _apply_output_defaults(args, parser)
    except (TypeError, ValueError) as exc:
        usage_error = (
            exc if isinstance(exc, _CLIUsageError) else _CLIUsageError(str(exc))
        )
        _emit(_error_status(command_hint, usage_error, stage="arguments"))
        return 2
    except KeyboardInterrupt as exc:
        _emit(_error_status(command_hint, exc, stage="arguments"))
        return 130

    if args.command == "describe":
        try:
            _emit(build_describe_payload(command_parser()))
            return 0
        except Exception as exc:  # noqa: BLE001 - machine error envelope
            _emit(_error_status("describe", exc, stage="describe"))
            return 1
        except KeyboardInterrupt as exc:
            _emit(_error_status("describe", exc, stage="describe"))
            return 130

    if args.command == "doctor":
        try:
            from .doctor import run_doctor

            with contextlib.redirect_stdout(sys.stderr):
                report = run_doctor(
                    config_path=args.config,
                    config_overrides=config_overrides,
                    config_override_root=Path.cwd(),
                    input_path=args.input,
                    output_dir=args.output_dir,
                )
            payload = dict(report)
            diagnostic_ok = bool(payload.get("ok", False))
            payload.update(
                {
                    "status_schema_version": _STATUS_SCHEMA_VERSION,
                    "ok": diagnostic_ok,
                    "command": "doctor",
                }
            )
            _emit(payload)
            return 0 if diagnostic_ok else 1
        except Exception as exc:  # noqa: BLE001 - machine error envelope
            _emit(_error_status("doctor", exc, stage="doctor"))
            return 1
        except KeyboardInterrupt as exc:
            _emit(_error_status("doctor", exc, stage="doctor"))
            return 130

    if args.dry_run:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                status = _dry_run_status(args, config_overrides)
            _emit(status)
            return 0 if bool(status.get("ok", False)) else 1
        except Exception as exc:  # noqa: BLE001 - machine error envelope
            _emit(_error_status(args.command, exc, stage="preflight"))
            return 1
        except KeyboardInterrupt as exc:
            _emit(_error_status(args.command, exc, stage="preflight"))
            return 130

    command = args.command
    try:
        progress = _progress_callback(args.progress)
        # InsightFace's older model-loading helpers still emit informational
        # ``print`` calls.  Treat those as diagnostics so stdout remains the
        # single status record even while models download or Sessions prepare.
        with _claim_output_targets(args), _recognition_logs(args.progress):
            with contextlib.redirect_stdout(sys.stderr):
                if command == "analyze":
                    result = analyze_streaming_pipeline(
                        config_path=args.config,
                        input_path=args.input,
                        workdir=args.workdir,
                        result_path=args.result,
                        config_overrides=config_overrides,
                        config_override_root=Path.cwd(),
                        progress=progress,
                    )
                elif command == "render":
                    result = render_streaming_artifacts(
                        input_path=args.input,
                        workdir=args.workdir,
                        result_path=args.result,
                        debug_path=args.debug,
                        redacted_path=args.redacted,
                        render_config=args.render_config,
                        config_overrides=config_overrides,
                        progress=progress,
                    )
                else:
                    result = run_streaming_pipeline(
                        config_path=args.config,
                        input_path=args.input,
                        debug_path=args.debug,
                        redacted_path=args.redacted,
                        render_config=args.render_config,
                        workdir=args.workdir,
                        result_path=args.result,
                        config_overrides=config_overrides,
                        config_override_root=Path.cwd(),
                        progress=progress,
                    )
    except Exception as exc:  # noqa: BLE001
        stage = (
            "preflight"
            if isinstance(
                exc,
                (
                    _OutputBusyError,
                    FileExistsError,
                    FileNotFoundError,
                    PermissionError,
                ),
            )
            else command
        )
        _emit(_error_status(command, exc, stage=stage))
        return 1
    except KeyboardInterrupt as exc:
        _emit(_error_status(command, exc, stage=command))
        return 130
    # Native PyAV/FFmpeg wrappers may become collectible only after the public
    # render function has returned.  Reclaim them before reporting success and
    # before a short-lived CLI interpreter starts tearing native libraries down.
    gc.collect()
    _emit(_success_status(args.command, args, result))
    return 0


__all__ = ["command_parser", "main"]
