"""Stdlib-only bootstrap for the optional InsightFace PrivateFrame CLI."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from typing import Any


def _command_hint(argv: list[str]) -> str:
    for token in argv:
        if token in {"analyze", "render", "process", "describe", "doctor"}:
            return token
        if token == "--version":
            return "version"
    return "unknown"


def _error_payload(argv: list[str], exc: BaseException) -> dict[str, Any]:
    dependency = getattr(exc, "name", None)
    return {
        "status_schema_version": 1,
        "ok": False,
        "command": _command_hint(argv),
        "error": {
            "code": (
                "missing_dependency"
                if isinstance(exc, ModuleNotFoundError)
                else "dependency_import_failed"
            ),
            "stage": "startup",
            "type": type(exc).__name__,
            "message": str(exc),
            "retryable": False,
            "hints": [
                'Install the optional runtime with: python -m pip install "insightface[privateframe]"'
            ],
            "dependency": str(dependency) if dependency else None,
        },
    }


def _cancelled_payload(argv: list[str]) -> dict[str, Any]:
    return {
        "status_schema_version": 1,
        "ok": False,
        "command": _command_hint(argv),
        "error": {
            "code": "cancelled",
            "stage": "startup",
            "type": "KeyboardInterrupt",
            "message": "operation interrupted by user",
            "retryable": True,
            "hints": [],
        },
    }


def _emit(value: dict[str, Any]) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Set this before importing InsightFace/ONNX Runtime so a CLI diagnostic
    # never attempts to persist an ORT telemetry identifier.
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    try:
        # Third-party packages imported by InsightFace can still contain
        # legacy print calls. Keep the CLI's stdout reserved for its one JSON
        # record (or the explicit --version line).
        with contextlib.redirect_stdout(sys.stderr):
            if raw == ["--version"]:
                from insightface import __version__

                implementation = None
            else:
                from insightface.app.privateframe.cli import main as implementation
    except (ImportError, ModuleNotFoundError) as exc:
        _emit(_error_payload(raw, exc))
        return 1
    except KeyboardInterrupt:
        _emit(_cancelled_payload(raw))
        return 130
    if raw == ["--version"]:
        print(f"insightface-privateframe {__version__}", flush=True)
        return 0
    assert implementation is not None
    try:
        return int(implementation(raw) or 0)
    except KeyboardInterrupt:
        _emit(_cancelled_payload(raw))
        return 130


__all__ = ["main"]
