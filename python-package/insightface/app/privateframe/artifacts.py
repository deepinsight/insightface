"""Atomic, reproducible analysis artifacts shared by CLI and renderers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _atomic_path(path: Path) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return descriptor, Path(name)


def write_json(path: str | Path, value: Any, *, indent: int | None = 2) -> None:
    destination = Path(path)
    descriptor, temporary = _atomic_path(destination)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=indent,
                separators=(",", ":") if indent is None else None,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl(path: str | Path, values: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    descriptor, temporary = _atomic_path(destination)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for item in values:
                exported = dict(item)
                if "box" in exported and "source_aabb" not in exported:
                    exported["source_aabb"] = exported["box"]
                stream.write(
                    json.dumps(
                        exported,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def git_version(root: str | Path) -> dict[str, Any]:
    """Best-effort development metadata; installed runtimes need not have Git."""
    repository = Path(root)

    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=repository,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


__all__ = [
    "canonical_json_bytes",
    "git_version",
    "sha256_file",
    "sha256_json",
    "write_json",
    "write_jsonl",
]
