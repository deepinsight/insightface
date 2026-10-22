from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from insightface.app.privateframe import pipeline
from insightface.app.privateframe.packet_cache import EncodedPacketCache


_CACHE_FILENAMES = (
    "encoded-packets.sqlite",
    "encoded-packets.sqlite-wal",
    "encoded-packets.sqlite-shm",
)


@pytest.mark.parametrize(
    "first_error",
    [
        InterruptedError("cancelled by test"),
        RuntimeError("analysis failed by test"),
    ],
    ids=("cancel", "failure"),
)
def test_analysis_cleans_runtime_cache_and_retries_same_workdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    first_error: Exception,
) -> None:
    workdir = tmp_path / "fixed-workdir"
    workdir.mkdir()
    # Simulate a cache left by an older process that did not have the
    # entry-point finalizer.
    for filename in _CACHE_FILENAMES:
        (workdir / filename).write_bytes(b"stale")
    audit = workdir / "summary.streaming-onnx.json"
    user_output = workdir / "customer-output.mp4"
    similarly_named = workdir / "encoded-packets.sqlite.backup"
    audit.write_text("audit", encoding="utf-8")
    user_output.write_bytes(b"video")
    similarly_named.write_bytes(b"keep")

    attempts = 0

    def progress(_current: int, _total: int, _message: str) -> None:
        return None

    def is_cancelled() -> bool:
        return False

    def analyze(**kwargs: Any) -> dict[str, int]:
        nonlocal attempts
        attempts += 1
        assert kwargs["workdir"] == workdir.resolve()
        assert kwargs["progress"] is progress
        assert kwargs["is_cancelled"] is is_cancelled
        assert all(not (workdir / filename).exists() for filename in _CACHE_FILENAMES)
        for filename in _CACHE_FILENAMES:
            (workdir / filename).write_bytes(f"attempt-{attempts}".encode())
        if attempts == 1:
            raise first_error
        return {"attempt": attempts}

    monkeypatch.setattr(pipeline, "_analyze_streaming_pipeline_impl", analyze)
    kwargs = {
        "config_path": tmp_path / "config.yaml",
        "input_path": tmp_path / "input.mp4",
        "workdir": workdir,
        "progress": progress,
        "is_cancelled": is_cancelled,
    }

    with pytest.raises(type(first_error), match="test"):
        pipeline.analyze_streaming_pipeline(**kwargs)

    assert all(not (workdir / filename).exists() for filename in _CACHE_FILENAMES)
    assert audit.read_text(encoding="utf-8") == "audit"
    assert user_output.read_bytes() == b"video"
    assert similarly_named.read_bytes() == b"keep"

    assert pipeline.analyze_streaming_pipeline(**kwargs) == {"attempt": 2}
    assert all(not (workdir / filename).exists() for filename in _CACHE_FILENAMES)
    assert audit.exists()
    assert user_output.exists()
    assert similarly_named.exists()


def test_packet_cache_close_is_idempotent_and_removes_only_runtime_files(
    tmp_path: Path,
) -> None:
    class Connection:
        commits = 0
        closes = 0

        def commit(self) -> None:
            self.commits += 1

        def close(self) -> None:
            self.closes += 1

    workdir = tmp_path / "work"
    workdir.mkdir()
    path = workdir / "encoded-packets.sqlite"
    for filename in _CACHE_FILENAMES:
        (workdir / filename).write_bytes(b"runtime")
    preserved = workdir / "encoded-packets.sqlite.backup"
    preserved.write_bytes(b"preserved")

    cache = object.__new__(EncodedPacketCache)
    cache.path = path
    cache.connection = Connection()
    cache._closed = False
    cache._pending_since_commit = 1

    cache.close()
    cache.close()

    assert cache.connection.commits == 1
    assert cache.connection.closes == 1
    assert all(not (workdir / filename).exists() for filename in _CACHE_FILENAMES)
    assert preserved.read_bytes() == b"preserved"


def test_packet_cache_close_releases_and_cleans_when_commit_fails(
    tmp_path: Path,
) -> None:
    class Connection:
        closed = False

        def commit(self) -> None:
            raise RuntimeError("commit failed by test")

        def close(self) -> None:
            self.closed = True

    path = tmp_path / "encoded-packets.sqlite"
    for filename in _CACHE_FILENAMES:
        (tmp_path / filename).write_bytes(b"runtime")

    cache = object.__new__(EncodedPacketCache)
    cache.path = path
    cache.connection = Connection()
    cache._closed = False
    cache._pending_since_commit = 1

    with pytest.raises(RuntimeError, match="commit failed"):
        cache.close()

    assert cache.connection.closed is True
    assert all(not (tmp_path / filename).exists() for filename in _CACHE_FILENAMES)
