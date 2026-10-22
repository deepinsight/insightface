from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from insightface.app.privateframe import artifact_render, pipeline, streaming
from insightface.app.privateframe.artifact_render import RenderTarget
from insightface.app.privateframe.model_catalog import DETECTION_TASK


def test_process_reports_one_combined_two_phase_progress_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def cancellation() -> bool:
        return False

    def analyze(**kwargs: Any) -> dict[str, str]:
        assert kwargs["is_cancelled"] is cancellation
        kwargs["progress"](0, 3, "ignored")
        kwargs["progress"](3, 3, "ignored")
        return {"phase": "analysis"}

    def render(**kwargs: Any) -> dict[str, str]:
        assert kwargs["is_cancelled"] is cancellation
        kwargs["progress"](0, 3, "ignored")
        kwargs["progress"](3, 3, "ignored")
        return {"phase": "render"}

    monkeypatch.setattr(pipeline, "analyze_streaming_pipeline", analyze)
    monkeypatch.setattr(pipeline, "render_streaming_artifacts", render)
    updates: list[tuple[int, int, str]] = []

    result = pipeline.run_streaming_pipeline(
        config_path=tmp_path / "config.yaml",
        input_path=tmp_path / "input.mp4",
        workdir=tmp_path / "work",
        debug_path=None,
        redacted_path=tmp_path / "output.mp4",
        progress=lambda current, total, message: updates.append(
            (current, total, message)
        ),
        is_cancelled=cancellation,
    )

    assert result == {
        "analysis": {"phase": "analysis"},
        "render": {"phase": "render"},
    }
    assert updates == [
        (0, 6, "analysis"),
        (3, 6, "analysis"),
        (3, 6, "render"),
        (6, 6, "render"),
    ]


def test_stream_reports_each_committed_frame_and_cleans_up_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Scanner:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Cache:
        committed = False

        def commit(self) -> None:
            self.committed = True

    class SceneCutDetector:
        def observe(self, frame_idx: int, _frame: Any, _timestamp: float):
            return [
                {
                    "frame_idx": frame_idx,
                    "scene_cut_finalized": True,
                    "scene_cut_from_previous": False,
                }
            ]

    engine = object.__new__(streaming.StreamingEngine)
    engine.source = tmp_path / "input.mp4"
    engine.metadata = SimpleNamespace(width=2, height=2, frame_count=1)
    engine.config = {
        "scan": {"pipeline_depth": 1},
        "streaming": {
            "recent_frame_cache_max_bytes": 0,
            "progress_every_frames": 100,
            "eviction_interval_frames": 100,
        },
    }
    engine.detector_frame_stride = 1
    engine.scanner = Scanner()
    engine.cache = Cache()
    engine.scene_cut_detector = SceneCutDetector()
    engine.audits = []
    engine.detections = []
    engine.tracks = []
    engine.states = []
    engine.forced_detector_scan_reasons = {}
    engine._detector_scan_reason = lambda _frame_idx: None
    engine._remember_frame = lambda _frame_idx, _frame: None
    engine.process = lambda *_args: None
    monkeypatch.setattr(
        streaming,
        "iter_cached_frames",
        lambda _source, _cache: iter([(0, 0.0, np.zeros((2, 2, 3), dtype=np.uint8))]),
    )
    cancelled = False
    updates: list[tuple[int, int, str]] = []

    def report(current: int, total: int, message: str) -> None:
        nonlocal cancelled
        updates.append((current, total, message))
        if current == 1:
            cancelled = True

    with pytest.raises(InterruptedError, match="cancelled"):
        engine.run(progress=report, is_cancelled=lambda: cancelled)

    assert updates == [(0, 1, "analysis"), (1, 1, "analysis")]
    assert engine.scanner.closed is True
    assert engine.cache.committed is True


def test_run_stream_forwards_callbacks_and_always_closes_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    detector = object()
    analysis = SimpleNamespace(models={DETECTION_TASK: detector})
    cleaned: list[str] = []

    def progress(_current: int, _total: int, _message: str) -> None:
        return None

    def is_cancelled() -> bool:
        return False

    class Engine:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs == {
                "progress": progress,
                "is_cancelled": is_cancelled,
            }
            raise InterruptedError("cancelled by test")

        def _clear_recognition_candidates(self) -> None:
            cleaned.append("candidates")

        def close(self) -> None:
            cleaned.append("engine")

    monkeypatch.setattr(streaming, "make_face_analysis", lambda _config: analysis)
    monkeypatch.setattr(streaming, "StreamingEngine", Engine)

    with pytest.raises(InterruptedError, match="test"):
        streaming.run_stream(
            tmp_path / "input.mp4",
            tmp_path,
            {},
            progress=progress,
            is_cancelled=is_cancelled,
        )

    assert cleaned == ["candidates", "engine"]


def test_render_reports_each_frame_and_aborts_temporary_output_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    instances: list[Any] = []

    class Writer:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.aborted = False
            self.committed = False
            instances.append(self)

        def write(self, _frame: np.ndarray) -> None:
            pass

        def finish(self) -> None:
            pass

        def commit(self) -> None:
            self.committed = True

        def abort(self) -> None:
            self.aborted = True

    frames = [
        (index, index / 30.0, index, np.zeros((2, 2, 3), dtype=np.uint8))
        for index in range(3)
    ]
    monkeypatch.setattr(artifact_render, "_PyAVWriter", Writer)
    monkeypatch.setattr(
        artifact_render,
        "iter_oriented_frames",
        lambda _source: iter(frames),
    )
    monkeypatch.setattr(
        artifact_render,
        "probe_video",
        lambda _path: SimpleNamespace(width=2, height=2, fps=30.0),
    )
    cancelled = False
    updates: list[tuple[int, int, str]] = []

    def report(current: int, total: int, message: str) -> None:
        nonlocal cancelled
        updates.append((current, total, message))
        if current == 1:
            cancelled = True

    with pytest.raises(InterruptedError, match="cancelled"):
        artifact_render.render_artifacts(
            source=tmp_path / "input.mp4",
            targets=[RenderTarget("redacted", tmp_path / "output.mp4")],
            settings={"backend": "pyav", "audio": {}},
            analysis_result={
                "source_video": {
                    "metadata": {
                        "width": 2,
                        "height": 2,
                        "fps": 30.0,
                        "frame_count": 3,
                    },
                },
                "observations": [],
            },
            progress=report,
            is_cancelled=lambda: cancelled,
        )

    assert updates == [(0, 3, "render"), (1, 3, "render")]
    assert len(instances) == 1
    assert instances[0].aborted is True
    assert instances[0].committed is False


def test_successful_pyav_render_collects_native_cycles_before_return(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class Writer:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def write(self, _frame: np.ndarray) -> None:
            pass

        def finish(self) -> None:
            events.append("finish")

        def commit(self) -> None:
            events.append("commit")

        def abort(self) -> None:
            events.append("abort")

    monkeypatch.setattr(artifact_render, "_PyAVWriter", Writer)
    monkeypatch.setattr(
        artifact_render,
        "iter_oriented_frames",
        lambda _source: iter(
            [(0, 0.0, 0, np.zeros((2, 2, 3), dtype=np.uint8))]
        ),
    )
    monkeypatch.setattr(
        artifact_render,
        "probe_video",
        lambda _path: SimpleNamespace(
            width=2,
            height=2,
            fps=30.0,
            to_dict=lambda: {"frame_count": 1},
        ),
    )
    monkeypatch.setattr(artifact_render, "sha256_file", lambda _path: "sha256")
    monkeypatch.setattr(
        artifact_render.gc,
        "collect",
        lambda: events.append("collect") or 0,
    )

    result = artifact_render.render_artifacts(
        source=tmp_path / "input.mp4",
        targets=[RenderTarget("redacted", tmp_path / "output.mp4")],
        settings={"backend": "pyav", "audio": {}},
        analysis_result={
            "source_video": {
                "metadata": {
                    "width": 2,
                    "height": 2,
                    "fps": 30.0,
                    "frame_count": 1,
                },
            },
            "observations": [],
        },
    )

    assert result["frame_count"] == 1
    assert events == ["finish", "commit", "collect"]
