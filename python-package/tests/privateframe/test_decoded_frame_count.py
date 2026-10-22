from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from insightface.app.privateframe import streaming
from insightface.app.privateframe.packet_cache import DecodedFrameStore
from insightface.app.privateframe.video import VideoMetadata


class _Scanner:
    def __init__(self) -> None:
        self.submitted_frames: list[int] = []
        self.closed = False

    def submit(self, frame: np.ndarray) -> int:
        frame_idx = int(frame[0, 0, 0])
        self.submitted_frames.append(frame_idx)
        return frame_idx

    def finish(self, _future: int) -> list[dict[str, Any]]:
        return []

    def close(self) -> None:
        self.closed = True


class _Cache:
    evicted_packets = 0
    evicted_bytes = 0
    historical_decode_requests = 0
    historical_packets_read = 0
    peak_decode_range_bytes = 0

    def __init__(self) -> None:
        self.committed = False

    def commit(self) -> None:
        self.committed = True

    def evict_before_frame(self, _frame_idx: int) -> None:
        return None

    def live_payload_bytes(self) -> int:
        return 0


class _SceneCutDetector:
    def observe(
        self,
        frame_idx: int,
        _frame: np.ndarray,
        _timestamp: float,
    ) -> list[dict[str, Any]]:
        return [
            {
                "frame_idx": frame_idx,
                "scene_cut_finalized": True,
                "scene_cut_from_previous": False,
                "scene_mean_absdiff": 0.0,
            }
        ]

    def flush(self) -> list[dict[str, Any]]:
        return []


def _engine(tmp_path: Path, *, reported_frame_count: int) -> streaming.StreamingEngine:
    engine = object.__new__(streaming.StreamingEngine)
    engine.source = tmp_path / "input.mp4"
    engine.metadata = VideoMetadata(
        path=str(engine.source),
        width=2,
        height=2,
        fps=25.0,
        frame_count=reported_frame_count,
        duration=reported_frame_count / 25.0,
    )
    engine.config = {
        "scan": {"pipeline_depth": 2, "max_analysis_fps": 6.25},
        "streaming": {
            "recent_frame_cache_max_bytes": 0,
            "progress_every_frames": 10_000,
            "eviction_interval_frames": 10_000,
        },
        "tracking": {
            "between_scan_frames": "interpolate",
            "fragment_stitching": {},
        },
        "recognition": {"mode": "all"},
        "render": {"box_stabilization": {}},
    }
    engine.fps = float(engine.metadata.fps)
    engine.max_analysis_fps = 6.25
    engine.analysis_fps_tolerance_multiplier = 1.05
    engine.allowed_regular_analysis_fps = 6.5625
    engine.nominal_regular_analysis_fps = 6.25
    engine.detector_frame_stride = 4
    engine.between_scan_frames = "interpolate"
    engine.interpolate_tracking = True
    engine.detector_scan_burst_frames = 1
    engine.forced_detector_scan_reasons = {}
    engine.max_retroactive_frames = 0
    engine.scanner = _Scanner()
    engine.cache = _Cache()
    engine.scene_cut_detector = _SceneCutDetector()
    engine.audits = []
    engine.detections = []
    engine.tracks = []
    engine.candidates = []
    engine.evidence = []
    engine.endpoint_affine_candidates = []
    engine.decoded_frame_store = DecodedFrameStore(0, 0)
    engine.decoded_frame_bytes = 2 * 2 * 3
    engine.detector = object()
    engine.reviewer = SimpleNamespace(detector=engine.detector)
    engine.fast_review_mode = True
    engine.local_review_stride = 4
    engine.local_review_phase = 2
    engine.local_review_attempts = 0
    engine.local_review_sampled_out = 0
    engine.local_review_forced = 0
    engine.verifier_review_calls = 0
    engine.verifier_review_cache_hits = 0
    for name in (
        "reverse_jobs",
        "reverse_frames",
        "bidirectional_gap_jobs",
        "bidirectional_gap_frames",
        "bidirectional_accepted_frames",
        "bidirectional_rejected_frames",
        "bidirectional_review_resolutions",
        "bidirectional_skipped_jobs",
        "bidirectional_association_attempts",
        "bidirectional_association_rescues",
        "long_gap_reanchors",
        "discarded_unanchored_tail_frames",
        "endpoint_affine_jobs",
        "endpoint_affine_frames",
        "endpoint_affine_published_frames",
    ):
        setattr(engine, name, 0)
    engine.bidirectional_audits = []
    engine._remember_frame = lambda _frame_idx, _frame: None
    engine.process = lambda *_args: None
    engine._capture_recognition_candidate = lambda *_args: None
    engine._finalize_recognition = lambda _aliases: {}
    engine._publish_endpoint_affine_candidates = (
        lambda observations, _evidence: observations
    )
    return engine


@pytest.mark.parametrize(
    ("reported_frame_count", "expected_endpoint_scans"),
    [
        pytest.param(84, 1, id="over-reported"),
        pytest.param(82, 2, id="under-reported"),
    ],
)
@pytest.mark.parametrize("debug_logging", [False, True])
def test_run_uses_decoded_frame_count_before_draining_eof_pending_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reported_frame_count: int,
    expected_endpoint_scans: int,
    debug_logging: bool,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    caplog.set_level(
        logging.DEBUG if debug_logging else logging.WARNING,
        logger=streaming.__name__,
    )
    actual_frame_count = 83
    engine = _engine(tmp_path, reported_frame_count=reported_frame_count)
    engine.config["streaming"]["progress_every_frames"] = 1
    closed_with: list[tuple[int, float, str]] = []
    state = SimpleNamespace(active=True)
    engine.states = [state]

    def close_state(_state: Any, *, reason: str) -> None:
        closed_with.append(
            (engine.metadata.frame_count, engine.metadata.duration, reason)
        )
        state.active = False

    engine._close = close_state
    frames = [
        (
            frame_idx,
            frame_idx / 25.0,
            np.full((2, 2, 3), frame_idx, dtype=np.uint8),
        )
        for frame_idx in range(actual_frame_count)
    ]
    monkeypatch.setattr(
        streaming,
        "iter_cached_frames",
        lambda _source, _cache: iter(frames),
    )
    monkeypatch.setattr(streaming, "_fragment_aliases", lambda *_args: {})
    monkeypatch.setattr(
        streaming,
        "_select_track_frame_candidates",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        streaming,
        "finalize_precomputed",
        lambda *_args, **_kwargs: {"observations": [], "evidence": []},
    )
    monkeypatch.setattr(
        streaming,
        "stabilize_observations",
        lambda observations, *_args, **_kwargs: observations,
    )
    monkeypatch.setattr(
        streaming,
        "_accepted_interval_coverage",
        lambda *_args, **_kwargs: (0, 0, 0),
    )
    progress: list[tuple[int, int, str]] = []

    result = engine.run(progress=lambda *update: progress.append(update))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    diagnostics = [record for record in caplog.records if record.name == streaming.__name__]
    if debug_logging:
        assert all(record.levelno == logging.DEBUG for record in diagnostics)
        assert any("live_cache=" in record.getMessage() for record in diagnostics)
        assert any("container metadata reported" in record.getMessage() for record in diagnostics)
    else:
        assert diagnostics == []
    assert engine.metadata.frame_count == actual_frame_count
    assert engine.metadata.duration == pytest.approx(actual_frame_count / 25.0)
    assert result["scan"]["frame_count"] == actual_frame_count
    assert result["scan"]["metadata"]["frame_count"] == actual_frame_count
    assert result["scan"]["metadata"]["duration"] == pytest.approx(
        actual_frame_count / 25.0
    )
    assert progress[0] == (0, reported_frame_count, "analysis")
    assert progress[-1] == (actual_frame_count, actual_frame_count, "analysis")
    assert progress.count(progress[-1]) == 1
    assert closed_with == [
        (actual_frame_count, pytest.approx(actual_frame_count / 25.0), "end_of_stream")
    ]

    # Frame 82 is off the regular stride=4 phase.  It must become the existing
    # end-of-stream scan after the decoder's real endpoint is known.
    assert (actual_frame_count - 1) % 4 != 0
    assert engine.audits[-1]["frame_idx"] == actual_frame_count - 1
    assert engine.audits[-1]["detector_scan_reason"] == "end_of_stream"
    assert engine.audits[-1]["detector_scan_performed"] is True
    assert engine.scanner.submitted_frames.count(actual_frame_count - 1) == 1
    assert (
        result["detector_sampling"]["reason_counts"]["end_of_stream"]
        == expected_endpoint_scans
    )
    expected_sampling = {
        "source_fps": 25.0,
        "max_analysis_fps": 6.25,
        "tolerance_multiplier": 1.05,
        "allowed_regular_analysis_fps": 6.5625,
        "effective_frame_stride": 4,
        "nominal_regular_analysis_fps": 6.25,
    }
    assert {
        key: result["detector_sampling"][key] for key in expected_sampling
    } == expected_sampling
    assert (
        result["local_review_sampling"]["effective_between_scan_frames"]
        == "interpolate"
    )
    assert engine.scanner.closed is True
    assert engine.cache.committed is True


def test_run_uses_the_same_geometry_fail_safe_for_every_detector_stride(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unrepairable privacy hole is unsafe with or without sampling."""

    frames = [(0, 0.0, np.zeros((2, 2, 3), dtype=np.uint8))]
    monkeypatch.setattr(
        streaming,
        "iter_cached_frames",
        lambda _source, _cache: iter(frames),
    )
    monkeypatch.setattr(streaming, "_fragment_aliases", lambda *_args: {})
    monkeypatch.setattr(
        streaming,
        "_select_track_frame_candidates",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        streaming,
        "finalize_precomputed",
        lambda *_args, **_kwargs: {"observations": [], "evidence": []},
    )
    monkeypatch.setattr(
        streaming,
        "stabilize_observations",
        lambda observations, *_args, **_kwargs: observations,
    )
    monkeypatch.setattr(
        streaming,
        "_accepted_interval_coverage",
        lambda *_args, **_kwargs: (1, 1, 0),
    )

    messages: list[str] = []
    for detector_stride in (1, 4):
        engine = _engine(tmp_path, reported_frame_count=1)
        engine.states = []
        engine.detector_frame_stride = detector_stride
        engine.max_analysis_fps = 25.0 / detector_stride
        engine.allowed_regular_analysis_fps = engine.max_analysis_fps * 1.05
        engine.nominal_regular_analysis_fps = 25.0 / detector_stride
        engine.config["scan"]["max_analysis_fps"] = engine.max_analysis_fps

        with pytest.raises(
            RuntimeError,
            match="final geometry coverage invariant failed",
        ) as error:
            engine.run()
        messages.append(str(error.value))

        assert engine.scanner.closed is True
        assert engine.cache.committed is True

    assert messages[0] == messages[1]


def test_run_rejects_a_video_that_decodes_no_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, reported_frame_count=84)
    monkeypatch.setattr(
        streaming,
        "iter_cached_frames",
        lambda _source, _cache: iter(()),
    )

    with pytest.raises(RuntimeError, match="decoder produced no frames"):
        engine.run()

    assert engine.scanner.closed is True
    assert engine.cache.committed is True


def test_forward_endpoint_repair_stops_at_the_latest_committed_frame() -> None:
    engine = object.__new__(streaming.StreamingEngine)
    engine.metadata = SimpleNamespace(frame_count=84)
    engine.audits = [{"frame_idx": frame_idx} for frame_idx in range(83)]
    engine._endpoint_affine_settings = lambda: {"max_frames": 10}
    calls: list[dict[str, Any]] = []
    engine._run_endpoint_affine = lambda **kwargs: calls.append(kwargs) or 2
    state = SimpleNamespace(
        track={"track_id": "t0001"},
        last_detection_frame=80,
        last_detection_box=np.asarray([1.0, 2.0, 3.0, 4.0]),
        pending={},
    )

    repaired = engine._repair_forward_endpoint(
        state,
        extension=10,
        reason="end_of_stream",
    )

    assert repaired == 2
    assert len(calls) == 1
    assert calls[0]["track_id"] == "t0001"
    assert calls[0]["anchor_frame"] == 80
    np.testing.assert_array_equal(
        calls[0]["anchor_box"],
        np.asarray([1.0, 2.0, 3.0, 4.0]),
    )
    assert calls[0]["target_frame"] == 82
    assert calls[0]["direction"] == 1
