from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from insightface.app.privateframe import streaming as streaming_module
from insightface.app.privateframe.packet_cache import DecodedFrameStore
from insightface.app.privateframe.streaming import StreamingEngine


def _endpoint_engine(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fps: float = 30.0,
    checkpoint_scores: dict[int, float] | None = None,
) -> tuple[
    StreamingEngine,
    list[tuple[int, int]],
    list[int],
    list[int],
]:
    """Build an endpoint runner whose observable work is deterministic."""

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.metadata = SimpleNamespace(width=160, height=120)
    engine.fps = fps
    engine.frame_stride = 2
    engine.between_scan_frames = "interpolate"
    engine.interpolate_tracking = True
    engine.config = {
        "streaming": {
            "pre_roll_corridor_expansion": 4.0,
            "max_corridor_side_pixels": 256,
            "pre_roll_decode_chunk_frames": 32,
        },
        "tracking": {
            "kalman_optical_flow": {
                "endpoint_affine_repair": {"enabled": True},
                "min_points": 4,
            }
        },
        "revalidation": {
            "policy": {
                "rule_gate": {
                    "short_track": {
                        "moderate_verifier_p50": 0.40,
                        "strong_verifier_p50": 0.80,
                    }
                }
            }
        },
    }
    engine.cache = SimpleNamespace(oldest_frame_index=lambda: 0)
    engine.endpoint_affine_candidates = []
    engine.endpoint_affine_jobs = 0
    engine.endpoint_affine_frames = 0
    engine.endpoint_verifier_checkpoints = 0
    engine.endpoint_verifier_refinement_frames = 0
    engine.endpoint_local_review_frames = {}
    engine.recognition_engine = SimpleNamespace(
        enabled=False,
        max_frames_per_track=3,
    )
    engine.recognition_candidates = {}
    engine._bidirectional_fusion_settings = lambda: {
        "max_corridor_side_pixels": 256,
        "max_materialized_bytes": 64 * 1024 * 1024,
    }

    decoded_ranges: list[tuple[int, int]] = []

    def decode_frames(
        first: int,
        last: int,
        **_kwargs: Any,
    ) -> dict[int, np.ndarray]:
        decoded_ranges.append((first, last))
        return {
            frame_idx: np.full((100, 100, 3), frame_idx, dtype=np.uint8)
            for frame_idx in range(first, last + 1)
        }

    engine._decode_frames = decode_frames
    stepped_frames: list[int] = []

    class FakeAffineState:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def step(self, image: np.ndarray) -> dict[str, Any]:
            stepped_frames.append(int(image[0, 0, 0]))
            return {
                "valid": True,
                "box": np.asarray([40.0, 30.0, 60.0, 50.0]),
                "selected": 12,
                "inliers": 10,
                "quality": 0.9,
                "affine_scale": 1.0,
                "affine_rotation_degrees": 0.0,
            }

    monkeypatch.setattr(streaming_module, "AffineEndpointState", FakeAffineState)
    verify_calls: list[int] = []

    def verify_once(
        _frame: np.ndarray,
        item: dict[str, Any],
        _box: list[float],
        **_kwargs: Any,
    ) -> float:
        frame_idx = int(item["frame_idx"])
        verify_calls.append(frame_idx)
        return float((checkpoint_scores or {}).get(frame_idx, 1.0))

    monkeypatch.setattr(engine, "_verify_once", verify_once)
    monkeypatch.setattr(
        engine,
        "_capture_tracking_recognition_candidate",
        lambda *_args, **_kwargs: None,
    )
    return engine, decoded_ranges, verify_calls, stepped_frames


def _run_sparse_endpoint(engine: StreamingEngine, *, target_frame: int) -> int:
    return engine._run_endpoint_affine(
        track_id="t00001",
        anchor_frame=0,
        anchor_box=np.asarray([40.0, 30.0, 60.0, 50.0]),
        target_frame=target_frame,
        direction=1,
        run_neural_review=True,
        sparse_neural_review=True,
    )


def test_sparse_endpoint_decode_block_is_independent_of_verifier_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, decoded_ranges, verify_calls, stepped_frames = _endpoint_engine(monkeypatch)

    emitted = _run_sparse_endpoint(engine, target_frame=24)

    assert emitted == 24
    assert decoded_ranges == [(0, 0), (1, 24)]
    assert decoded_ranges[1][1] - decoded_ranges[1][0] + 1 > 8
    assert verify_calls == [8, 16, 24]
    assert stepped_frames == list(range(1, 25))
    assert [
        int(item["frame_idx"]) for item in engine.endpoint_affine_candidates
    ] == list(range(1, 25))


def test_sparse_endpoint_failure_does_not_publish_prefetched_future_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, decoded_ranges, verify_calls, stepped_frames = _endpoint_engine(
        monkeypatch,
        checkpoint_scores={5: 0.0, 8: 0.0},
    )
    # Keep the test on boundary refinement rather than its independent
    # Local-SCRFD anomaly fallback.
    engine.endpoint_local_review_frames = {"t00001": {0}}

    emitted = _run_sparse_endpoint(engine, target_frame=24)

    assert decoded_ranges == [(0, 0), (1, 24)]
    assert stepped_frames == list(range(1, 9))
    assert verify_calls == [8, 1, 2, 3, 4, 5]
    assert emitted == 4
    assert [
        int(item["frame_idx"]) for item in engine.endpoint_affine_candidates
    ] == [1, 2, 3, 4]


def test_stride_one_dense_endpoint_review_and_publication_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, decoded_ranges, verify_calls, stepped_frames = _endpoint_engine(monkeypatch)
    engine.frame_stride = 1
    engine.between_scan_frames = "visual"
    engine.interpolate_tracking = False
    reviewed_frames: list[int] = []

    def measure_review(
        _image: np.ndarray,
        item: dict[str, Any],
        *_args: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        reviewed_frames.append(int(item["frame_idx"]))
        return engine._empty_local_review()

    monkeypatch.setattr(engine, "_measure_review", measure_review)

    emitted = engine._run_endpoint_affine(
        track_id="t00001",
        anchor_frame=0,
        anchor_box=np.asarray([40.0, 30.0, 60.0, 50.0]),
        target_frame=20,
        direction=1,
        run_neural_review=True,
        sparse_neural_review=False,
    )

    assert emitted == 20
    assert decoded_ranges == [(0, 0), (1, 20)]
    assert stepped_frames == list(range(1, 21))
    assert reviewed_frames == list(range(1, 21))
    assert verify_calls == []
    assert [
        int(item["frame_idx"]) for item in engine.endpoint_affine_candidates
    ] == list(range(1, 21))


def test_streaming_history_reuses_full_frames_across_different_crops() -> None:
    store = DecodedFrameStore(frame_target=8, byte_capacity=1024 * 1024)
    source_frames = {
        frame_idx: np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        + frame_idx
        for frame_idx in range(5)
    }
    decode_calls: list[tuple[int, int, dict[str, Any]]] = []

    class PacketCache:
        def decode_range(
            self,
            first: int,
            last: int,
            **kwargs: Any,
        ) -> dict[int, np.ndarray]:
            decode_calls.append((first, last, kwargs))
            return {
                frame_idx: source_frames[frame_idx]
                for frame_idx in range(first, last + 1)
            }

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.decoded_frame_store = store
    engine.decoded_frame_bytes = int(source_frames[0].nbytes)
    engine.cache = PacketCache()
    # Normal streaming fills the store before any historical read. Prime the
    # synthetic store likewise so its byte-bounded decode block is known.
    engine._remember_frame(0, source_frames[0])

    left = engine._decode_frames(2, 4, crop=(0, 0, 4, 3))
    calls_after_initial_decode = list(decode_calls)
    right = engine._decode_frames(2, 4, crop=(4, 3, 8, 6))

    assert calls_after_initial_decode == [(2, 4, {})]
    assert decode_calls == calls_after_initial_decode
    assert store.hits == 3
    assert store.frame_count == 4
    for frame_idx in range(2, 5):
        np.testing.assert_array_equal(left[frame_idx], source_frames[frame_idx][:3, :4])
        np.testing.assert_array_equal(right[frame_idx], source_frames[frame_idx][3:, 4:])


@pytest.mark.parametrize(
    ("frame_target", "byte_capacity", "decoded_frame_bytes"),
    [
        pytest.param(0, 1024, 144, id="retention-disabled"),
        pytest.param(8, 0, 144, id="byte-cache-disabled"),
        pytest.param(8, 143, 144, id="source-frame-exceeds-byte-cap"),
    ],
)
def test_streaming_history_preserves_crop_at_decode_when_store_cannot_retain(
    frame_target: int,
    byte_capacity: int,
    decoded_frame_bytes: int,
) -> None:
    decode_calls: list[tuple[int, int, dict[str, Any]]] = []
    expected = {
        frame_idx: np.full((3, 4, 3), frame_idx, dtype=np.uint8)
        for frame_idx in range(2, 5)
    }

    class PacketCache:
        def decode_range(
            self,
            first: int,
            last: int,
            **kwargs: Any,
        ) -> dict[int, np.ndarray]:
            decode_calls.append((first, last, kwargs))
            return expected

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.decoded_frame_store = DecodedFrameStore(
        frame_target=frame_target,
        byte_capacity=byte_capacity,
    )
    engine.decoded_frame_bytes = decoded_frame_bytes
    engine.cache = PacketCache()
    crop = (0, 0, 4, 3)

    actual = engine._decode_frames(2, 4, crop=crop)

    assert actual is expected
    assert decode_calls == [(2, 4, {"crop": crop, "crops": None})]
    assert engine.decoded_frame_store.frame_count == 0


@pytest.mark.parametrize(
    ("fps", "expected"),
    [
        pytest.param(10.0, 46, id="configured-endpoint-horizon-wins"),
        pytest.param(30.0, 61, id="two-second-history-at-30-fps"),
        pytest.param(60.0, 121, id="two-second-history-at-60-fps"),
    ],
)
def test_decoded_frame_store_null_target_tracks_longest_replay_horizon(
    fps: float,
    expected: int,
) -> None:
    config = {
        "streaming": {
            "recent_frame_cache_frames": None,
            "max_retroactive_seconds": 2.0,
        },
        "tracking": {
            "reliable_pre_roll_extension": 30,
            "reliable_endpoint_extension": 45,
            "kalman_optical_flow": {
                "bidirectional_fusion": {"max_gap_frames": 12},
            },
        },
    }

    assert streaming_module._decoded_frame_store_target(config, fps) == expected


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        pytest.param(0, 0, id="explicitly-disabled"),
        pytest.param(17, 17, id="explicit-frame-target"),
    ],
)
def test_decoded_frame_store_explicit_target_overrides_automatic_policy(
    configured: int,
    expected: int,
) -> None:
    config = {
        "streaming": {
            "recent_frame_cache_frames": configured,
            "max_retroactive_seconds": 300.0,
        },
        "tracking": {},
    }

    assert streaming_module._decoded_frame_store_target(config, 120.0) == expected
