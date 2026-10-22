from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from insightface.app.privateframe import base_config
from insightface.app.privateframe import streaming as streaming_module
from insightface.app.privateframe.config import load_config
from insightface.app.privateframe.streaming import ObjectState, StreamingEngine


CONFIG_PATH = Path(base_config.__file__).with_name("configs") / "base.yaml"


def _without_model_materialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base_config, "materialize_model_package", lambda _config: None)
    monkeypatch.setattr(
        base_config,
        "validate_model_package_contracts",
        lambda _config: None,
    )


@pytest.mark.parametrize("mode", ["visual", "interpolate"])
def test_between_scan_frames_accepts_supported_modes(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    _without_model_materialization(monkeypatch)

    config = load_config(
        CONFIG_PATH,
        config_overrides={"tracking.between_scan_frames": mode},
    )

    assert config["tracking"]["between_scan_frames"] == mode


def test_between_scan_frames_defaults_to_interpolate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_model_materialization(monkeypatch)

    config = load_config(CONFIG_PATH)

    assert config["tracking"]["between_scan_frames"] == "interpolate"


@pytest.mark.parametrize(
    ("mode", "stride", "expected"),
    [
        ("interpolate", 1, False),
        ("interpolate", 2, True),
        ("interpolate", 3, True),
        ("interpolate", 4, True),
        ("interpolate", 8, True),
        ("interpolate", 16, True),
        ("visual", 2, False),
        ("visual", 4, False),
        ("visual", 16, False),
    ],
)
def test_interpolate_tracking_is_effective_only_for_sampled_strides(
    mode: str,
    stride: int,
    expected: bool,
) -> None:
    assert streaming_module._interpolate_tracking_enabled(mode, stride) is expected


@pytest.mark.parametrize("mode", ["flow", "INTERPOLATE", ""])
def test_between_scan_frames_rejects_unknown_mode(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    _without_model_materialization(monkeypatch)

    with pytest.raises(ValueError, match="tracking.between_scan_frames"):
        load_config(
            CONFIG_PATH,
            config_overrides={"tracking.between_scan_frames": mode},
        )


@pytest.mark.parametrize("mode", [None, True, 1, ["interpolate"]])
def test_between_scan_frames_rejects_non_string(
    monkeypatch: pytest.MonkeyPatch,
    mode: object,
) -> None:
    _without_model_materialization(monkeypatch)

    with pytest.raises(TypeError, match="tracking.between_scan_frames"):
        load_config(
            CONFIG_PATH,
            config_overrides={"tracking.between_scan_frames": mode},
        )


def _interpolation_engine(*, width: int = 100, height: int = 80) -> StreamingEngine:
    engine = StreamingEngine.__new__(StreamingEngine)
    engine.metadata = SimpleNamespace(width=width, height=height)
    engine.between_scan_frames = "interpolate"
    engine.interpolate_tracking = True
    engine.candidates = []
    engine.evidence = []
    engine.interpolation_jobs = 0
    engine.interpolated_frames = 0
    return engine


def _state(
    *,
    left_frame: int = 0,
    left_box: list[float] | None = None,
    pending: dict[int, dict[str, Any]] | None = None,
) -> ObjectState:
    return ObjectState(
        track={"track_id": "t00001"},
        # Pure interpolation deliberately does not consult optical flow. A
        # sentinel-free None makes any accidental use fail loudly in the test.
        flow=None,  # type: ignore[arg-type]
        last_detection_frame=left_frame,
        last_detection_box=np.asarray(
            left_box or [10.0, 10.0, 30.0, 30.0],
            dtype=np.float64,
        ),
        pending={} if pending is None else pending,
    )


def _by_frame(items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(item["frame_idx"]): item for item in items}


def _endpoint_affine_engine(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fps: float,
    scores: dict[int, float] | None = None,
) -> tuple[StreamingEngine, list[tuple[int, int]]]:
    engine = _interpolation_engine()
    engine.fps = fps
    engine.config = {
        "streaming": {
            "pre_roll_corridor_expansion": 4.0,
            "max_corridor_side_pixels": 256,
            "pre_roll_decode_chunk_frames": 3,
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
        "max_materialized_bytes": 16 * 1024 * 1024,
    }
    engine._decode_frames = lambda first, last, **_kwargs: {
        frame_idx: np.zeros((32, 32, 3), dtype=np.uint8)
        for frame_idx in range(first, last + 1)
    }

    class FakeAffineState:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def step(self, _frame: np.ndarray) -> dict[str, Any]:
            return {
                "valid": True,
                "box": np.asarray([10.0, 10.0, 30.0, 30.0]),
                "selected": 12,
                "inliers": 10,
                "quality": 0.9,
                "affine_scale": 1.0,
                "affine_rotation_degrees": 0.0,
            }

    monkeypatch.setattr(streaming_module, "AffineEndpointState", FakeAffineState)
    verify_calls: list[tuple[int, int]] = []

    def verify_once(
        _frame: np.ndarray,
        item: dict[str, Any],
        _box: list[float],
        **_kwargs: Any,
    ) -> float:
        verify_calls.append(
            (
                int(item["frame_idx"]),
                len(engine.endpoint_affine_candidates),
            )
        )
        return float((scores or {}).get(int(item["frame_idx"]), 1.0))

    monkeypatch.setattr(engine, "_verify_once", verify_once)
    monkeypatch.setattr(
        engine,
        "_capture_tracking_recognition_candidate",
        lambda *_args, **_kwargs: None,
    )
    return engine, verify_calls


def test_detector_anchor_interpolation_uses_linear_center_and_log_size() -> None:
    engine = _interpolation_engine()
    state = _state(pending={1: {"visual_path_must_not_be_reused": True}})
    right_box = np.asarray([50.0, 30.0, 90.0, 90.0], dtype=np.float64)

    engine._finish_pending_interpolation(
        state,
        anchor_frame=4,
        anchor_box=right_box,
    )

    assert [item["frame_idx"] for item in engine.candidates] == [1, 2, 3]
    assert len(engine.evidence) == 3
    assert engine.interpolation_jobs == 1
    assert engine.interpolated_frames == 3
    assert state.pending == {}

    midpoint = _by_frame(engine.candidates)[2]
    midpoint_box = np.asarray(midpoint["box"], dtype=np.float64)
    midpoint_center = (midpoint_box[:2] + midpoint_box[2:]) * 0.5
    midpoint_size = midpoint_box[2:] - midpoint_box[:2]
    np.testing.assert_allclose(midpoint_center, [45.0, 40.0])
    np.testing.assert_allclose(
        midpoint_size,
        [np.sqrt(20.0 * 40.0), np.sqrt(20.0 * 60.0)],
    )

    assert midpoint["track_id"] == "t00001"
    assert midpoint["source"] == "kalman_optical_flow"
    assert midpoint["geometry_source"] == "detector_anchor_interpolation"
    assert midpoint["flow_continuity"] == "detector_anchor_interpolation"
    assert midpoint["reduced_assurance"] is True
    assert midpoint["interpolation_left_frame"] == 0
    assert midpoint["interpolation_right_frame"] == 4
    assert midpoint["interpolation_fraction"] == pytest.approx(0.5)

    midpoint_evidence = _by_frame(engine.evidence)[2]
    assert midpoint_evidence["box"] == midpoint["box"]
    assert midpoint_evidence["source"] == "kalman_optical_flow"
    assert midpoint_evidence["geometry_source"] == "detector_anchor_interpolation"
    assert midpoint_evidence["reduced_assurance"] is True
    assert midpoint_evidence["local_match_count"] == -1
    assert midpoint_evidence["local_confidence"] is None
    assert midpoint_evidence["verifier_face_probability"] is None


def test_detector_anchor_interpolation_clips_each_published_box() -> None:
    engine = _interpolation_engine(width=64, height=48)
    state = _state(left_box=[-24.0, -16.0, 16.0, 24.0])

    engine._finish_pending_interpolation(
        state,
        anchor_frame=4,
        anchor_box=np.asarray([48.0, 32.0, 88.0, 72.0], dtype=np.float64),
    )

    candidates = _by_frame(engine.candidates)
    np.testing.assert_allclose(np.asarray(candidates[1]["box"])[:2], [0.0, 0.0])
    np.testing.assert_allclose(np.asarray(candidates[3]["box"])[2:], [64.0, 48.0])
    for item in engine.candidates:
        x1, y1, x2, y2 = item["box"]
        assert 0.0 <= x1 <= x2 <= 64.0
        assert 0.0 <= y1 <= y2 <= 48.0


def test_interpolate_finish_without_right_anchor_discards_pending() -> None:
    engine = _interpolation_engine()
    state = _state(pending={1: {"box": [0.0, 0.0, 1.0, 1.0]}})

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("visual decode/review path must not run")

    engine._decode_pending = forbidden  # type: ignore[method-assign]
    engine._decode_frames = forbidden  # type: ignore[method-assign]
    engine._review = forbidden  # type: ignore[method-assign]
    engine._finish_pending_consensus = forbidden  # type: ignore[method-assign]

    engine._finish_pending(state)

    assert state.pending == {}
    assert engine.candidates == []
    assert engine.evidence == []
    assert engine.interpolation_jobs == 0
    assert engine.interpolated_frames == 0


def test_interpolate_finish_with_right_anchor_bypasses_decode_and_review() -> None:
    engine = _interpolation_engine()
    state = _state(pending={1: {"box": [0.0, 0.0, 1.0, 1.0]}})

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("visual decode/review path must not run")

    engine._decode_pending = forbidden  # type: ignore[method-assign]
    engine._decode_frames = forbidden  # type: ignore[method-assign]
    engine._review = forbidden  # type: ignore[method-assign]
    engine._finish_pending_consensus = forbidden  # type: ignore[method-assign]

    engine._finish_pending(
        state,
        anchor_frame=2,
        anchor_box=np.asarray([30.0, 10.0, 50.0, 30.0], dtype=np.float64),
    )

    assert [item["frame_idx"] for item in engine.candidates] == [1]
    assert len(engine.evidence) == 1
    assert state.pending == {}
    assert engine.interpolation_jobs == 1
    assert engine.interpolated_frames == 1


def test_process_interpolation_skips_flow_and_rescue_and_associates_from_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _interpolation_engine()
    engine.settings = {"max_missed_frames": 100}
    engine.recognition_engine = SimpleNamespace(enabled=False)

    class ForbiddenFlow:
        box = np.asarray([70.0, 60.0, 90.0, 80.0], dtype=np.float64)

        @staticmethod
        def step(_frame: np.ndarray) -> dict[str, Any]:
            raise AssertionError("active-state optical flow must not run")

    anchor_box = np.asarray([10.0, 10.0, 30.0, 30.0], dtype=np.float64)
    state = ObjectState(
        track={"track_id": "t00001"},
        flow=ForbiddenFlow(),  # type: ignore[arg-type]
        last_detection_frame=0,
        last_detection_box=anchor_box,
    )
    engine.states = [state]

    association_calls: list[dict[str, Any]] = []

    def association_score(
        observed_state: ObjectState,
        _detection: dict[str, Any],
        observed_frame_idx: int,
        _settings: dict[str, Any],
        *,
        reference_box: np.ndarray | None = None,
        allow_long_gap_flow: bool = True,
    ) -> None:
        association_calls.append(
            {
                "state": observed_state,
                "frame_idx": observed_frame_idx,
                "reference_box": reference_box,
                "allow_long_gap_flow": allow_long_gap_flow,
            }
        )
        return None

    def empty_review(
        _frame: np.ndarray,
        _item: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return {}

    def no_refinement(
        _item: dict[str, Any],
        _review: dict[str, Any],
        _reference: np.ndarray | None,
    ) -> bool:
        return False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("bidirectional association rescue must not run")

    new_states: list[int] = []
    monkeypatch.setattr(streaming_module, "_association_score", association_score)
    monkeypatch.setattr(engine, "_measure_review", empty_review)
    monkeypatch.setattr(engine, "_refine_detection_geometry", no_refinement)
    monkeypatch.setattr(engine, "_bidirectional_anchor_association_score", forbidden)
    monkeypatch.setattr(
        engine,
        "_new_state",
        lambda frame_idx, _frame, _detection, **_kwargs: new_states.append(frame_idx),
    )

    engine.process(
        2,
        np.zeros((8, 8, 3), dtype=np.uint8),
        [{"frame_idx": 2, "source": "detector", "box": [12.0, 10.0, 32.0, 30.0]}],
        False,
    )

    assert len(association_calls) == 1
    call = association_calls[0]
    assert call["state"] is state
    assert call["frame_idx"] == 2
    assert call["reference_box"] is anchor_box
    assert call["allow_long_gap_flow"] is False
    assert new_states == [2]


def test_process_interpolation_forces_local_review_on_detector_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _interpolation_engine()
    engine.states = []
    engine.settings = {"max_missed_frames": 100}
    engine.recognition_engine = SimpleNamespace(enabled=False)
    review_calls: list[dict[str, Any]] = []

    def measured_review(
        _frame: np.ndarray,
        _item: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        review_calls.append(kwargs)
        return {}

    monkeypatch.setattr(engine, "_measure_review", measured_review)
    monkeypatch.setattr(
        engine,
        "_refine_detection_geometry",
        lambda _item, _review, _reference: False,
    )
    monkeypatch.setattr(
        engine,
        "_new_state",
        lambda _frame_idx, _frame, _detection, **_kwargs: None,
    )

    engine.process(
        4,
        np.zeros((8, 8, 3), dtype=np.uint8),
        [{"frame_idx": 4, "source": "detector", "box": [1.0, 1.0, 5.0, 5.0]}],
        False,
    )

    assert review_calls == [
        {
            "force_local": True,
            "local_review_reason": "interpolation_anchor",
        }
    ]


@pytest.mark.parametrize(
    ("close_reason", "boundary_frame", "last_audit_frame", "expected_boundary"),
    [
        ("natural", None, 18, 19),
        ("scene_cut", 20, 20, 20),
        ("end_of_stream", None, 99, 100),
    ],
)
def test_interpolate_close_routes_every_endpoint_through_the_same_repair(
    monkeypatch: pytest.MonkeyPatch,
    close_reason: str,
    boundary_frame: int | None,
    last_audit_frame: int,
    expected_boundary: int,
) -> None:
    engine = _interpolation_engine()
    engine.audits = [{"frame_idx": last_audit_frame}]
    engine.settings = {
        "endpoint_extension": 8,
        "reliable_endpoint_extension": 45,
        "reliable_endpoint_min_detector_frames": 20,
    }
    engine.discarded_unanchored_tail_frames = 0
    state = _state(
        left_frame=10,
        pending={11: {"box": [11.0, 10.0, 31.0, 30.0]}},
    )
    state.track["detections"] = [
        {"frame_idx": 10, "box": [10.0, 10.0, 30.0, 30.0]}
    ]
    calls: list[dict[str, Any]] = []

    def repair_endpoint(
        observed_state: ObjectState,
        *,
        boundary_frame_exclusive: int,
        close_reason: str,
    ) -> int:
        calls.append(
            {
                "state": observed_state,
                "boundary_frame_exclusive": boundary_frame_exclusive,
                "close_reason": close_reason,
            }
        )
        return 0

    monkeypatch.setattr(engine, "_repair_interpolate_endpoint", repair_endpoint)

    engine._close(
        state,
        reason=close_reason,
        boundary_frame=boundary_frame,
    )

    assert calls == [
        {
            "state": state,
            "boundary_frame_exclusive": expected_boundary,
            "close_reason": close_reason,
        }
    ]
    assert state.pending == {}
    assert state.track["close_reason"] == close_reason
    assert state.active is False


@pytest.mark.parametrize(
    ("close_reason", "anchor_frame", "boundary", "detection_count", "target"),
    [
        ("natural", 10, 15, 1, 14),
        ("scene_cut", 10, 20, 20, 19),
        ("end_of_stream", 95, 100, 1, 99),
    ],
)
def test_interpolate_endpoint_is_capped_before_every_close_boundary(
    monkeypatch: pytest.MonkeyPatch,
    close_reason: str,
    anchor_frame: int,
    boundary: int,
    detection_count: int,
    target: int,
) -> None:
    engine = _interpolation_engine()
    engine.settings = {
        "endpoint_extension": 8,
        "reliable_endpoint_extension": 45,
        "reliable_endpoint_min_detector_frames": 20,
    }
    state = _state(left_frame=anchor_frame)
    state.track["detections"] = [
        {
            "frame_idx": anchor_frame - detection_count + index + 1,
            "box": [10.0, 10.0, 30.0, 30.0],
        }
        for index in range(detection_count)
    ]
    calls: list[dict[str, Any]] = []

    def run_endpoint(**kwargs: Any) -> int:
        calls.append(kwargs)
        return int(kwargs["target_frame"]) - int(kwargs["anchor_frame"])

    monkeypatch.setattr(engine, "_run_endpoint_affine", run_endpoint)

    emitted = engine._repair_interpolate_endpoint(
        state,
        boundary_frame_exclusive=boundary,
        close_reason=close_reason,
    )

    assert emitted == target - anchor_frame
    assert len(calls) == 1
    call = dict(calls[0])
    anchor_box = call.pop("anchor_box")
    np.testing.assert_allclose(anchor_box, state.last_detection_box)
    assert call == {
        "track_id": "t00001",
        "anchor_frame": anchor_frame,
        "target_frame": target,
        "direction": 1,
        "run_neural_review": True,
        "sparse_neural_review": True,
        "emit_before": None,
        "repair_reason": "interpolate_unanchored_endpoint",
        "boundary_reason": close_reason,
        "boundary_frame_exclusive": boundary,
    }
    assert target < boundary


def test_interpolate_endpoint_affine_runs_sparse_verifier_without_local_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, verify_calls = _endpoint_affine_engine(
        monkeypatch,
        fps=30.0,
    )
    engine.settings = {
        "endpoint_extension": 8,
        "reliable_endpoint_extension": 45,
        "reliable_endpoint_min_detector_frames": 20,
    }

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("strong sparse checkpoint must not run Local SCRFD")

    monkeypatch.setattr(engine, "_measure_review", forbidden)

    state = _state(left_frame=0)
    state.track["detections"] = [
        {"frame_idx": 0, "box": [10.0, 10.0, 30.0, 30.0]}
    ]
    emitted = engine._repair_interpolate_endpoint(
        state,
        boundary_frame_exclusive=3,
        close_reason="natural",
    )

    assert emitted == 2
    assert verify_calls == [(2, 0)]
    assert len(engine.endpoint_affine_candidates) == 2
    assert all(
        item["endpoint_repair_reason"] == "interpolate_unanchored_endpoint"
        and item["endpoint_boundary_reason"] == "natural"
        and item["endpoint_boundary_frame_exclusive"] == 3
        and item["reduced_assurance"] is True
        and item["_review_measurement"]["local_match_count"] == -1
        for item in engine.endpoint_affine_candidates
    )
    assert engine.endpoint_affine_candidates[0]["_review_measurement"][
        "verifier_face_probability"
    ] is None
    assert engine.endpoint_affine_candidates[1]["_review_measurement"][
        "verifier_face_probability"
    ] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("direction", "anchor_frame", "target_frame", "expected_checks"),
    [
        (1, 0, 10, [8, 10]),
        (-1, 10, 0, [2, 0]),
    ],
)
def test_sparse_endpoint_verifier_is_symmetric_and_commits_whole_buckets(
    monkeypatch: pytest.MonkeyPatch,
    direction: int,
    anchor_frame: int,
    target_frame: int,
    expected_checks: list[int],
) -> None:
    engine, verify_calls = _endpoint_affine_engine(
        monkeypatch,
        fps=30.0,
    )

    emitted = engine._run_endpoint_affine(
        track_id="t00001",
        anchor_frame=anchor_frame,
        anchor_box=np.asarray([10.0, 10.0, 30.0, 30.0]),
        target_frame=target_frame,
        direction=direction,
        run_neural_review=True,
        sparse_neural_review=True,
    )

    assert emitted == 10
    assert [frame for frame, _count in verify_calls] == expected_checks
    # Nothing from the active bucket is visible at its checkpoint. The second
    # checkpoint sees only the eight frames released by the first one.
    assert [count for _frame, count in verify_calls] == [0, 8]
    assert [item["frame_idx"] for item in engine.endpoint_affine_candidates] == list(
        range(anchor_frame + direction, target_frame + direction, direction)
    )


def test_sparse_endpoint_failed_checkpoint_refines_only_the_boundary_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, verify_calls = _endpoint_affine_engine(
        monkeypatch,
        fps=8.0,
        scores={4: 0.0},
    )
    # Keep this test focused on Verifier boundary refinement rather than the
    # independent Local-SCRFD anomaly fallback.
    engine.endpoint_local_review_frames = {"t00001": {0}}

    emitted = engine._run_endpoint_affine(
        track_id="t00001",
        anchor_frame=0,
        anchor_box=np.asarray([10.0, 10.0, 30.0, 30.0]),
        target_frame=4,
        direction=1,
        run_neural_review=True,
        sparse_neural_review=True,
    )

    assert emitted == 3
    assert [item["frame_idx"] for item in engine.endpoint_affine_candidates] == [
        1,
        2,
        3,
    ]
    assert [frame for frame, _count in verify_calls] == [2, 4, 3]
    assert engine.endpoint_verifier_refinement_frames == 1


def test_sparse_endpoint_checkpoint_interval_respects_materialized_memory_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, verify_calls = _endpoint_affine_engine(
        monkeypatch,
        fps=120.0,
    )
    # The 20px test box with corridor expansion 4 materializes a 100px crop.
    frame_bytes = 100 * 100 * 3
    engine._bidirectional_fusion_settings = lambda: {
        "max_corridor_side_pixels": 256,
        "max_materialized_bytes": 4 * frame_bytes,
    }
    decoded_ranges: list[tuple[int, int]] = []

    def decode_frames(first: int, last: int, **_kwargs: Any) -> dict[int, np.ndarray]:
        decoded_ranges.append((first, last))
        return {
            frame_idx: np.zeros((32, 32, 3), dtype=np.uint8)
            for frame_idx in range(first, last + 1)
        }

    engine._decode_frames = decode_frames

    emitted = engine._run_endpoint_affine(
        track_id="t00001",
        anchor_frame=0,
        anchor_box=np.asarray([10.0, 10.0, 30.0, 30.0]),
        target_frame=10,
        direction=1,
        run_neural_review=True,
        sparse_neural_review=True,
    )

    assert emitted == 10
    # 120 FPS would normally check every 30 frames. A four-crop memory budget
    # tightens that interval so the pending bucket cannot exceed the budget.
    assert [frame for frame, _count in verify_calls] == [4, 8, 10]
    assert max(last - first + 1 for first, last in decoded_ranges[1:]) <= 3


@pytest.mark.parametrize(
    ("candidate_frames", "expected_local_frames"),
    [
        ([0, 8, 16], []),
        ([0, 1, 2], [2, 10, 18]),
    ],
)
def test_endpoint_local_scrfd_runs_at_most_one_hz_for_recognition_gaps(
    monkeypatch: pytest.MonkeyPatch,
    candidate_frames: list[int],
    expected_local_frames: list[int],
) -> None:
    engine, _verify_calls = _endpoint_affine_engine(
        monkeypatch,
        fps=8.0,
    )
    engine.recognition_engine = SimpleNamespace(
        enabled=True,
        max_frames_per_track=3,
    )
    engine.recognition_candidates = {
        "t00001": [
            (f"detection:{frame_idx}", SimpleNamespace(frame_index=frame_idx))
            for frame_idx in candidate_frames
        ]
    }
    local_calls: list[int] = []

    def local_review(
        _image: np.ndarray,
        item: dict[str, Any],
        *_args: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        local_calls.append(int(item["frame_idx"]))
        review = engine._empty_local_review()
        review.update(
            {
                "local_match_count": 1,
                "local_confidence": 0.9,
                "local_box": list(item["box"]),
                "local_landmarks": None,
                "local_review_reason": "endpoint_recognition_candidate",
                "verifier_face_probability": 1.0,
            }
        )
        return review

    monkeypatch.setattr(engine, "_measure_review", local_review)
    monkeypatch.setattr(
        engine,
        "_refine_tracking_geometry",
        lambda *_args, **_kwargs: None,
    )

    engine._run_endpoint_affine(
        track_id="t00001",
        anchor_frame=0,
        anchor_box=np.asarray([10.0, 10.0, 30.0, 30.0]),
        target_frame=18,
        direction=1,
        run_neural_review=True,
        sparse_neural_review=True,
    )

    assert local_calls == expected_local_frames


def test_interpolate_backward_endpoint_stops_at_the_latest_scene_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _interpolation_engine()
    engine.settings = {
        "endpoint_extension": 8,
        "reliable_endpoint_extension": 45,
        "reliable_endpoint_min_detector_frames": 20,
    }
    engine.cache = SimpleNamespace(oldest_frame_index=lambda: 0)
    engine.audits = [
        {"frame_idx": 5, "scene_cut_from_previous": True},
    ]
    state = _state(left_frame=10)
    state.track["detections"] = [
        {"frame_idx": 10, "box": [10.0, 10.0, 30.0, 30.0]},
    ]
    calls: list[dict[str, Any]] = []

    def run_endpoint(**kwargs: Any) -> int:
        calls.append(kwargs)
        return abs(int(kwargs["target_frame"]) - int(kwargs["anchor_frame"]))

    monkeypatch.setattr(engine, "_run_endpoint_affine", run_endpoint)

    emitted = engine._repair_interpolate_endpoint(
        state,
        direction=-1,
        extension=8,
        close_reason="initial_pre_roll",
    )

    assert emitted == 5
    assert calls[0]["anchor_frame"] == 10
    assert calls[0]["target_frame"] == 5
    assert calls[0]["direction"] == -1
    assert calls[0]["boundary_frame_exclusive"] is None


def test_interpolate_endpoint_publication_filters_rejected_and_extends_contiguously() -> None:
    engine = _interpolation_engine()
    engine.config = {
        "revalidation": {
            "policy": {
                "continuity": {
                    "segment_max_center_jump": 1.0,
                    "segment_max_area_ratio": 2.0,
                }
            }
        }
    }
    track = {
        "track_id": "t00001",
        "accepted": True,
        "accepted_intervals": [[0, 0]],
    }
    rejected_track = {
        "track_id": "t00002",
        "accepted": False,
        "accepted_intervals": [],
    }
    engine.tracks = [track, rejected_track]
    engine.endpoint_affine_published_frames = 0
    engine.interpolate_endpoint_published_frames = 0
    review = engine._empty_local_review()
    review["local_review_reason"] = "interpolate_unanchored_endpoint"
    engine.endpoint_affine_candidates = [
        {
            "track_id": track_id,
            "frame_idx": frame_idx,
            "direction": 1,
            "anchor_frame": 0,
            "box": [10.0 + frame_idx, 10.0, 30.0 + frame_idx, 30.0],
            "motion_box": [10.0 + frame_idx, 10.0, 30.0 + frame_idx, 30.0],
            "inlier_points": 10,
            "quality": 0.8,
            "endpoint_repair": "affine_ransac",
            "endpoint_repair_reason": "interpolate_unanchored_endpoint",
            "reduced_assurance": True,
            "_review_measurement": dict(review),
        }
        for track_id in ("t00001", "t00002")
        for frame_idx in (1, 2)
    ]
    observations = [
        {
            "track_id": "t00001",
            "frame_idx": 0,
            "box": [10.0, 10.0, 30.0, 30.0],
        },
        {
            "track_id": "t00002",
            "frame_idx": 0,
            "box": [10.0, 10.0, 30.0, 30.0],
        },
    ]
    evidence: list[dict[str, Any]] = []

    published = engine._publish_endpoint_affine_candidates(observations, evidence)

    assert [
        (item["track_id"], item["frame_idx"])
        for item in published
    ] == [
        ("t00001", 0),
        ("t00002", 0),
        ("t00001", 1),
        ("t00001", 2),
    ]
    assert track["accepted_intervals"] == [[0, 2]]
    assert rejected_track["accepted_intervals"] == []
    assert len(evidence) == 2
    assert {item["track_id"] for item in evidence} == {"t00001"}
    assert all(item["reduced_assurance"] is True for item in evidence)
    assert engine.interpolate_endpoint_published_frames == 2
