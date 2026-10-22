from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from insightface.app.privateframe.streaming import ObjectState, _association_score


def _settings() -> dict[str, object]:
    return {
        "max_missed_frames": 300,
        "source_fps": 30.0,
        "association_max_scan_gap": 12,
        "association_max_gap_seconds": 1.0,
        "association_strict_geometry_after_seconds": 0.4,
        "association_max_area_ratio": 5.0,
        "association_max_center_distance": 1.75,
        "association_sparse_max_center_distance": 1.0,
        "association_min_iou": 0.05,
        "association_min_score": 0.0,
        "long_gap_requires_continuous_flow": True,
        "long_gap_min_iou": 0.15,
        "long_gap_max_center_distance": 0.50,
        "kalman_optical_flow": {"max_coast_frames": 8},
    }


def _state(*, frame_idx: int = 80, scan_rank: int | None = 40) -> ObjectState:
    box = np.asarray([10.0, 10.0, 30.0, 30.0], dtype=np.float64)
    return ObjectState(
        track={"track_id": "t00001"},
        flow=SimpleNamespace(box=box, coast=0),
        last_detection_frame=frame_idx,
        last_detection_box=box,
        last_detection_scan_rank=scan_rank,
    )


def _detection(*, frame_idx: int, scan_rank: int | None) -> dict[str, object]:
    result: dict[str, object] = {
        "frame_idx": frame_idx,
        "box": [10.0, 10.0, 30.0, 30.0],
    }
    if scan_rank is not None:
        result["detector_scan_rank"] = scan_rank
    return result


def test_interpolate_association_counts_real_scan_opportunities() -> None:
    # Fourteen decoded frames at stride 2 are only seven detector
    # opportunities. Sampled-out frames must not consume the association
    # budget.
    score = _association_score(
        _state(),
        _detection(frame_idx=94, scan_rank=47),
        94,
        _settings(),
        reference_box=np.asarray([10.0, 10.0, 30.0, 30.0]),
        allow_long_gap_flow=False,
    )

    assert score is not None


def test_sparse_interpolate_association_rejects_a_nearby_competing_face() -> None:
    # The distractor is within the broad sparse center limit, but after the
    # old 0.4-second every-frame window it lacks both strict overlap and strict
    # center agreement. A no-flow interpolate path must not switch identities.
    score = _association_score(
        _state(),
        {
            "frame_idx": 94,
            "detector_scan_rank": 47,
            "box": [26.0, 10.0, 46.0, 30.0],
        },
        94,
        _settings(),
        reference_box=np.asarray([10.0, 10.0, 30.0, 30.0]),
        allow_long_gap_flow=False,
    )

    assert score is None


def test_interpolate_association_rejects_too_many_real_scan_misses() -> None:
    score = _association_score(
        _state(),
        _detection(frame_idx=94, scan_rank=53),
        94,
        _settings(),
        reference_box=np.asarray([10.0, 10.0, 30.0, 30.0]),
        allow_long_gap_flow=False,
    )

    assert score is None


def test_interpolate_association_retains_an_elapsed_time_safety_cap() -> None:
    score = _association_score(
        _state(),
        _detection(frame_idx=120, scan_rank=41),
        120,
        _settings(),
        reference_box=np.asarray([10.0, 10.0, 30.0, 30.0]),
        allow_long_gap_flow=False,
    )

    assert score is None


def test_rankless_call_preserves_every_frame_gap_semantics() -> None:
    score = _association_score(
        _state(scan_rank=None),
        _detection(frame_idx=94, scan_rank=None),
        94,
        _settings(),
        reference_box=np.asarray([10.0, 10.0, 30.0, 30.0]),
        allow_long_gap_flow=False,
    )

    assert score is None
