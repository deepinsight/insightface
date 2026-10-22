from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from insightface.app.privateframe.geometry import mutually_covers
from insightface.app.privateframe.revalidation import (
    _admission_decision,
    _summary,
    finalize_precomputed,
)
from insightface.app.privateframe.streaming import (
    StreamingEngine,
    _accepted_interval_coverage,
    _deduplicate_final_cross_tracks,
    _detector_pipeline_depth,
    _detector_scan_burst_frames,
    resolve_analysis_stride,
)


def _engine(*, frame_stride: int = 2, frame_count: int = 12) -> StreamingEngine:
    engine = object.__new__(StreamingEngine)
    engine.detector_frame_stride = frame_stride
    engine.detector_scan_burst_frames = 4
    engine.forced_detector_scan_reasons = {}
    engine.metadata = SimpleNamespace(frame_count=frame_count)
    return engine


@pytest.mark.parametrize(
    ("source_fps", "max_analysis_fps", "expected_stride"),
    [
        (29.97, 30.0, 1),
        (30.02, 30.0, 1),
        (31.5, 30.0, 1),
        (31.5001, 30.0, 2),
        (60.0, 30.0, 2),
        (120.0, 30.0, 4),
        (120.0, 120.0, 1),
        (30.0, 15.0, 2),
        (60.0, 15.0, 4),
        (120.0, 15.0, 8),
        (240.0, 15.0, 16),
        (25.0, 15.0, 2),
    ],
)
def test_analysis_stride_uses_a_five_percent_soft_ceiling(
    source_fps: float,
    max_analysis_fps: float,
    expected_stride: int,
) -> None:
    assert resolve_analysis_stride(source_fps, max_analysis_fps) == expected_stride


def test_every_frame_mode_never_samples_out_a_frame() -> None:
    engine = _engine(frame_stride=1)

    assert [engine._detector_scan_reason(frame_idx) for frame_idx in range(12)] == [
        "every_frame"
    ] * 12


def test_stride_uses_regular_frames_and_forces_the_endpoint() -> None:
    engine = _engine(frame_stride=3, frame_count=11)

    assert engine._detector_scan_reason(0) == "regular_stride"
    assert engine._detector_scan_reason(1) is None
    assert engine._detector_scan_reason(3) == "regular_stride"
    assert engine._detector_scan_reason(10) == "end_of_stream"


def test_admission_burst_only_adds_frames_between_regular_samples() -> None:
    engine = _engine(frame_stride=3)
    engine._force_detector_scan_range(1, 3, "new_track_burst")

    assert engine._detector_scan_reason(1) == "new_track_burst"
    assert engine._detector_scan_reason(2) == "new_track_burst"
    assert engine._detector_scan_reason(3) == "regular_stride"
    assert engine._detector_scan_reason(4) is None


def test_scene_cut_reason_wins_over_other_forced_reasons() -> None:
    engine = _engine(frame_stride=4)
    engine._force_detector_scan_range(1, 1, "video_start_burst")
    engine._force_detector_scan_range(1, 1, "scene_cut")

    assert engine._detector_scan_reason(1) == "scene_cut"


def test_force_range_is_not_truncated_by_the_untrusted_reported_frame_count() -> None:
    engine = _engine(frame_stride=2, frame_count=5)
    engine._force_detector_scan_range(-3, 9, "scene_cut_burst")

    # An under-reported container count must not cut off a real adaptive burst.
    # Reasons beyond the decoded EOF are never consumed and leave with the engine.
    assert sorted(engine.forced_detector_scan_reasons) == list(range(10))


def test_admission_uses_real_observations_on_the_sampling_cadence() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "insightface/app/privateframe/configs/base.yaml"
    )
    policy = yaml.safe_load(config_path.read_text(encoding="utf-8"))["revalidation"][
        "policy"
    ]
    values = [
        {
            "frame_idx": frame_idx,
            "source": "detector",
            "box": [10.0, 10.0, 30.0, 30.0],
            "confidence": 0.9,
            "local_match_count": 1,
            "local_confidence": 0.8,
            "verifier_face_probability": 0.9,
        }
        for frame_idx in range(0, 12, 2)
    ]

    every_frame_summary = _summary(values, policy, detector_frame_stride=1)
    sampled_summary = _summary(values, policy, detector_frame_stride=2)

    assert every_frame_summary["detector_source_frames"] == 6
    assert sampled_summary["detector_source_frames"] == 6
    assert every_frame_summary["joint_strong_anchor"] == 0.0
    assert sampled_summary["joint_strong_anchor"] == 1.0
    assert _admission_decision(every_frame_summary, policy)["accepted"] is False
    assert _admission_decision(sampled_summary, policy)["accepted"] is True


def test_missing_a_scheduled_detection_breaks_the_sampled_anchor() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "insightface/app/privateframe/configs/base.yaml"
    )
    policy = yaml.safe_load(config_path.read_text(encoding="utf-8"))["revalidation"][
        "policy"
    ]
    values = [
        {
            "frame_idx": frame_idx,
            "source": "detector",
            "box": [10.0, 10.0, 30.0, 30.0],
            "confidence": 0.9,
            "local_match_count": 1,
            "local_confidence": 0.8,
            "verifier_face_probability": 0.9,
        }
        for frame_idx in (0, 2, 6, 8, 12, 14)
    ]

    summary = _summary(values, policy, detector_frame_stride=2)

    assert summary["joint_strong_anchor"] == 0.0
    assert summary["leading_consecutive_detector_frames"] == 2


def test_missing_a_forced_scan_breaks_detector_and_local_anchor_continuity() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "insightface/app/privateframe/configs/base.yaml"
    )
    policy = yaml.safe_load(config_path.read_text(encoding="utf-8"))["revalidation"][
        "policy"
    ]
    values = [
        {
            "frame_idx": frame_idx,
            "source": "detector",
            "box": [10.0, 10.0, 30.0, 30.0],
            "confidence": 0.9,
            "local_match_count": 1,
            "local_confidence": 0.8,
            "verifier_face_probability": 0.9,
        }
        for frame_idx in (0, 2, 4, 6, 8, 10)
    ]

    summary = _summary(
        values,
        policy,
        detector_frame_stride=2,
        detector_scan_rank={
            frame_idx: index
            for index, frame_idx in enumerate((0, 1, 2, 4, 6, 7, 8, 10))
        },
    )

    # Frames 1 and 7 were real full-frame opportunities on which this target
    # was absent.  They break both detector persistence and the joint anchor;
    # merely being aligned to the regular stride cannot hide a forced miss.
    assert summary["joint_strong_anchor"] == 0.0
    assert summary["leading_consecutive_detector_frames"] == 1


@pytest.mark.parametrize("detector_stride", [3, 4, 8, 16])
def test_unscanned_frames_are_neutral_for_any_detector_stride(
    detector_stride: int,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "insightface/app/privateframe/configs/base.yaml"
    )
    policy = yaml.safe_load(config_path.read_text(encoding="utf-8"))["revalidation"][
        "policy"
    ]
    hit_frames = tuple(index * detector_stride for index in range(4))
    values = [
        {
            "frame_idx": frame_idx,
            "source": "detector",
            "box": [10.0, 10.0, 30.0, 30.0],
            "confidence": 0.9,
            "local_match_count": 1,
            "local_confidence": 0.8,
            "verifier_face_probability": 0.9,
        }
        for frame_idx in hit_frames
    ]

    summary = _summary(
        values,
        policy,
        detector_frame_stride=detector_stride,
        detector_scan_rank={
            frame_idx: index for index, frame_idx in enumerate(hit_frames)
        },
    )

    assert summary["leading_consecutive_detector_frames"] == 4
    assert summary["joint_strong_anchor"] == 1.0


@pytest.mark.parametrize("detector_stride", [3, 4, 8, 16])
def test_finalization_uses_runtime_detector_stride_as_review_cadence(
    detector_stride: int,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "insightface/app/privateframe/configs/base.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hit_frames = tuple(index * detector_stride for index in range(6))
    values = [
        {
            "track_id": "sampled",
            "frame_idx": frame_idx,
            "source": "detector",
            "box": [10.0, 10.0, 30.0, 30.0],
            "confidence": 0.9,
            "local_match_count": 1,
            "local_confidence": 0.8,
            "verifier_face_probability": 0.9,
        }
        for frame_idx in hit_frames
    ]
    track = {"track_id": "sampled"}

    finalize_precomputed(
        {
            "frames": [
                {"frame_idx": frame_idx, "detector_scan_performed": True}
                for frame_idx in hit_frames
            ],
            "detections": values,
        },
        [track],
        {"observations": [], "shadows": []},
        values,
        config,
        detector_frame_stride=detector_stride,
    )

    assert track["revalidation_summary"]["local_review_stride"] == detector_stride
    assert track["revalidation_summary"]["joint_strong_anchor"] == 1.0
    assert track["accepted"] is True


@pytest.mark.parametrize("detector_stride", [3, 4, 8, 16])
def test_real_scan_miss_breaks_sampled_strong_windows(
    detector_stride: int,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "insightface/app/privateframe/configs/base.yaml"
    )
    policy = yaml.safe_load(config_path.read_text(encoding="utf-8"))["revalidation"][
        "policy"
    ]
    hit_frames = tuple(index * detector_stride for index in range(4))
    forced_miss = 1
    scan_frames = (hit_frames[0], forced_miss, *hit_frames[1:])
    values = [
        {
            "frame_idx": frame_idx,
            "source": "detector",
            "box": [10.0, 10.0, 30.0, 30.0],
            "confidence": 0.9,
            "local_match_count": 1,
            "local_confidence": 0.8,
            "verifier_face_probability": 0.9,
        }
        for frame_idx in hit_frames
    ]

    summary = _summary(
        values,
        policy,
        detector_frame_stride=detector_stride,
        detector_scan_rank={
            frame_idx: index for index, frame_idx in enumerate(scan_frames)
        },
    )

    assert summary["leading_consecutive_detector_frames"] == 1
    assert summary["joint_strong_anchor"] == 0.0


@pytest.mark.parametrize(
    ("review_stride", "review_phase", "frame_count", "attempted_frames"),
    [
        (1, 0, 8, 8),
        (2, 1, 8, 4),
        (4, 2, 15, 4),
    ],
)
def test_local_summary_uses_only_attempted_frames_as_its_denominator(
    review_stride: int,
    review_phase: int,
    frame_count: int,
    attempted_frames: int,
) -> None:
    values = []
    attempted_index = 0
    for frame_idx in range(frame_count):
        attempted = frame_idx % review_stride == review_phase
        matched = attempted and attempted_index % 2 == 0
        values.append(
            {
                "frame_idx": frame_idx,
                "source": "tracking",
                "box": [10.0, 10.0, 30.0, 30.0],
                "local_match_count": 1 if matched else 0 if attempted else -1,
                "local_confidence": 0.8 if matched else None,
                "verifier_face_probability": 0.9,
            }
        )
        if attempted:
            attempted_index += 1

    summary = _summary(values, local_review_stride=review_stride)
    matched_frames = (attempted_frames + 1) // 2

    assert summary["local_review_attempted_frames"] == attempted_frames
    assert summary["local_review_sampled_out_frames"] == frame_count - attempted_frames
    assert summary["local_review_failed_frames"] == attempted_frames - matched_frames
    assert summary["local_match_fraction"] == pytest.approx(
        matched_frames / attempted_frames
    )
    assert summary["track_fraction_with_confidence_gte_035"] == pytest.approx(
        matched_frames / attempted_frames
    )
    assert summary["verifier_frames"] == frame_count
    assert summary["verifier_coverage_fraction"] == 1.0


@pytest.mark.parametrize(
    ("review_stride", "review_phase"),
    [(1, 0), (2, 1), (4, 2)],
)
def test_joint_anchor_uses_the_offset_local_review_cadence(
    review_stride: int,
    review_phase: int,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "insightface/app/privateframe/configs/base.yaml"
    )
    policy = yaml.safe_load(config_path.read_text(encoding="utf-8"))["revalidation"][
        "policy"
    ]
    last_review = review_phase + 3 * review_stride
    values = [
        {
            "frame_idx": frame_idx,
            "source": "tracking",
            "box": [10.0, 10.0, 30.0, 30.0],
            "local_match_count": (
                1
                if frame_idx >= review_phase
                and (frame_idx - review_phase) % review_stride == 0
                else -1
            ),
            "local_confidence": (
                0.8
                if frame_idx >= review_phase
                and (frame_idx - review_phase) % review_stride == 0
                else None
            ),
            "verifier_face_probability": 0.9,
        }
        for frame_idx in range(last_review + 1)
    ]

    summary = _summary(
        values,
        policy,
        detector_frame_stride=1,
        local_review_stride=review_stride,
    )

    assert summary["detector_source_frames"] == 0
    assert summary["local_review_attempted_frames"] == 4
    assert summary["joint_strong_anchor"] == 1.0


@pytest.mark.parametrize(
    ("review_stride", "review_phase"),
    [(1, 0), (2, 1), (4, 2)],
)
def test_failed_local_attempt_occupies_rank_and_breaks_the_strong_window(
    review_stride: int,
    review_phase: int,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "insightface/app/privateframe/configs/base.yaml"
    )
    policy = yaml.safe_load(config_path.read_text(encoding="utf-8"))["revalidation"][
        "policy"
    ]
    scheduled = [review_phase + index * review_stride for index in range(7)]
    failed_frame = scheduled[3]
    values = [
        {
            "frame_idx": frame_idx,
            "source": "tracking",
            "box": [10.0, 10.0, 30.0, 30.0],
            "local_match_count": (
                0 if frame_idx == failed_frame else 1 if frame_idx in scheduled else -1
            ),
            "local_confidence": (
                None if frame_idx == failed_frame or frame_idx not in scheduled else 0.8
            ),
            "verifier_face_probability": 0.9,
        }
        for frame_idx in range(scheduled[-1] + 1)
    ]

    summary = _summary(
        values,
        policy,
        detector_frame_stride=1,
        local_review_stride=review_stride,
    )

    assert summary["local_review_attempted_frames"] == 7
    assert summary["local_review_failed_frames"] == 1
    assert summary["joint_strong_anchor"] == 0.0


def test_accepted_interval_coverage_detects_render_holes() -> None:
    tracks = [
        {
            "track_id": "accepted",
            "accepted": True,
            "accepted_intervals": [[2, 4], [7, 8]],
        },
        {
            "track_id": "rejected",
            "accepted": False,
            "accepted_intervals": [[0, 9]],
        },
    ]
    observations = [
        {"track_id": "accepted", "frame_idx": 2},
        {"track_id": "accepted", "frame_idx": 3},
        {"track_id": "accepted", "frame_idx": 7},
        {"track_id": "accepted", "frame_idx": 8},
    ]

    assert _accepted_interval_coverage(tracks, observations) == (5, 1, 0)


def test_all_mode_accepts_geometric_coverage_from_a_deduplicating_track() -> None:
    tracks = [
        {
            "track_id": "duplicate",
            "accepted": True,
            "accepted_intervals": [[4, 4]],
        }
    ]
    observations = [
        {
            "track_id": "suppressor",
            "frame_idx": 4,
            "box": [11.0, 10.0, 29.0, 30.0],
        }
    ]
    evidence = [
        {
            "track_id": "duplicate",
            "frame_idx": 4,
            "box": [10.0, 10.0, 30.0, 30.0],
        }
    ]

    assert _accepted_interval_coverage(tracks, observations, evidence) == (1, 1, 0)
    assert _accepted_interval_coverage(
        tracks,
        observations,
        evidence,
        allow_cross_track_coverage=True,
    ) == (1, 0, 1)


def test_final_cross_track_dedup_uses_the_stabilized_stadium_geometry() -> None:
    """The real frame-2645 suppressor moved below safe coverage when smoothed."""

    reference_box = [
        263.53942385536544,
        847.934008744694,
        374.68342185793216,
        994.3868835892906,
    ]
    raw_detector_box = [
        249.72206878662112,
        855.9078140258789,
        359.420654296875,
        997.3791961669922,
    ]
    stabilized_detector_box = [
        245.22773938549395,
        860.1194372732741,
        356.8281995684341,
        999.9084206786799,
    ]
    assert mutually_covers(
        reference_box,
        raw_detector_box,
        min_coverage=0.80,
        max_area_ratio=2.50,
    )
    assert not mutually_covers(
        reference_box,
        stabilized_detector_box,
        min_coverage=0.80,
        max_area_ratio=2.50,
    )

    def observations(detector_box: list[float]) -> list[dict[str, object]]:
        return [
            {
                "track_id": "t00109",
                "frame_idx": 2645,
                "source": "kalman_optical_flow",
                "box": reference_box,
            },
            {
                "track_id": "t00112",
                "frame_idx": 2645,
                "source": "detector",
                "confidence": 0.5601000785827637,
                "box": detector_box,
            },
        ]

    evidence = [
        {
            "track_id": "t00109",
            "frame_idx": 2645,
            "box": reference_box,
        },
        {
            "track_id": "t00112",
            "frame_idx": 2645,
            "box": raw_detector_box,
        },
    ]
    # This proves why final cross-track suppression must not see raw boxes.
    assert {
        item["track_id"]
        for item in _deduplicate_final_cross_tracks(
            observations(raw_detector_box),
            evidence,
            recognition_mode="all",
        )
    } == {"t00112"}

    final = _deduplicate_final_cross_tracks(
        observations(stabilized_detector_box),
        evidence,
        recognition_mode="all",
    )
    assert {item["track_id"] for item in final} == {"t00109", "t00112"}
    assert _accepted_interval_coverage(
        [
            {
                "track_id": track_id,
                "accepted": True,
                "accepted_intervals": [[2645, 2645]],
            }
            for track_id in ("t00109", "t00112")
        ],
        final,
        evidence,
        allow_cross_track_coverage=True,
    ) == (2, 0, 0)


@pytest.mark.parametrize("recognition_mode", ["exempt", "blur_only"])
def test_selective_recognition_never_deduplicates_across_tracks(
    recognition_mode: str,
) -> None:
    observations = [
        {
            "track_id": track_id,
            "frame_idx": 8,
            "source": "detector",
            "confidence": confidence,
            "box": [10.0, 10.0, 30.0, 30.0],
        }
        for track_id, confidence in (("first", 0.9), ("second", 0.8))
    ]
    evidence = [
        {
            "track_id": item["track_id"],
            "frame_idx": item["frame_idx"],
            "box": item["box"],
        }
        for item in observations
    ]

    result = _deduplicate_final_cross_tracks(
        observations,
        evidence,
        recognition_mode=recognition_mode,
    )

    assert {item["track_id"] for item in result} == {"first", "second"}


def test_sampling_pipeline_depth_is_capped_by_the_frame_byte_budget() -> None:
    frame_bytes_4k = 3840 * 2160 * 3
    byte_limit = 256 * 1024 * 1024

    assert _detector_pipeline_depth(6, 1, frame_bytes_4k, byte_limit) == 6
    assert _detector_pipeline_depth(6, 2, frame_bytes_4k, byte_limit) == 10
    assert _detector_pipeline_depth(6, 4, frame_bytes_4k, byte_limit) == 10


def test_sampling_burst_satisfies_anchor_and_persistence_gates() -> None:
    assert _detector_scan_burst_frames({}) == 1
    assert (
        _detector_scan_burst_frames(
            {
                "strong_anchor_window_frames": 4,
                "min_detector_frames": 6,
            }
        )
        == 6
    )
