"""Privacy invariants for final, stabilized render geometry."""

from __future__ import annotations

from typing import Any

from insightface.app.privateframe import revalidation
from insightface.app.privateframe import streaming


def _observation(
    track_id: str,
    box: list[float],
    *,
    source: str = "kalman_optical_flow",
) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "frame_idx": 5,
        "source": source,
        "confidence": 0.9 if source == "detector" else None,
        "box": list(box),
    }


def _evidence(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "track_id": item["track_id"],
            "frame_idx": item["frame_idx"],
            "box": list(item["box"]),
        }
        for item in observations
    ]


def test_final_cross_track_dedup_suppresses_only_safe_render_duplicates() -> None:
    observations = [
        _observation("detector", [0.0, 0.0, 100.0, 100.0], source="detector"),
        _observation("duplicate", [5.0, 0.0, 105.0, 100.0]),
    ]

    result = streaming._deduplicate_final_cross_tracks(
        observations,
        _evidence(observations),
        recognition_mode="all",
    )

    assert [item["track_id"] for item in result] == ["detector"]


def test_final_cross_track_dedup_keeps_geometry_below_safe_coverage() -> None:
    observations = [
        _observation("detector", [0.0, 0.0, 79.0, 100.0], source="detector"),
        _observation("accepted", [0.0, 0.0, 100.0, 100.0]),
    ]

    result = streaming._deduplicate_final_cross_tracks(
        observations,
        _evidence(observations),
        recognition_mode="all",
    )

    assert {item["track_id"] for item in result} == {"detector", "accepted"}


def test_same_track_disjoint_final_box_is_a_geometry_hole() -> None:
    tracks = [
        {
            "track_id": "accepted",
            "accepted": True,
            "accepted_intervals": [[5, 5]],
        }
    ]
    observations = [_observation("accepted", [200.0, 0.0, 300.0, 100.0])]
    evidence = [
        {
            "track_id": "accepted",
            "frame_idx": 5,
            "box": [0.0, 0.0, 100.0, 100.0],
        }
    ]

    assert streaming._accepted_interval_coverage(
        tracks,
        observations,
        evidence,
    ) == (1, 1, 0)
    assert streaming._deduplicate_final_cross_tracks(
        observations,
        evidence,
        recognition_mode="all",
    ) == []


def test_unsafe_stabilized_box_falls_back_to_safe_raw_output_box() -> None:
    raw_box = [0.0, 0.0, 100.0, 100.0]
    observation = {
        **_observation("accepted", [30.0, 0.0, 130.0, 100.0]),
        "raw_output_box": raw_box,
        "motion_box": [30.0, 0.0, 130.0, 100.0],
        "box_stabilization": "robust_bidirectional",
    }

    result = streaming._validate_final_track_geometry(
        [observation],
        _evidence([_observation("accepted", raw_box)]),
    )

    assert len(result) == 1
    assert result[0]["box"] == raw_box
    assert result[0]["motion_box"] == raw_box
    assert result[0]["box_stabilization"] == "raw_evidence_fallback"


def test_endpoint_geometry_uses_the_same_raw_evidence_fallback() -> None:
    raw_box = [10.0, 10.0, 30.0, 30.0]
    endpoint = {
        **_observation("accepted", [40.0, 10.0, 60.0, 30.0]),
        "raw_output_box": raw_box,
        "endpoint_repair": "affine_ransac",
    }

    result = streaming._deduplicate_final_cross_tracks(
        [endpoint],
        _evidence([_observation("accepted", raw_box)]),
        recognition_mode="all",
    )

    assert len(result) == 1
    assert result[0]["endpoint_repair"] == "affine_ransac"
    assert result[0]["box"] == raw_box
    assert result[0]["box_stabilization"] == "raw_evidence_fallback"


def test_historical_observation_without_evidence_remains_valid() -> None:
    observation = _observation("legacy", [0.0, 0.0, 100.0, 100.0])

    assert streaming._validate_final_track_geometry([observation], []) == [
        observation
    ]


def test_stitched_detector_and_evidence_select_the_same_high_confidence_fragment() -> None:
    canonical_box = [0.0, 0.0, 100.0, 100.0]
    alias_box = [200.0, 0.0, 300.0, 100.0]
    canonical_detection = {
        "track_id": "canonical",
        "frame_idx": 75,
        "source": "detector",
        "confidence": 0.7328691482543945,
        "box": canonical_box,
    }
    alias_detection = {
        "track_id": "alias",
        "frame_idx": 75,
        "source": "detector",
        "confidence": 0.7500922679901123,
        "box": alias_box,
    }
    tracks = [
        {"track_id": "canonical", "detections": [canonical_detection]},
        {"track_id": "alias", "detections": [alias_detection]},
    ]
    detections = [canonical_detection, alias_detection]
    evidence = [
        {
            "track_id": "canonical",
            "frame_idx": 75,
            "source": "detector",
            "confidence": canonical_detection["confidence"],
            "local_confidence": 0.99,
            "box": canonical_box,
        },
        {
            "track_id": "alias",
            "frame_idx": 75,
            "source": "detector",
            "confidence": alias_detection["confidence"],
            "local_confidence": 0.10,
            "box": alias_box,
        },
    ]

    merged_tracks, selected_evidence = streaming._apply_fragment_aliases(
        tracks,
        detections,
        [],
        evidence,
        {"canonical": "canonical", "alias": "canonical"},
    )
    merged_tracks[0].update(
        {"accepted": True, "accepted_intervals": [[75, 75]]}
    )
    observations = revalidation._finalize(
        merged_tracks,
        {"detections": detections},
        {"observations": []},
        selected_evidence,
    )

    assert len(selected_evidence) == 1
    assert selected_evidence[0]["confidence"] == alias_detection["confidence"]
    assert selected_evidence[0]["box"] == alias_box
    assert len(observations) == 1
    assert observations[0]["confidence"] == alias_detection["confidence"]
    assert observations[0]["box"] == alias_box
    assert streaming._validate_final_track_geometry(
        observations,
        selected_evidence,
    ) == observations


def test_endpoint_publication_does_not_duplicate_an_existing_track_frame() -> None:
    engine = object.__new__(streaming.StreamingEngine)
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
        "track_id": "accepted",
        "accepted": True,
        "accepted_intervals": [[5, 5]],
    }
    engine.tracks = [track]
    engine.endpoint_affine_published_frames = 0
    engine.interpolate_endpoint_published_frames = 0
    endpoint_template = {
        "track_id": "accepted",
        "direction": 1,
        "anchor_frame": 5,
        "source": "kalman_optical_flow",
        "inlier_points": 10,
        "quality": 0.8,
        "endpoint_repair": "affine_ransac",
        "_review_measurement": engine._empty_local_review(),
    }
    engine.endpoint_affine_candidates = [
        {
            **endpoint_template,
            "frame_idx": frame_idx,
            "box": box,
            "motion_box": box,
        }
        for frame_idx, box in (
            (5, [50.0, 50.0, 70.0, 70.0]),
            (6, [11.0, 10.0, 31.0, 30.0]),
        )
    ]
    original = {
        "track_id": "accepted",
        "frame_idx": 5,
        "source": "detector",
        "box": [10.0, 10.0, 30.0, 30.0],
    }
    evidence: list[dict[str, Any]] = []

    result = engine._publish_endpoint_affine_candidates([original], evidence)

    assert [(item["track_id"], item["frame_idx"]) for item in result] == [
        ("accepted", 5),
        ("accepted", 6),
    ]
    assert result[0]["box"] == original["box"]
    assert track["accepted_intervals"] == [[5, 6]]
    assert len(evidence) == 1
    assert engine.endpoint_affine_published_frames == 1
