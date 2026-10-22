"""Collector and endpoint publication safety, using only synthetic geometry.

No models or media are loaded. Tests are intended for the unified remote run.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from insightface.app.privateframe.base_config import read_default_config
from insightface.app.privateframe.endpoint_conflicts import EndpointConflictReview
from insightface.app.privateframe.endpoint_matching import region_for_box
from insightface.app.privateframe.streaming import (
    StreamingEngine,
    _accepted_interval_coverage,
    _validate_final_track_geometry,
)


FACE = [40.0, 40.0, 60.0, 60.0]
PREDICTION = [35.0, 35.0, 65.0, 65.0]
IMAGE = np.zeros((100, 100, 3), dtype=np.uint8)


def _config(*, mode="all", **overrides):
    config = read_default_config()
    config["output"]["artifacts_level"] = "audit"
    config["recognition"]["mode"] = mode
    config["revalidation"]["angles"] = [0, 90, -90]
    config["tracking"]["endpoint_conflicts"]["enabled"] = True
    # Existing budget tests describe three-view review explicitly. The new
    # default's one-view budget is exercised independently below.
    config["tracking"]["endpoint_conflicts"]["angles"] = [0, 90, -90]
    config["tracking"]["endpoint_conflicts"].update(overrides)
    return config


def _face(frame=10):
    return {
        "box": list(FACE), "confidence": 0.95,
        "detection_id": f"d{frame}:face", "frame_idx": frame,
    }


def _owner(frame=10, track_id="owner"):
    return {
        "track_id": track_id, "frame_idx": frame, "source": "detector",
        "box": list(FACE), "detector_box": list(FACE), "confidence": 0.95,
    }


def _local_review():
    return {"local_box": list(FACE), "local_confidence": 0.95, "local_match_count": 1}


def _endpoint(frame=10, *, track_id="endpoint", direction=1, anchor=None, localized=False):
    return {
        "track_id": track_id, "frame_idx": frame,
        "box": list(PREDICTION), "motion_box": list(PREDICTION),
        "source": "kalman_optical_flow", "direction": direction,
        "anchor_frame": frame - direction if anchor is None else anchor,
        "endpoint_repair": "affine_ransac",
        "endpoint_repair_reason": "interpolate_unanchored_endpoint",
        "reduced_assurance": True, "inlier_points": 12, "quality": 0.9,
        "_review_measurement": _local_review() if localized else {},
    }


class FakeReviewer:
    def __init__(self, *, detections=None, complete=True, error=None):
        self.detections = [_face()] if detections is None else detections
        self.complete = complete
        self.error = error
        self.calls = []
        self.angles_seen = []

    def detect_region(self, image, *, origin, max_calls, angles):
        self.calls.append({"shape": image.shape, "origin": origin, "max_calls": max_calls})
        self.angles_seen.append(tuple(angles))
        if self.error is not None:
            raise self.error
        assert max_calls >= len(angles), "collector must reserve all configured angles"
        return {
            "detections": deepcopy(self.detections),
            "complete": self.complete,
            "detector_calls": len(angles),
        }


def _no_decode(*_args, **_kwargs):
    raise AssertionError("this synthetic frame already contains the full review region")


def _collect(collector, candidate, reviewer):
    collector.collect(candidate, image=IMAGE, origin=(0, 0), decode=_no_decode, reviewer=reviewer)


def _prepare_frame(collector, frame=10, *, scans=True):
    owner = _owner(frame)
    collector.core[frame] = {"owner": owner}
    collector.evidence[("owner", frame)] = _local_review()
    if scans:
        collector.register_scan(frame, [_face(frame)])
    return owner


@pytest.mark.parametrize("mode,enabled", [("all", False), ("blur_only", True), ("exempt", True)])
def test_disabled_or_selective_mode_never_collects_or_suppresses(mode, enabled):
    collector = EndpointConflictReview(_config(mode=mode, enabled=enabled), 30, 300)
    owner = _prepare_frame(collector)
    reviewer = FakeReviewer()
    _collect(collector, _endpoint(localized=True), reviewer)

    assert collector.enabled is False
    assert collector.scans == {}
    assert collector.bundles == {}
    assert reviewer.calls == []
    assert collector.resolve([owner], [], {}) == set()


def test_scan_geometry_is_copied_before_association_can_mutate_it():
    collector = EndpointConflictReview(_config(), 30, 300)
    detection = {**_face(), "detector_box": list(FACE)}
    collector.register_scan(10, [detection])
    detection["box"][0] = -500
    detection["detector_box"][1] = -600
    detection["confidence"] = 0

    assert collector.scans[10][0]["box"] == FACE
    assert collector.scans[10][0]["confidence"] == 0.95


@pytest.mark.parametrize("scan_state", ["not_scanned", "empty_scan"])
def test_empty_evidence_is_not_proof_of_a_duplicate(scan_state):
    collector = EndpointConflictReview(_config(), 30, 300)
    owner = _prepare_frame(collector, scans=False)
    if scan_state == "empty_scan":
        collector.register_scan(10, [])
    reviewer = FakeReviewer(detections=[])
    _collect(collector, _endpoint(), reviewer)

    assert reviewer.calls
    assert collector.resolve([owner], [], {}) == set()
    assert collector.decisions == []


@pytest.mark.parametrize("overrides", [
    {"max_calls_total": 0},
    {"max_calls_per_video_second": 0.0},
    {"max_calls_per_frame": 2},
])
def test_budget_shortfall_keeps_endpoint_without_independent_target_evidence(overrides):
    collector = EndpointConflictReview(_config(**overrides), 30, 300)
    owner = _prepare_frame(collector)
    candidate = _endpoint()
    reviewer = FakeReviewer()
    _collect(collector, candidate, reviewer)

    assert reviewer.calls == []
    assert collector.resolve([owner], [], {}) == set()
    assert "local_box" not in candidate
    assert candidate["_review_measurement"] == {}


def test_complete_existing_same_frame_localizations_need_no_added_inference():
    collector = EndpointConflictReview(_config(max_calls_total=0), 30, 300)
    owner = _prepare_frame(collector)
    reviewer = FakeReviewer()
    _collect(collector, _endpoint(localized=True), reviewer)

    assert reviewer.calls == []
    assert collector.resolve([owner], [], {}) == {("endpoint", 10, 1)}
    assert collector.stats["existing_evidence_reuses"] == 1


def test_upright_conflict_review_uses_one_call_without_changing_tracking_angles():
    config = _config(angles=[0], max_calls_total=1, max_calls_per_frame=1)
    collector = EndpointConflictReview(config, 30, 300)
    owner = _prepare_frame(collector)
    reviewer = FakeReviewer()
    _collect(collector, _endpoint(), reviewer)

    assert reviewer.angles_seen == [(0,)]
    assert collector.stats["detector_calls"] == 1
    assert collector.calls_by_frame[10] == 1
    assert collector.summary()["angles"] == [0]
    assert config["revalidation"]["angles"] == [0, 90, -90]
    assert collector.resolve([owner], [], {}) == {("endpoint", 10, 1)}


def test_upright_review_keeps_endpoint_when_no_suitable_face_is_found():
    collector = EndpointConflictReview(_config(angles=[0]), 30, 300)
    owner = _prepare_frame(collector)
    reviewer = FakeReviewer(detections=[])
    _collect(collector, _endpoint(), reviewer)

    assert reviewer.angles_seen == [(0,)]
    assert collector.resolve([owner], [], {}) == set()


def test_upright_review_retains_a_second_face_returned_below_claim_threshold():
    collector = EndpointConflictReview(_config(angles=[0]), 30, 300)
    owner = _prepare_frame(collector)
    reviewer = FakeReviewer(detections=[_face(), {
        "box": [34.0, 45.0, 39.0, 55.0], "confidence": 0.20,
        "detection_id": "weak-other-face", "frame_idx": 10,
    }])
    _collect(collector, _endpoint(), reviewer)

    assert reviewer.angles_seen == [(0,)]
    assert collector.resolve([owner], [], {}) == set()
    assert any(row["reason"] == "another_detected_face_has_no_core_coverage"
               for row in collector.decisions if row["track_id"] == "endpoint")


def test_shared_scan_alone_cannot_replace_independent_target_localization():
    collector = EndpointConflictReview(_config(), 30, 300)
    owner = _prepare_frame(collector)
    # The canonical full-frame pool contains the owner's real face. An empty
    # independent endpoint crop must not be replaced by its predicted rectangle.
    _collect(collector, _endpoint(), FakeReviewer(detections=[]))
    assert collector.resolve([owner], [], {}) == set()


def test_positive_independent_target_localization_can_confirm_a_duplicate():
    collector = EndpointConflictReview(_config(), 30, 300)
    owner = _prepare_frame(collector)
    candidate = _endpoint()
    reviewer = FakeReviewer()
    _collect(collector, candidate, reviewer)

    expected_roi = region_for_box(PREDICTION, collector.revalidation["crop_expansion"])
    assert reviewer.calls == [{
        "shape": (expected_roi[3] - expected_roi[1], expected_roi[2] - expected_roi[0], 3),
        "origin": expected_roi[:2], "max_calls": collector.settings["max_calls_per_frame"],
    }]
    assert collector.resolve([owner], [], {}) == {("endpoint", 10, 1)}
    assert collector.decisions[0]["owner_track_id"] == "owner"
    assert collector.decisions[0]["detection_id"] == "d10:face"


@pytest.mark.parametrize("existing_localization", [False, True])
@pytest.mark.parametrize("target_max_area_ratio,expected_duplicate", [(6.0, True), (2.5, False)])
def test_target_localization_uses_configured_review_geometry(
    existing_localization, target_max_area_ratio, expected_duplicate,
):
    config = _config()
    config["revalidation"]["match_max_area_ratio"] = target_max_area_ratio
    collector = EndpointConflictReview(config, 30, 300)
    owner = _prepare_frame(collector)
    candidate = _endpoint(localized=existing_localization)
    # The face is fully inside a prediction with five times its area. This
    # supports the existing review geometry (IoU 0.2, containment 1.0), but
    # cannot pass the stricter local-face/shared-instance area and IoU gates.
    candidate["box"] = candidate["motion_box"] = [25.0, 30.0, 75.0, 70.0]
    reviewer = FakeReviewer()
    _collect(collector, candidate, reviewer)

    expected = {("endpoint", 10, 1)} if expected_duplicate else set()
    assert collector.resolve([owner], [], {}, endpoint_candidates=[candidate]) == expected
    assert len(reviewer.calls) == (0 if existing_localization else 1)
    assert config["tracking"]["endpoint_conflicts"]["match_max_area_ratio"] == 2.5
    assert config["revalidation"]["match_max_area_ratio"] == target_max_area_ratio


@pytest.mark.parametrize("shared_min_iou,expected_duplicate", [(0.3, True), (0.8, False)])
def test_shared_instance_uses_conflict_geometry_and_owner_covers_actual_local_face(
    shared_min_iou, expected_duplicate,
):
    collector = EndpointConflictReview(_config(match_min_iou=shared_min_iou), 30, 300)
    owner = _prepare_frame(collector, scans=False)
    # The canonical scan is coarser than both independent local detections.
    # The owner covers their actual face, but only 59% of the canonical box.
    scan = {**_face(), "box": [37.0, 37.0, 63.0, 63.0]}
    collector.register_scan(10, [scan])
    _collect(collector, _endpoint(), FakeReviewer())

    expected = {("endpoint", 10, 1)} if expected_duplicate else set()
    assert collector.resolve([owner], [], {}) == expected
    if expected_duplicate:
        assert collector.coverage_references[("endpoint", 10, 1)]["box"] == FACE


@pytest.mark.parametrize("artifacts_level", ["audit", "debug", "final"])
def test_target_selection_audit_preserves_landmarks_and_bounded_raw_candidates(artifacts_level):
    config = _config()
    config["output"]["artifacts_level"] = artifacts_level
    collector = EndpointConflictReview(config, 30, 300)
    owner = _prepare_frame(collector)
    landmarks = [[44.0, 46.0], [56.0, 46.0], [50.0, 50.0], [46.0, 56.0], [54.0, 56.0]]
    detections = [
        {**_face(), "angle": 90, "landmarks": landmarks},
        *[{"box": list(FACE), "confidence": 0.2, "angle": 0} for _ in range(17)],
    ]
    _collect(collector, _endpoint(), FakeReviewer(detections=detections))

    assert collector.resolve([owner], [], {}) == {("endpoint", 10, 1)}
    bundle = collector.bundles[("endpoint", 10, 1)]
    localized = next(item for item in bundle["candidates"] if item["track_id"] == "endpoint")
    assert localized["local_landmarks"] == landmarks
    if artifacts_level == "final":
        assert "localization_selection" not in localized
        assert "localization_candidates" not in localized
        assert collector.decisions == []
        return
    diagnostic = next(
        item for item in collector.decisions[0]["localizations"] if item["track_id"] == "endpoint"
    )
    assert diagnostic["local_landmarks"] == landmarks
    assert diagnostic["localization_selection"] == {
        "status": "supported", "reason": "unique_local_detection",
        "score": pytest.approx(400 / 900), "margin": None,
    }
    raw = diagnostic["localization_candidates"]
    assert len(raw) == 16
    assert raw[0]["box"] == FACE
    assert raw[0]["confidence"] == 0.95
    assert raw[0]["angle"] == 90
    assert raw[0]["landmarks"] == landmarks
    assert raw[0]["frame_idx"] == 10
    assert raw[0]["detection_id"].startswith("r10:")


def test_failed_target_selection_is_auditable_without_inventing_local_evidence():
    config = _config()
    config["revalidation"]["match_max_area_ratio"] = 1.0
    collector = EndpointConflictReview(config, 30, 300)
    owner = _prepare_frame(collector)
    _collect(collector, _endpoint(), FakeReviewer())

    assert collector.resolve([owner], [], {}) == set()
    diagnostic = next(
        item for item in collector.decisions[0]["localizations"] if item["track_id"] == "endpoint"
    )
    assert diagnostic["local_box"] is None
    assert diagnostic["localization_selection"] == {
        "status": "unmatched", "reason": "no_matching_detection", "score": None, "margin": None,
    }
    assert diagnostic["localization_candidates"][0]["box"] == FACE


def test_weak_second_face_from_independent_roi_vetoes_removal_without_audit():
    config = _config()
    config["output"]["artifacts_level"] = "final"
    collector = EndpointConflictReview(config, 30, 300)
    owner = _prepare_frame(collector)
    weak_face = {"box": [60.0, 40.0, 70.0, 60.0], "confidence": 0.27}
    reviewer = FakeReviewer(detections=[_face(), weak_face])
    _collect(collector, _endpoint(), reviewer)

    # The main face positively localizes this endpoint, while the weaker
    # adjacent face cannot be claimed by it and has no accepted core owner.
    # The prediction covers half of that second face, so deletion is unsafe.
    bundle = collector.bundles[("endpoint", 10, 1)]
    localized = next(item for item in bundle["candidates"] if item["track_id"] == "endpoint")
    assert localized["local_box"] == FACE
    assert len(reviewer.calls) == 1
    assert collector.resolve([owner], [], {}) == set()
    assert collector.stats["reason_another_detected_face_has_no_core_coverage"] == 1
    assert collector.decisions == []
    assert "localization_candidates" not in localized


@pytest.mark.parametrize("later_frame", [10, 11])
def test_later_complete_roi_evidence_updates_only_same_frame_final_decision(later_frame):
    config = _config()
    config["output"]["artifacts_level"] = "final"
    collector = EndpointConflictReview(config, 30, 300)
    owner = _prepare_frame(collector)
    _collect(collector, _endpoint(localized=True), FakeReviewer())
    assert collector.resolve([owner], [], {}) == {("endpoint", 10, 1)}

    # This query happens after the earlier endpoint bundle was collected.
    # A complete query on another frame must not change frame 10's verdict.
    reviewer = FakeReviewer(detections=[
        {"box": [60.0, 40.0, 70.0, 60.0], "confidence": 0.27},
    ])
    result = collector._detect(
        later_frame, (20, 20, 80, 80), image=IMAGE, origin=(0, 0),
        decode=_no_decode, reviewer=reviewer,
    )
    assert result is not None and result["complete"]
    # Streaming can release its frame-local scan/core index before final
    # publication; the already collected bundle must retain known evidence.
    collector.prune_before(12)

    expected = set() if later_frame == 10 else {("endpoint", 10, 1)}
    assert collector.resolve([owner], [], {}) == expected
    assert collector.decisions == []


def test_incomplete_positive_recheck_does_not_authorize_suppression():
    collector = EndpointConflictReview(_config(), 30, 300)
    owner = _prepare_frame(collector)
    _collect(collector, _endpoint(), FakeReviewer(complete=False))

    assert collector.stats["incomplete_rechecks"] == 1
    assert collector.cache == {}
    assert collector.resolve([owner], [], {}) == set()


@pytest.mark.parametrize("detections", [[], [_face()]])
def test_roi_cache_includes_empty_results_but_never_crosses_frames(detections):
    collector = EndpointConflictReview(_config(), 30, 300)
    reviewer = FakeReviewer(detections=detections)
    roi = (20, 20, 80, 80)
    kwargs = {"image": IMAGE, "origin": (0, 0), "decode": _no_decode, "reviewer": reviewer}
    first = collector._detect(10, roi, **kwargs)
    second = collector._detect(10, roi, allow_inference=False, **kwargs)
    absent = collector._detect(11, roi, allow_inference=False, **kwargs)
    third = collector._detect(11, roi, **kwargs)

    assert first is second
    assert absent is None
    assert third is not None and third is not first
    assert len(reviewer.calls) == 2
    assert collector.stats["detector_calls"] == 6
    assert set(collector.cache) == {(10, roi), (11, roi)}


def test_collect_can_reuse_complete_roi_cache_after_budget_is_exhausted():
    collector = EndpointConflictReview(_config(max_calls_total=3), 30, 300)
    owner = _prepare_frame(collector)
    candidate = _endpoint()
    roi = region_for_box(candidate["box"], collector.revalidation["crop_expansion"])
    reviewer = FakeReviewer()
    collector._detect(10, roi, image=IMAGE, origin=(0, 0), decode=_no_decode, reviewer=reviewer)
    assert collector.stats["detector_calls"] == collector.call_limit == 3

    _collect(collector, candidate, reviewer)
    assert len(reviewer.calls) == 1
    assert collector.resolve([owner], [], {}) == {("endpoint", 10, 1)}

    # Identical geometry on the next frame cannot consume the previous frame's
    # cached localization after the budget has run out.
    next_owner = _prepare_frame(collector, 11)
    _collect(collector, _endpoint(11), reviewer)
    assert len(reviewer.calls) == 1
    assert collector.resolve([next_owner], [], {}) == set()


def test_cache_capacity_evicts_old_geometry_without_retaining_image_arrays():
    collector = EndpointConflictReview(_config(cache_entries=2), 30, 300)
    reviewer = FakeReviewer()
    roi = (20, 20, 80, 80)
    for frame in (10, 11, 12):
        collector._detect(frame, roi, image=IMAGE, origin=(0, 0), decode=_no_decode, reviewer=reviewer)

    assert list(collector.cache) == [(11, roi), (12, roi)]
    assert collector.stats["peak_cache_entries"] == 2
    assert all(set(entry) == {"detections", "complete", "detector_calls"} for entry in collector.cache.values())


def test_call_allowance_uses_source_duration_including_sub_one_fps():
    collector = EndpointConflictReview(
        _config(max_calls_total=20, max_calls_per_video_second=1.5),
        fps=0.5, frame_count=5,
    )
    assert collector.call_limit == 15  # Five frames are ten source-video seconds.


def test_model_failure_propagates_instead_of_becoming_an_empty_localization():
    collector = EndpointConflictReview(_config(), 30, 300)
    owner = _prepare_frame(collector)
    with pytest.raises(RuntimeError, match="synthetic detector failure"):
        _collect(collector, _endpoint(), FakeReviewer(error=RuntimeError("synthetic detector failure")))
    assert collector.resolve([owner], [], {}) == set()


@pytest.mark.parametrize("final_owner,aliases", [
    ([], {}),
    ([_owner(track_id="endpoint")], {"owner": "endpoint"}),
])
def test_rejected_owner_or_same_track_alias_cannot_replace_an_endpoint(final_owner, aliases):
    collector = EndpointConflictReview(_config(max_calls_total=0), 30, 300)
    _prepare_frame(collector)
    _collect(collector, _endpoint(localized=True), FakeReviewer())

    assert collector.bundles
    assert collector.resolve(final_owner, [], aliases) == set()


def test_final_alias_mapping_preserves_distinct_valid_owner_identity():
    collector = EndpointConflictReview(_config(max_calls_total=0), 30, 300)
    _prepare_frame(collector)
    _collect(collector, _endpoint(localized=True), FakeReviewer())
    aliases = {"owner": "accepted-owner", "endpoint": "accepted-endpoint"}

    assert collector.resolve([_owner(track_id="accepted-owner")], [], aliases) == {
        ("accepted-endpoint", 10, 1),
    }


@pytest.mark.parametrize("has_stale_prediction", [False, True])
def test_streaming_collector_indexes_detector_records_before_prediction_records(has_stale_prediction):
    collector = EndpointConflictReview(_config(max_calls_total=0), 30, 300)
    collector.register_scan(10, [_face()])
    engine = StreamingEngine.__new__(StreamingEngine)
    engine.endpoint_conflicts = collector
    engine.detections = [_owner()]
    engine.candidates = ([{
        "track_id": "owner", "frame_idx": 10,
        "source": "kalman_optical_flow", "box": [70.0, 70.0, 90.0, 90.0],
    }] if has_stale_prediction else [])
    engine.evidence = []
    engine._conflict_detection_cursor = 0
    engine._conflict_candidate_cursor = 0
    engine._conflict_evidence_cursor = 0
    engine._decode_frames = _no_decode
    engine.reviewer = FakeReviewer()

    engine._collect_endpoint_conflict_evidence(_endpoint(localized=True), IMAGE, (0, 0))

    assert collector.core[10]["owner"]["source"] == "detector"
    assert collector.core[10]["owner"]["box"] == FACE
    assert collector.resolve([_owner()], [], {}) == {("endpoint", 10, 1)}
    assert engine.reviewer.calls == []


@pytest.mark.parametrize("change_geometry,change_localization", [
    (True, False), (False, True), (True, True),
])
def test_resolution_uses_final_endpoint_geometry_and_localization(change_geometry, change_localization):
    collector = EndpointConflictReview(_config(max_calls_total=0), 30, 300)
    owner = _prepare_frame(collector)
    candidate = _endpoint(localized=True)
    _collect(collector, candidate, FakeReviewer())
    assert collector.resolve([owner], [], {}) == {("endpoint", 10, 1)}

    corrected = deepcopy(candidate)
    if change_geometry:
        # Publication reads motion_box, even if the earlier box has not yet
        # been replaced. Final resolution must use that same geometry.
        corrected["motion_box"] = [62.0, 37.0, 88.0, 63.0]
    if change_localization:
        corrected["_review_measurement"]["local_box"] = [65.0, 40.0, 85.0, 60.0]

    assert collector.resolve([owner], [], {}, endpoint_candidates=[corrected]) == set()


@pytest.mark.parametrize("final_landmarks", [None, [[45.0, 45.0], [55.0, 45.0]]])
def test_final_localization_refreshes_landmarks_in_audit(final_landmarks):
    collector = EndpointConflictReview(_config(max_calls_total=0), 30, 300)
    owner = _prepare_frame(collector)
    candidate = _endpoint(localized=True)
    candidate["_review_measurement"]["local_landmarks"] = [[1.0, 2.0], [3.0, 4.0]]
    _collect(collector, candidate, FakeReviewer())
    corrected = deepcopy(candidate)
    corrected["_review_measurement"]["local_box"] = [39.0, 40.0, 60.0, 60.0]
    corrected["_review_measurement"]["local_landmarks"] = final_landmarks

    assert collector.resolve([owner], [], {}, endpoint_candidates=[corrected]) == {("endpoint", 10, 1)}
    diagnostic = next(
        item for item in collector.decisions[0]["localizations"] if item["track_id"] == "endpoint"
    )
    assert diagnostic["local_box"] == corrected["_review_measurement"]["local_box"]
    assert diagnostic["local_landmarks"] == final_landmarks


def test_publisher_rechecks_the_final_winning_endpoint_candidate():
    config = _config(max_calls_total=0)
    collector = EndpointConflictReview(config, 30, 300)
    owner = _prepare_frame(collector, 2)
    collected = _endpoint(2, anchor=0, localized=True)
    _collect(collector, collected, FakeReviewer())
    corrected = deepcopy(collected)
    corrected["inlier_points"] += 5
    corrected["motion_box"] = [62.0, 37.0, 88.0, 63.0]
    corrected["_review_measurement"]["local_box"] = [65.0, 40.0, 85.0, 60.0]
    engine = StreamingEngine.__new__(StreamingEngine)
    engine.config = config
    engine.endpoint_conflicts = collector
    engine._endpoint_conflict_aliases = {}
    engine.endpoint_affine_published_frames = 0
    engine.interpolate_endpoint_published_frames = 0
    endpoint_track = {"track_id": "endpoint", "accepted": True, "accepted_intervals": [[0, 0]]}
    engine.tracks = [endpoint_track, {"track_id": "owner", "accepted": True, "accepted_intervals": [[2, 2]]}]
    engine.endpoint_affine_candidates = [
        _endpoint(1, anchor=0, localized=True), collected, corrected,
        _endpoint(3, anchor=0, localized=True),
    ]
    observations = [{"track_id": "endpoint", "frame_idx": 0, "box": list(PREDICTION), "source": "detector"}, owner]
    evidence = deepcopy(observations)

    published = engine._publish_endpoint_affine_candidates(observations, evidence)

    final_endpoint = next(item for item in published if item["track_id"] == "endpoint" and item["frame_idx"] == 2)
    assert final_endpoint["box"] == corrected["motion_box"]
    assert final_endpoint["local_box"] == corrected["_review_measurement"]["local_box"]
    assert endpoint_track["accepted_intervals"] == [[0, 3]]
    assert collector.stats["suppressed_frames"] == 0
    assert _accepted_interval_coverage(engine.tracks, published, evidence) == (5, 0, 0)


def test_suppressed_endpoint_localization_stays_covered_after_owner_stabilization():
    config = _config(max_calls_total=0)
    collector = EndpointConflictReview(config, 30, 300)
    owner = _prepare_frame(collector, 1)
    candidate = _endpoint(1, anchor=0, localized=True)
    actual_face = [39.0, 40.0, 60.0, 60.0]
    candidate["_review_measurement"]["local_box"] = actual_face
    _collect(collector, candidate, FakeReviewer())
    engine = StreamingEngine.__new__(StreamingEngine)
    engine.config = config
    engine.endpoint_conflicts = collector
    engine._endpoint_conflict_aliases = {}
    engine.endpoint_affine_published_frames = 0
    engine.interpolate_endpoint_published_frames = 0
    engine.tracks = [
        {"track_id": "endpoint", "accepted": True, "accepted_intervals": [[0, 0]]},
        {"track_id": "owner", "accepted": True, "accepted_intervals": [[1, 1]]},
    ]
    engine.endpoint_affine_candidates = [candidate]
    anchor = {"track_id": "endpoint", "frame_idx": 0, "box": list(PREDICTION), "source": "detector"}
    observations = [anchor, owner]
    inner_box = [42.0, 42.0, 58.0, 58.0]
    evidence = [deepcopy(anchor), {**deepcopy(owner), "box": inner_box}]
    original_evidence = deepcopy(evidence)

    published = engine._publish_endpoint_affine_candidates(observations, evidence)

    assert not any(item["track_id"] == "endpoint" and item["frame_idx"] == 1 for item in published)
    assert any(item["track_id"] == "owner" and item["frame_idx"] == 1 and item["box"] == actual_face for item in evidence)
    stabilized = deepcopy(published)
    stabilized_owner = next(item for item in stabilized if item["track_id"] == "owner")
    stabilized_owner["box"] = list(inner_box)
    stabilized_owner["motion_box"] = list(inner_box)
    stabilized_owner["raw_output_box"] = list(FACE)

    without_transferred_reference = _validate_final_track_geometry(stabilized, original_evidence)
    assert next(item for item in without_transferred_reference if item["track_id"] == "owner")["box"] == inner_box
    protected = _validate_final_track_geometry(stabilized, evidence)
    protected_owner = next(item for item in protected if item["track_id"] == "owner")
    assert protected_owner["box"] == FACE
    assert protected_owner["box_stabilization"] == "raw_evidence_fallback"
    assert _accepted_interval_coverage(engine.tracks, protected, evidence) == (2, 0, 0)


def test_covered_secondary_face_keeps_its_owner_coverage_after_suppression():
    config = _config()
    collector = EndpointConflictReview(config, 30, 300)
    owner = _prepare_frame(collector, 1)
    second_face = [60.0, 40.0, 70.0, 60.0]
    secondary = {
        **_owner(1, track_id="secondary"),
        "box": list(second_face), "detector_box": list(second_face),
    }
    collector.core[1]["secondary"] = secondary
    collector.evidence[("secondary", 1)] = {
        "local_box": list(second_face), "local_confidence": 0.95,
    }
    candidate = _endpoint(1, anchor=0)
    _collect(collector, candidate, FakeReviewer(detections=[
        _face(1), {"box": list(second_face), "confidence": 0.27},
    ]))
    engine = StreamingEngine.__new__(StreamingEngine)
    engine.config = config
    engine.endpoint_conflicts = collector
    engine._endpoint_conflict_aliases = {}
    engine.endpoint_affine_published_frames = 0
    engine.interpolate_endpoint_published_frames = 0
    engine.tracks = [
        {"track_id": "endpoint", "accepted": True, "accepted_intervals": [[0, 0]]},
        {"track_id": "owner", "accepted": True, "accepted_intervals": [[1, 1]]},
        {"track_id": "secondary", "accepted": True, "accepted_intervals": [[1, 1]]},
    ]
    engine.endpoint_affine_candidates = [candidate]
    anchor = {"track_id": "endpoint", "frame_idx": 0, "box": list(PREDICTION), "source": "detector"}
    # Area 112: the raw face (area 200) stays within the 2.5x size cap
    # relative to this original evidence, while smoothing covers only 56%
    # of the additional real-face reference and must therefore fall back.
    inner_box = [61.0, 43.0, 69.0, 57.0]
    observations = [anchor, owner, secondary]
    evidence = [deepcopy(anchor), deepcopy(owner), {**deepcopy(secondary), "box": list(inner_box)}]
    original_evidence = deepcopy(evidence)

    published = engine._publish_endpoint_affine_candidates(observations, evidence)

    assert not any(item["track_id"] == "endpoint" and item["frame_idx"] == 1 for item in published)
    assert any(item["track_id"] == "secondary" and item["box"] == second_face for item in evidence)
    stabilized = deepcopy(published)
    stabilized_secondary = next(item for item in stabilized if item["track_id"] == "secondary")
    stabilized_secondary.update(box=list(inner_box), motion_box=list(inner_box), raw_output_box=list(second_face))
    unprotected = _validate_final_track_geometry(stabilized, original_evidence)
    assert next(item for item in unprotected if item["track_id"] == "secondary")["box"] == inner_box
    protected = _validate_final_track_geometry(stabilized, evidence)
    protected_secondary = next(item for item in protected if item["track_id"] == "secondary")
    assert protected_secondary["box"] == second_face
    assert protected_secondary["box_stabilization"] == "raw_evidence_fallback"
    assert _accepted_interval_coverage(engine.tracks, protected, evidence) == (3, 0, 0)


@pytest.mark.parametrize("direction,anchor,frames,expected_intervals", [
    (1, 0, [1, 2, 3], [[0, 1], [3, 3]]),
    (-1, 4, [3, 2, 1], [[1, 1], [3, 4]]),
])
def test_suppressing_middle_endpoint_preserves_later_frames_and_exact_accepted_intervals(
    direction, anchor, frames, expected_intervals,
):
    config = _config(max_calls_total=0)
    collector = EndpointConflictReview(config, 30, 300)
    owner = _prepare_frame(collector, 2)
    candidate = _endpoint(2, direction=direction, anchor=anchor, localized=True)
    _collect(collector, candidate, FakeReviewer())
    engine = StreamingEngine.__new__(StreamingEngine)
    engine.config = config
    engine.endpoint_conflicts = collector
    engine._endpoint_conflict_aliases = {}
    engine.endpoint_affine_published_frames = 0
    engine.interpolate_endpoint_published_frames = 0
    endpoint_track = {"track_id": "endpoint", "accepted": True, "accepted_intervals": [[anchor, anchor]]}
    owner_track = {"track_id": "owner", "accepted": True, "accepted_intervals": [[2, 2]]}
    engine.tracks = [endpoint_track, owner_track]
    engine.endpoint_affine_candidates = [
        _endpoint(frame, direction=direction, anchor=anchor, localized=True) for frame in frames
    ]
    observations = [
        {"track_id": "endpoint", "frame_idx": anchor, "box": list(PREDICTION), "source": "detector"},
        owner,
    ]
    evidence = deepcopy(observations)

    published = engine._publish_endpoint_affine_candidates(observations, evidence)

    endpoint_frames = {item["frame_idx"] for item in published if item["track_id"] == "endpoint"}
    assert endpoint_frames == {anchor, 1, 3}
    assert ("owner", 2) in {(item["track_id"], item["frame_idx"]) for item in published}
    assert endpoint_track["accepted_intervals"] == expected_intervals
    assert not any(item["track_id"] == "endpoint" and item["frame_idx"] == 2 for item in evidence)
    assert engine.endpoint_affine_published_frames == 2
    assert engine.interpolate_endpoint_published_frames == 2
    assert collector.stats["suppressed_frames"] == 1
    # No coverage exception or cross-track allowance is needed: suppressed
    # endpoint frames never become requirements in that track's intervals.
    assert _accepted_interval_coverage(engine.tracks, published, evidence) == (4, 0, 0)
