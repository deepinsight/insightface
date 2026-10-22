from copy import deepcopy

import pytest

from insightface.app.privateframe.endpoint_matching import (
    find_conflict_groups,
    localization_match_settings,
    match_endpoint_candidates,
    plan_rechecks,
    region_for_box,
    select_local_face,
)


@pytest.fixture
def settings():
    return {
        "nearby_iou": 0.1,
        "nearby_center_distance": 1.0,
        "match_min_iou": 0.3,
        "match_max_center_distance": 0.5,
        "match_max_area_ratio": 2.5,
        "match_min_confidence": 0.35,
        "match_min_margin": 0.1,
        "equivalence_iou": 0.9,
        "recheck_min_frame_gap": 2,
        "max_calls_per_frame": 6,
    }


def face(identifier="face", box=(0, 0, 100, 100), confidence=0.9, **extra):
    return {"detection_id": identifier, "box": list(box), "confidence": confidence, **extra}


def candidate(track, *, box=(0, 0, 100, 100), local=(0, 0, 100, 100), endpoint=False, **extra):
    item = {"track_id": track, "frame_idx": 10, "scene_segment_id": 0,
            "box": list(box), "source": "detector" if not endpoint else "kalman_optical_flow",
            "local_box": list(local) if local is not None else None, "local_confidence": 0.9,
            **extra}
    if endpoint:
        item["endpoint_repair"] = "affine_ransac"
        item["direction"] = -1
    return item


def resolve(items, detections, settings, owners=frozenset({"core"}), **kwargs):
    return {item["track_id"]: item for item in match_endpoint_candidates(
        items, detections, settings=settings, valid_owner_track_ids=owners, **kwargs,
    )}


def test_groups_require_nearby_distinct_tracks_and_an_endpoint(settings):
    items = [candidate("core"), candidate("repair", endpoint=True),
             candidate("far", endpoint=True, box=(1000, 0, 1100, 100)),
             candidate("other_scene", endpoint=True, scene_segment_id=1),
             candidate("other_frame", endpoint=True, frame_idx=11)]
    groups = find_conflict_groups(items, settings)
    assert len(groups) == 1
    assert groups[0]["track_ids"] == ["core", "repair"]
    assert groups[0]["box"] == [0, 0, 100, 100]
    assert groups == find_conflict_groups(list(reversed(items)), settings)
    assert find_conflict_groups([candidate("a"), candidate("b")], settings) == []
    assert find_conflict_groups([candidate("same"), candidate("same", endpoint=True)], settings) == []


def test_region_bounds_preserve_padding_outside_frame():
    assert region_for_box([0, 0, 10, 20], 2.0) == (-15, -10, 25, 30)
    assert region_for_box([1, 1, 3, 3], 1.0) == (-2, -2, 6, 6)
    with pytest.raises(ValueError, match="valid box"):
        region_for_box([0, 0, float("nan"), 5], 1.0)


def test_select_local_face_requires_unique_geometry_not_just_confidence(settings):
    choices = [face("left", (-10, 0, 90, 100), 0.99), face("right", (10, 0, 110, 100), 0.6)]
    result = select_local_face([0, 0, 100, 100], choices, settings)
    assert result["status"] == "ambiguous"
    assert result["detection"] is None
    assert result["margin"] == pytest.approx(0)
    assert result == select_local_face([0, 0, 100, 100], list(reversed(choices)), settings)


def test_local_selection_keeps_all_distinct_faces_available(settings):
    choices = [face("left"), face("right", (120, 0, 220, 100))]
    before = deepcopy(choices)
    assert select_local_face([0, 0, 100, 100], choices, settings)["detection"]["detection_id"] == "left"
    assert select_local_face([120, 0, 220, 100], choices, settings)["detection"]["detection_id"] == "right"
    assert choices == before


@pytest.mark.parametrize("choices", [[], [face(confidence=0.1)], [face(box=(500, 0, 600, 100))],
                                      [face(box=(0, 0, float("nan"), 100))]])
def test_unmatched_local_detection_is_not_duplicate(settings, choices):
    assert select_local_face([0, 0, 100, 100], choices, settings)["status"] == "unmatched"


def test_independent_localizations_allow_same_instance_endpoint_suppression(settings):
    items = [candidate("core"), candidate("repair", endpoint=True, box=(-10, -10, 110, 110))]
    result = resolve(items, [face()], settings)
    assert result["core"]["status"] == "supported"
    assert result["repair"]["status"] == "duplicate"
    assert result["repair"]["owner_track_id"] == "core"
    assert result["repair"]["detection_id"] == "face"
    assert result == resolve(list(reversed(items)), [face()], settings)


def test_one_face_in_shared_crop_is_not_endpoint_localization(settings):
    result = resolve([candidate("core"), candidate("repair", endpoint=True, local=None)], [face()], settings)
    assert result["repair"]["status"] == "ambiguous"
    assert result["repair"]["reason"] == "missing_candidate_localization"


def test_true_two_face_instances_remain_separate(settings):
    second = (60, -30, 160, 70)
    result = resolve([candidate("core"), candidate("repair", endpoint=True, box=second, local=second)],
                     [face("one"), face("two", second)], settings)
    assert result["core"]["detection_id"] == "one"
    assert result["repair"]["detection_id"] == "two"
    assert result["repair"]["status"] == "supported"
    assert result["repair"]["owner_track_id"] is None


def test_unmatched_endpoint_is_retained_even_when_owner_has_a_detection(settings):
    second = (120, 0, 220, 100)
    result = resolve([candidate("core"), candidate("repair", endpoint=True, box=second, local=second)], [face()], settings)
    assert result["repair"]["status"] == "unmatched"
    assert result["repair"]["owner_track_id"] is None


@pytest.mark.parametrize("owners", [frozenset(), frozenset({"not_in_this_frame"})])
def test_rejected_or_absent_owner_cannot_suppress_endpoint(settings, owners):
    result = resolve([candidate("core"), candidate("repair", endpoint=True)], [face()], settings, owners)
    assert result["repair"]["status"] == "supported"
    assert result["repair"]["reason"] == "no_valid_core_owner"


def test_endpoints_cannot_suppress_each_other(settings):
    result = resolve([candidate("core", endpoint=True), candidate("repair", endpoint=True)], [face()], settings)
    assert all(item["status"] == "supported" for item in result.values())


def test_incomplete_multi_angle_evidence_never_suppresses(settings):
    result = resolve([candidate("core"), candidate("repair", endpoint=True)], [face()], settings, evidence_complete=False)
    assert all(item["status"] == "ambiguous" for item in result.values())


def test_two_plausible_pool_instances_remain_ambiguous(settings):
    result = resolve([candidate("core"), candidate("repair", endpoint=True)],
                     [face("a", (-1, 0, 99, 100)), face("b", (1, 0, 101, 100))], settings)
    assert all(item["status"] == "ambiguous" for item in result.values())


def test_owner_must_cover_actual_face_not_merely_intersect_it(settings):
    result = resolve([candidate("core", box=(0, 0, 50, 100)), candidate("repair", endpoint=True)], [face()], settings)
    assert result["repair"]["status"] != "duplicate"


def test_owner_must_cover_endpoint_localization_even_if_canonical_agrees(settings):
    result = resolve([candidate("core", box=(0, 0, 90, 100)),
                      candidate("repair", endpoint=True, local=(2, 0, 102, 100))], [face()], settings)
    assert result["repair"]["status"] == "supported"
    assert result["repair"]["reason"] == "no_valid_core_owner"


def test_independent_localizations_must_agree_with_each_other(settings):
    result = resolve([candidate("core", local=(-4, 0, 96, 100)),
                      candidate("repair", endpoint=True, local=(4, 0, 104, 100))], [face()], settings)
    assert result["repair"]["status"] == "supported"
    assert result["repair"]["reason"] == "no_valid_core_owner"


def test_pool_extent_can_differ_while_independent_localizations_agree(settings):
    items = [candidate("core", box=(-15, -15, 115, 115)),
             candidate("repair", endpoint=True, box=(-15, -15, 115, 115))]
    result = resolve(items, [face(box=(-10, -10, 110, 110))], settings)
    assert result["repair"]["status"] == "duplicate"


def test_coarse_canonical_extent_is_not_a_second_coverage_requirement(settings):
    result = resolve([candidate("core"), candidate("repair", endpoint=True)],
                     [face(box=(0, -20, 100, 120))], settings)
    assert result["repair"]["status"] == "duplicate"


def test_localization_settings_inherit_review_geometry_without_changing_evidence_gates(settings):
    review = {"match_min_iou": 0.08, "match_min_containment": 0.35,
              "match_max_center_distance": 0.85, "match_max_area_ratio": 6.0,
              "confidence_threshold": 0.18}
    original = deepcopy(settings)
    derived = localization_match_settings(review, settings)
    assert derived["match_min_iou"] == 0.08
    assert derived["match_min_containment"] == 0.35
    assert derived["match_max_center_distance"] == 0.85
    assert derived["match_max_area_ratio"] == 6.0
    assert derived["match_min_confidence"] == settings["match_min_confidence"]
    assert derived["match_min_margin"] == settings["match_min_margin"]
    assert settings == original
    assert localization_match_settings({}, {}) == {}


def test_prediction_drift_uses_existing_review_search_but_still_needs_local_evidence(settings):
    target_settings = localization_match_settings(
        {"match_min_iou": 0.08, "match_min_containment": 0.35,
         "match_max_center_distance": 0.85, "match_max_area_ratio": 6.0}, settings)
    actual = (0, 50, 100, 150)
    drifted = (-20, -80, 140, 140)
    assert select_local_face(drifted, [face(box=actual)], settings)["status"] == "unmatched"
    assert select_local_face(drifted, [face(box=actual)], target_settings)["status"] == "supported"
    items = [candidate("core", box=actual, local=actual),
             candidate("repair", endpoint=True, box=drifted, local=actual)]
    result = resolve(items, [face(box=actual)], settings, candidate_match_settings=target_settings)
    assert result["repair"]["status"] == "duplicate"
    items[-1]["local_box"] = None
    result = resolve(items, [face(box=actual)], settings, candidate_match_settings=target_settings)
    assert result["repair"]["status"] == "ambiguous"


def test_broader_target_search_does_not_relax_canonical_instance_matching(settings):
    target_settings = localization_match_settings(
        {"match_min_iou": 0.08, "match_min_containment": 0.35,
         "match_max_center_distance": 0.85, "match_max_area_ratio": 6.0}, settings)
    result = resolve([candidate("core"), candidate("repair", endpoint=True)],
                     [face(box=(70, 0, 170, 100))], settings, candidate_match_settings=target_settings)
    assert result["repair"]["status"] == "unmatched"


def test_local_search_can_use_containment_but_keeps_ambiguity_guard(settings):
    target_settings = {**settings, "match_min_iou": 0.8, "match_min_containment": 0.9}
    contained = [face("one", (10, 10, 90, 90))]
    assert select_local_face([0, 0, 100, 100], contained, target_settings)["status"] == "supported"
    contained.append(face("two", (11, 10, 91, 90)))
    assert select_local_face([0, 0, 100, 100], contained, target_settings)["status"] == "ambiguous"


def test_endpoint_covering_another_unclaimed_detected_face_is_not_suppressed(settings):
    second = (100, 0, 200, 100)
    items = [candidate("core"), candidate("repair", endpoint=True, box=(0, 0, 180, 100))]
    result = resolve(items, [face("one"), face("two", second)], settings)
    assert result["repair"]["status"] == "ambiguous"
    assert result["repair"]["reason"] == "another_detected_face_has_no_core_coverage"
    items.append(candidate("second_core", box=second, local=second))
    result = resolve(items, [face("one"), face("two", second)], settings,
                     frozenset({"core", "second_core"}))
    assert result["repair"]["status"] == "duplicate"


def test_weak_second_detected_face_vetoes_suppression_without_supporting_a_claim(settings):
    items = [candidate("core"), candidate("repair", endpoint=True, box=(0, 0, 180, 100))]
    second = face("possible_second", (100, 0, 200, 100), confidence=0.278)
    assert select_local_face(second["box"], [second], settings)["status"] == "unmatched"
    result = resolve(items, [face("one"), second], settings)
    assert result["repair"]["status"] == "ambiguous"
    assert result["repair"]["reason"] == "another_detected_face_has_no_core_coverage"


def test_additional_weak_face_missing_from_shared_pool_still_vetoes(settings):
    items = [candidate("core"), candidate("repair", endpoint=True, box=(0, 0, 180, 100))]
    extra = face("extra", (100, 0, 200, 100), confidence=0.278, frame_idx=10)
    result = resolve(items, [face()], settings, additional_detections=[extra])
    assert result["repair"]["status"] == "ambiguous"
    assert result["repair"]["additional_detection_id"] == "extra"
    assert result["repair"]["additional_canonical_status"] == "unmatched"


def test_additional_detections_cannot_supply_positive_canonical_claims(settings):
    result = resolve([candidate("core"), candidate("repair", endpoint=True)], [], settings,
                     additional_detections=[face("only_in_independent_crop")])
    assert all(item["status"] == "unmatched" for item in result.values())


def test_additional_same_face_crop_inside_core_coverage_does_not_block(settings):
    items = [candidate("core", box=(-10, -10, 110, 110)), candidate("repair", endpoint=True)]
    result = resolve(items, [face()], settings,
                     additional_detections=[face("crop_variant", (-2, -2, 102, 102), confidence=0.278)])
    assert result["repair"]["status"] == "duplicate"


def test_unique_canonical_match_does_not_hide_uncovered_nearby_extra_face(settings):
    items = [candidate("core"), candidate("repair", endpoint=True, box=(0, 0, 150, 100))]
    extra = face("nearby", (25, 0, 125, 100), confidence=0.278)
    assert select_local_face(extra["box"], [face()], settings)["status"] == "supported"
    result = resolve(items, [face()], settings, additional_detections=[extra])
    assert result["repair"]["status"] == "ambiguous"
    assert result["repair"]["additional_canonical_status"] == "supported"
    assert result["repair"]["additional_canonical_detection_id"] == "face"


def test_additional_same_canonical_id_is_not_a_second_record(settings):
    coarse = face(box=(0, -20, 100, 120))
    result = resolve([candidate("core"), candidate("repair", endpoint=True)], [coarse], settings,
                     additional_detections=[dict(coarse)])
    assert result["repair"]["status"] == "duplicate"


@pytest.mark.parametrize("extra", [face("other_frame", (100, 0, 200, 100), frame_idx=11),
                                  face("other_scene", (100, 0, 200, 100), scene_segment_id=1)])
def test_additional_face_from_other_context_is_not_reused(settings, extra):
    result = resolve([candidate("core"), candidate("repair", endpoint=True, box=(0, 0, 180, 100))],
                     [face()], settings, additional_detections=[extra])
    assert result["repair"]["status"] == "duplicate"


def test_extra_face_outside_endpoint_was_not_covered_by_that_endpoint(settings):
    result = resolve([candidate("core"), candidate("repair", endpoint=True)], [face()], settings,
                     additional_detections=[face("outside", (120, 0, 220, 100))])
    assert result["repair"]["status"] == "duplicate"


def test_unrelated_frame_and_scene_evidence_is_not_reused(settings):
    items = [candidate("core"), candidate("repair", endpoint=True)]
    for pool in ([face(frame_idx=11)], [face(scene_segment_id=1)]):
        result = resolve(items, pool, settings)
        assert all(item["status"] == "unmatched" for item in result.values())
    with pytest.raises(ValueError, match="one frame and one scene"):
        resolve([candidate("core"), candidate("repair", endpoint=True, scene_segment_id=1)], [face()], settings)


def test_recheck_planning_respects_complete_call_budget_and_history(settings):
    group = find_conflict_groups([candidate("core"), candidate("repair", endpoint=True)], settings)[0]
    history = {(0, ("core", "repair")): 9}
    assert plan_rechecks([group], frame_idx=10, history=history, remaining_calls=6,
                         settings=settings, calls_per_recheck=3) == []
    assert history == {(0, ("core", "repair")): 9}
    assert plan_rechecks([group], frame_idx=10, history={}, remaining_calls=2,
                         settings=settings, calls_per_recheck=3) == []
    requests = plan_rechecks([group, group], frame_idx=10, history={}, remaining_calls=100,
                             settings=settings, calls_per_recheck=3)
    assert len(requests) == 1
    assert requests[0]["estimated_calls"] == 3
    assert requests[0]["history_key"] == (0, ("core", "repair"))


def test_raw_detector_localization_is_supported_but_prediction_is_not(settings):
    core = candidate("core", local=None, detector_box=[0, 0, 100, 100], confidence=0.9)
    result = resolve([core, candidate("repair", endpoint=True)], [face()], settings)
    assert result["repair"]["status"] == "duplicate"


def test_owner_choice_is_deterministic_and_always_a_real_core_candidate(settings):
    items = [candidate("core_b"), candidate("repair", endpoint=True), candidate("core_a")]
    result = resolve(items, [face()], settings, frozenset({"core_a", "core_b"}))
    assert result["repair"]["owner_track_id"] == "core_a"
    assert result == resolve(list(reversed(items)), [face()], settings, frozenset({"core_a", "core_b"}))
