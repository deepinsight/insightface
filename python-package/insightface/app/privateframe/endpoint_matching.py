"""Pure geometry and positive-evidence matching for endpoint conflicts.

Predicted boxes only nominate conflicts. Suppression requires independent
localizations for both tracks and an already-published core owner. A missing
or ambiguous localization never means that an endpoint is redundant.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set
from typing import Any

from .geometry import area, area_ratio, containment, covers_reference, intersection, iou, normalized_center_distance


def _box(value: Any) -> list[float] | None:
    try:
        result = [float(component) for component in value]
    except (TypeError, ValueError):
        return None
    if len(result) != 4 or not all(math.isfinite(component) for component in result):
        return None
    return result if result[2] > result[0] and result[3] > result[1] else None


def _confidence(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return result if math.isfinite(result) else -math.inf


def _endpoint(item: Mapping[str, Any]) -> bool:
    return bool(item.get("endpoint_repair") or item.get("endpoint", False))


def _candidate_id(item: Mapping[str, Any]) -> str:
    return str(item.get("candidate_id") or (
        f"{item['track_id']}:{int(item['frame_idx'])}:{int(item.get('direction', 0))}"
    ))


def region_for_box(
    value: Sequence[float], expansion: float, minimum_side: int = 8,
) -> tuple[int, int, int, int]:
    """Return the exact padded-square bounds used by local face review."""

    target = _box(value)
    if target is None or not math.isfinite(expansion) or expansion <= 0:
        raise ValueError("review region requires a valid box and positive expansion")
    if isinstance(minimum_side, bool) or not isinstance(minimum_side, int) or minimum_side <= 0:
        raise ValueError("minimum_side must be a positive integer")
    side = max(minimum_side, math.ceil(max(target[2] - target[0], target[3] - target[1]) * expansion))
    x1 = math.floor((target[0] + target[2] - side) * 0.5)
    y1 = math.floor((target[1] + target[3] - side) * 0.5)
    return x1, y1, x1 + side, y1 + side


def localization_match_settings(
    revalidation: Mapping[str, Any], conflict_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Reuse local-review search geometry, retaining conflict evidence gates.

    Prediction drift must not impose a narrower face-search corridor than the
    existing reviewer. This only controls localization relative to a predicted
    box: canonical instance matching and independent-localization equivalence
    still use their separate conflict settings. Empty disabled configurations
    can be constructed without introducing another copy of Base defaults.
    """

    result = dict(conflict_settings)
    for key in ("match_min_iou", "match_min_containment", "match_max_center_distance", "match_max_area_ratio"):
        value = revalidation.get(key)
        if value is not None:
            result[key] = value
    return result


def find_conflict_groups(
    candidates: Sequence[Mapping[str, Any]], settings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Group nearby different tracks, separately for every frame and scene."""

    by_context: dict[tuple[int, Any], list[dict[str, Any]]] = {}
    for candidate in candidates:
        value = _box(candidate.get("box"))
        if value is None:
            continue
        item = dict(candidate)
        item["box"] = value
        by_context.setdefault((int(item["frame_idx"]), item.get("scene_segment_id")), []).append(item)
    groups: list[dict[str, Any]] = []
    for (frame_idx, scene), items in sorted(by_context.items(), key=lambda entry: (entry[0][0], str(entry[0][1]))):
        items.sort(key=_candidate_id)
        links = [set() for _item in items]
        for index, first in enumerate(items):
            for other in range(index + 1, len(items)):
                second = items[other]
                if str(first["track_id"]) == str(second["track_id"]):
                    continue
                if (iou(first["box"], second["box"]) >= float(settings["nearby_iou"])
                        or normalized_center_distance(first["box"], second["box"])
                        <= float(settings["nearby_center_distance"])):
                    links[index].add(other)
                    links[other].add(index)
        visited: set[int] = set()
        for index in range(len(items)):
            if index in visited:
                continue
            pending, members = [index], []
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                visited.add(current)
                members.append(items[current])
                pending.extend(sorted(links[current] - visited, reverse=True))
            if len(members) < 2 or not any(_endpoint(item) for item in members):
                continue
            members.sort(key=_candidate_id)
            groups.append({
                "frame_idx": frame_idx,
                "scene_segment_id": scene,
                "track_ids": sorted({str(item["track_id"]) for item in members}),
                "candidate_ids": [_candidate_id(item) for item in members],
                "candidates": members,
                "box": [min(item["box"][0] for item in members), min(item["box"][1] for item in members),
                        max(item["box"][2] for item in members), max(item["box"][3] for item in members)],
            })
    return groups


def plan_rechecks(
    groups: Sequence[Mapping[str, Any]], *, frame_idx: int,
    history: Mapping[tuple[Any, tuple[str, ...]], int], remaining_calls: int,
    settings: Mapping[str, Any], calls_per_recheck: int = 1,
) -> list[dict[str, Any]]:
    """Plan complete rechecks without mutating history or consuming budget."""

    if isinstance(calls_per_recheck, bool) or not isinstance(calls_per_recheck, int) or calls_per_recheck <= 0:
        raise ValueError("calls_per_recheck must be a positive integer")
    budget = max(0, min(int(remaining_calls), int(settings["max_calls_per_frame"])))
    requests = []
    seen = set()
    for group in sorted(groups, key=lambda value: (str(value.get("scene_segment_id")), tuple(value["track_ids"]))):
        if int(group["frame_idx"]) != int(frame_idx):
            continue
        key = (group.get("scene_segment_id"), tuple(sorted(str(value) for value in group["track_ids"])))
        if key in seen:
            continue
        seen.add(key)
        previous = history.get(key)
        if previous is not None and abs(int(frame_idx) - int(previous)) < int(settings["recheck_min_frame_gap"]):
            continue
        if budget < calls_per_recheck:
            break
        requests.append({**group, "history_key": key, "estimated_calls": calls_per_recheck})
        budget -= calls_per_recheck
    return requests


def select_local_face(
    target: Sequence[float], detections: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Select a uniquely localized face; confidence cannot break geometric ties."""

    target_box = _box(target)
    result: dict[str, Any] = {"status": "unmatched", "reason": "no_matching_detection", "detection": None,
                              "score": None, "margin": None}
    if target_box is None:
        return {**result, "reason": "invalid_target_geometry"}
    ranked = []
    for detection in detections:
        value = _box(detection.get("box"))
        confidence = _confidence(detection.get("confidence"))
        if value is None or confidence < float(settings["match_min_confidence"]):
            continue
        overlap = iou(target_box, value)
        distance = normalized_center_distance(target_box, value)
        contained = (settings.get("match_min_containment") is not None
                     and containment(target_box, value) >= float(settings["match_min_containment"]))
        if ((overlap < float(settings["match_min_iou"]) and not contained)
                or distance > float(settings["match_max_center_distance"])
                or area_ratio(target_box, value) > float(settings["match_max_area_ratio"])):
            continue
        ranked.append((overlap, distance, confidence, tuple(value), str(detection.get("detection_id", "")), dict(detection)))
    ranked.sort(key=lambda value: (-value[0], value[1], -value[2], value[3], value[4]))
    if not ranked:
        return result
    best = ranked[0]
    margin = best[0] - ranked[1][0] if len(ranked) > 1 else None
    if margin is not None and margin < float(settings["match_min_margin"]):
        return {**result, "status": "ambiguous", "reason": "multiple_plausible_detections", "score": best[0], "margin": margin}
    return {"status": "supported", "reason": "unique_local_detection", "detection": best[-1], "score": best[0], "margin": margin}


def match_endpoint_candidates(
    candidates: Sequence[Mapping[str, Any]], detections: Sequence[Mapping[str, Any]], *,
    valid_owner_track_ids: Set[str], settings: Mapping[str, Any],
    frame_idx: int | None = None, scene_segment_id: int | None = None,
    evidence_complete: bool = True,
    candidate_match_settings: Mapping[str, Any] | None = None,
    additional_detections: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Resolve same-instance claims against final, same-frame core owners.

    The caller supplies only genuinely observed ``local_box`` measurements,
    never a copy of a predicted box. Full-scan detector geometry is also valid
    localization evidence. Owners must already have survived core admission
    and have an observation at this frame; endpoint-to-endpoint suppression
    is deliberately unsupported so no dependency cycle can hide a face.
    ``additional_detections`` from independent region reviews only veto unsafe
    suppression. They never become canonical instances or positive claims.
    """

    frames = {int(item["frame_idx"]) for item in candidates}
    scenes = {item.get("scene_segment_id") for item in candidates if item.get("scene_segment_id") is not None}
    if len(frames) > 1 or len(scenes) > 1:
        raise ValueError("endpoint matching requires one frame and one scene")
    if frame_idx is not None and frames and frames != {int(frame_idx)}:
        raise ValueError("candidate frame does not match frame_idx")
    actual_frame = next(iter(frames), frame_idx)
    actual_scene = next(iter(scenes), scene_segment_id)
    if scene_segment_id is not None and scenes and scenes != {scene_segment_id}:
        raise ValueError("candidate scene does not match scene_segment_id")
    pool = []
    for index, source in enumerate(sorted(detections, key=lambda value: (str(value.get("detection_id", "")), str(value.get("box"))))):
        if source.get("frame_idx") is not None and int(source["frame_idx"]) != actual_frame:
            continue
        if source.get("scene_segment_id") is not None and actual_scene is not None and source["scene_segment_id"] != actual_scene:
            continue
        pool.append({**source, "detection_id": str(source.get("detection_id") or f"face:{index}")})
    additional = []
    known_ids = {item["detection_id"] for item in pool}
    for index, source in enumerate(sorted(additional_detections, key=lambda value: (str(value.get("detection_id", "")), str(value.get("box"))))):
        if source.get("frame_idx") is not None and int(source["frame_idx"]) != actual_frame:
            continue
        if source.get("scene_segment_id") is not None and actual_scene is not None and source["scene_segment_id"] != actual_scene:
            continue
        identifier = str(source.get("detection_id") or f"additional:{index}")
        if identifier in known_ids:
            continue
        known_ids.add(identifier)
        additional.append({**source, "detection_id": identifier})
    rows: list[dict[str, Any]] = []
    supported: dict[str, tuple[Mapping[str, Any], dict[str, Any], list[float], float]] = {}
    for item in sorted(candidates, key=_candidate_id):
        identifier = _candidate_id(item)
        row = {"candidate_id": identifier, "track_id": str(item["track_id"]), "frame_idx": int(item["frame_idx"]),
               "status": "ambiguous", "reason": "missing_candidate_localization", "detection_id": None, "owner_track_id": None}
        rows.append(row)
        if not evidence_complete:
            row["reason"] = "incomplete_detection_evidence"
            continue
        local_box = _box(item.get("local_box"))
        confidence = _confidence(item.get("local_confidence"))
        if local_box is None and item.get("source") == "detector":
            local_box = _box(item.get("detector_box"))
            confidence = _confidence(item.get("confidence"))
        if local_box is None:
            continue
        localization = {"box": local_box, "confidence": confidence}
        local = select_local_face(item.get("box", []), [localization],
                                  settings if candidate_match_settings is None else candidate_match_settings)
        if local["status"] != "supported":
            row.update(status=local["status"], reason="localization_does_not_support_candidate")
            continue
        # Full-scan and cropped detections may differ in extent. Their common
        # instance is selected geometrically; the stronger equivalence gate
        # below compares the two independent cropped localizations directly.
        matched = select_local_face(local_box, pool, settings)
        row.update(status=matched["status"], reason=matched["reason"])
        if matched["status"] == "supported":
            row["detection_id"] = matched["detection"]["detection_id"]
            supported[identifier] = (item, matched["detection"], local_box, confidence)
    owners: dict[str, list[tuple[Mapping[str, Any], dict[str, Any], list[float], float]]] = {}
    for item, detection, local_box, confidence in supported.values():
        if _endpoint(item) or str(item["track_id"]) not in valid_owner_track_ids:
            continue
        # Canonical full-scan geometry identifies the instance; it is not an
        # extra coarse bounding-box coverage requirement. The actual endpoint
        # localization is checked against the owner below and is protected by
        # the caller's final coverage references after suppression.
        owners.setdefault(detection["detection_id"], []).append((item, detection, local_box, confidence))

    def covered_by_core(value: list[float]) -> bool:
        return any(
            not _endpoint(core) and str(core["track_id"]) in valid_owner_track_ids
            and _box(core.get("box")) is not None
            and covers_reference(value, core["box"], min_coverage=float(settings["equivalence_iou"]),
                                 max_candidate_area_ratio=float(settings["match_max_area_ratio"]))
            for core in candidates
        )

    for row in rows:
        candidate = supported.get(row["candidate_id"])
        if candidate is None or not _endpoint(candidate[0]):
            continue
        item, detection, local_box, _confidence_value = candidate
        eligible = [owner for owner in owners.get(detection["detection_id"], [])
                    if str(owner[0]["track_id"]) != str(item["track_id"])
                    and iou(local_box, owner[2]) >= float(settings["equivalence_iou"])
                    and covers_reference(local_box, owner[0]["box"],
                                         min_coverage=float(settings["equivalence_iou"]),
                                         max_candidate_area_ratio=float(settings["match_max_area_ratio"]))]
        if not eligible:
            row["reason"] = "no_valid_core_owner"
            continue
        uncovered_other_face = False
        for other in pool:
            value = _box(other.get("box"))
            if (other["detection_id"] == detection["detection_id"] or value is None
                    or not math.isfinite(_confidence(other.get("confidence")))):
                continue
            # A detector-returned possible second face is a reason to retain
            # coverage even when it is too weak to support a positive claim.
            # Do not reuse the stronger claim-confidence floor as absence.
            if intersection(value, item["box"]) / area(value) < float(settings["match_min_iou"]):
                continue
            if not covered_by_core(value):
                uncovered_other_face = True
                break
        if uncovered_other_face:
            row.update(status="ambiguous", reason="another_detected_face_has_no_core_coverage")
            continue
        for other in additional:
            value = _box(other.get("box"))
            if value is None or not math.isfinite(_confidence(other.get("confidence"))):
                continue
            if intersection(value, item["box"]) / area(value) < float(settings["match_min_iou"]):
                continue
            correspondence = select_local_face(value, pool, settings)
            # A unique, loose canonical match is not proof that an extra
            # visible face is merely another crop of the same instance. A
            # nearby ghost can also meet that association gate. Only ignore
            # this independent observation when actual core geometry covers
            # it; otherwise it can veto, never support a positive claim.
            if covered_by_core(value):
                continue
            row.update(
                status="ambiguous", reason="another_detected_face_has_no_core_coverage",
                additional_detection_id=other["detection_id"],
                additional_canonical_status=correspondence["status"],
                additional_canonical_detection_id=(correspondence["detection"]["detection_id"]
                                                   if correspondence["detection"] is not None else None),
            )
            uncovered_other_face = True
            break
        if uncovered_other_face:
            continue
        eligible.sort(key=lambda owner: (0 if owner[0].get("source") == "detector" else 1,
                                         -owner[3], str(owner[0]["track_id"]), _candidate_id(owner[0])))
        row.update(status="duplicate", reason="same_face_instance_as_valid_core_owner",
                   owner_track_id=str(eligible[0][0]["track_id"]), owner_candidate_id=_candidate_id(eligible[0][0]))
    return rows


__all__ = ["find_conflict_groups", "localization_match_settings", "match_endpoint_candidates", "plan_rechecks", "region_for_box", "select_local_face"]
