"""Bounded localization evidence for duplicate endpoint observations.

This collector never changes tracking or admission. Only an already published
core observation can replace an endpoint, after final admission and aliases.
Images stay in the existing decoded-frame store; this cache holds geometry.
"""

from __future__ import annotations

import math
import time
from collections import Counter, OrderedDict
from typing import Any, Callable

import numpy as np

from .endpoint_matching import (
    find_conflict_groups,
    localization_match_settings,
    match_endpoint_candidates,
    plan_rechecks,
    region_for_box,
    select_local_face,
)
from .geometry import area, covers_reference, intersection


class EndpointConflictReview:
    def __init__(self, config: dict[str, Any], fps: float, frame_count: int):
        self.settings = config.get("tracking", {}).get("endpoint_conflicts", {})
        self.enabled = bool(self.settings.get("enabled", False)) and (
            config.get("recognition", {}).get("mode", "all") == "all"
        )
        self.revalidation = config.get("revalidation", {})
        # Freeze the review views for this video. The existing local tracking
        # reviewer retains its own angles and shares only the model session.
        self.angles = tuple(self.settings.get("angles", self.revalidation.get("angles", [0])))
        self.candidate_match_settings = localization_match_settings(self.revalidation, self.settings)
        self.audit = config.get("output", {}).get("artifacts_level") in {"audit", "debug"}
        self.max_region_bytes = int(
            config.get("tracking", {}).get("kalman_optical_flow", {})
            .get("bidirectional_fusion", {}).get("max_materialized_bytes", 268435456)
        )
        self.call_limit = min(
            int(self.settings.get("max_calls_total", 0)),
            math.ceil(
                frame_count / max(fps, 1e-9)
                * float(self.settings.get("max_calls_per_video_second", 0.0))
            ),
        )
        self.calls_by_frame: Counter[int] = Counter()
        self.history: dict[tuple[Any, ...], int] = {}
        self.cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self.scans: dict[int, list[dict[str, Any]]] = {}
        self.local_detections: dict[int, dict[str, dict[str, Any]]] = {}
        self.core: dict[int, dict[str, dict[str, Any]]] = {}
        self.evidence: dict[tuple[str, int], dict[str, Any]] = {}
        self.bundles: dict[tuple[str, int, int], dict[str, Any]] = {}
        self.stats: Counter[str] = Counter()
        self.decisions: list[dict[str, Any]] = []
        self.coverage_references: dict[tuple[str, int, int], dict[str, Any]] = {}
        self.additional_coverage_references: dict[tuple[str, int, int], list[dict[str, Any]]] = {}

    def register_scan(self, frame_idx: int, detections: list[dict[str, Any]]) -> None:
        if self.enabled:
            # Capture before association can refine or otherwise mutate box.
            self.scans[frame_idx] = [
                {
                    "box": list(item.get("detector_box", item["box"])),
                    "confidence": float(item.get("confidence", 0.0)),
                    "detection_id": str(item["detection_id"]),
                    "frame_idx": frame_idx,
                }
                for item in detections
            ]

    def prune_before(self, frame_idx: int) -> None:
        for mapping in (self.scans, self.core):
            for key in [key for key in mapping if key < frame_idx]:
                del mapping[key]
        for key in [key for key in self.evidence if key[1] < frame_idx]:
            del self.evidence[key]

    def _prepared(self, item: dict[str, Any]) -> dict[str, Any]:
        value = dict(item)
        key = (str(item["track_id"]), int(item["frame_idx"]))
        review = self.evidence.get(key, {})
        if item.get("endpoint_repair"):
            review = item.get("_review_measurement", {})
        for field in ("local_box", "local_confidence", "local_landmarks"):
            if review.get(field) is not None:
                value[field] = review[field]
        return value

    def _detect(
        self,
        frame_idx: int,
        roi: tuple[int, int, int, int],
        *,
        image: np.ndarray,
        origin: tuple[int, int],
        decode: Callable[..., dict[int, np.ndarray]],
        reviewer: Any,
        allow_inference: bool = True,
    ) -> dict[str, Any] | None:
        key = (frame_idx, roi)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            self.stats["cache_hits"] += 1
            return cached
        if not allow_inference:
            return None
        calls = len(self.angles)
        remaining = min(
            self.call_limit - self.stats["detector_calls"],
            int(self.settings["max_calls_per_frame"]) - self.calls_by_frame[frame_idx],
        )
        if calls > remaining:
            self.stats["budget_skips"] += 1
            return None
        x1, y1, x2, y2 = roi
        if (x2 - x1) * (y2 - y1) * 3 > self.max_region_bytes:
            self.stats["oversized_regions"] += 1
            return None
        ox, oy = origin
        if (
            x1 >= ox and y1 >= oy
            and x2 <= ox + image.shape[1] and y2 <= oy + image.shape[0]
        ):
            crop = image[y1 - oy:y2 - oy, x1 - ox:x2 - ox]
        else:
            started = time.perf_counter()
            try:
                frames = decode(frame_idx, frame_idx, crop=roi)
            except KeyError:
                self.stats["unavailable_frames"] += 1
                return None
            finally:
                self.stats["decode_seconds"] += time.perf_counter() - started
            crop = frames.get(frame_idx)
            if crop is None:
                self.stats["unavailable_frames"] += 1
                return None
        started = time.perf_counter()
        result = reviewer.detect_region(
            crop, origin=(x1, y1), max_calls=remaining, angles=self.angles,
        )
        self.stats["detection_seconds"] += time.perf_counter() - started
        used = int(result["detector_calls"])
        self.stats["detector_calls"] += used
        self.calls_by_frame[frame_idx] += used
        self.stats["region_rechecks"] += 1
        if not result.get("complete", False):
            self.stats["incomplete_rechecks"] += 1
            return None
        for index, detection in enumerate(result["detections"]):
            detection["frame_idx"] = frame_idx
            detection["detection_id"] = f"r{frame_idx}:{x1}:{y1}:{x2}:{y2}:{index}"
            self.local_detections.setdefault(frame_idx, {})[detection["detection_id"]] = detection
        capacity = int(self.settings["cache_entries"])
        if capacity > 0:
            self.cache[key] = result
            while len(self.cache) > capacity:
                self.cache.popitem(last=False)
            self.stats["peak_cache_entries"] = max(self.stats["peak_cache_entries"], len(self.cache))
        return result

    def collect(
        self,
        candidate: dict[str, Any],
        *,
        image: np.ndarray,
        origin: tuple[int, int],
        decode: Callable[..., dict[int, np.ndarray]],
        reviewer: Any,
    ) -> None:
        if not self.enabled:
            return
        started = time.perf_counter()
        frame_idx = int(candidate["frame_idx"])
        track_id = str(candidate["track_id"])
        peers = [
            self._prepared(item)
            for peer_id, item in self.core.get(frame_idx, {}).items()
            if peer_id != track_id
        ]
        if not peers:
            self.stats["geometry_seconds"] += time.perf_counter() - started
            return
        own = self._prepared(candidate)
        groups = find_conflict_groups([own, *peers], self.settings)
        self.stats["geometry_seconds"] += time.perf_counter() - started
        for group in groups:
            if track_id not in group["track_ids"]:
                continue
            self.stats["conflict_groups"] += 1
            values = group["candidates"]
            detections = self.scans.get(frame_idx)
            if detections:
                self.stats["full_scan_reuses"] += 1
                existing = match_endpoint_candidates(
                    values, detections,
                    valid_owner_track_ids={str(value["track_id"]) for value in values if not value.get("endpoint_repair")},
                    settings=self.settings, frame_idx=frame_idx,
                    candidate_match_settings=self.candidate_match_settings,
                    additional_detections=list(self.local_detections.get(frame_idx, {}).values()),
                )
                if any(row["status"] == "duplicate" and row["track_id"] == track_id for row in existing):
                    self.bundles[(track_id, frame_idx, int(candidate["direction"]))] = {
                        "candidates": values, "detections": detections,
                        "additional_detections": self.local_detections.setdefault(frame_idx, {}),
                    }
                    self.stats["existing_evidence_reuses"] += 1
                    continue
            remaining = min(
                self.call_limit - self.stats["detector_calls"],
                int(self.settings["max_calls_per_frame"]) - self.calls_by_frame[frame_idx],
            )
            requests = plan_rechecks(
                [group], frame_idx=frame_idx, history=self.history,
                remaining_calls=remaining, settings=self.settings,
                calls_per_recheck=len(self.angles),
            )
            if requests or self.cache:
                before = self.stats["detector_calls"]
                # A joint crop establishes all nearby instances; independent
                # target crops supply each track's positive localization.
                if not detections:
                    result = self._detect(
                        frame_idx, region_for_box(group["box"], float(self.revalidation["crop_expansion"])),
                        image=image, origin=origin, decode=decode, reviewer=reviewer,
                        allow_inference=bool(requests),
                    )
                    if result is not None:
                        detections = result["detections"]
                if detections:
                    for value in values:
                        if value.get("local_box") is not None:
                            continue
                        result = self._detect(
                            frame_idx, region_for_box(value["box"], float(self.revalidation["crop_expansion"])),
                            image=image, origin=origin, decode=decode, reviewer=reviewer,
                            allow_inference=bool(requests),
                        )
                        if result is None:
                            continue
                        selected = select_local_face(value["box"], result["detections"], self.candidate_match_settings)
                        if self.audit:
                            value["localization_selection"] = {
                                key: selected.get(key) for key in ("status", "reason", "score", "margin")
                            }
                            value["localization_candidates"] = result["detections"][:16]
                        face = selected.get("detection")
                        if face is not None and selected["status"] == "supported":
                            value["local_box"] = list(face["box"])
                            value["local_confidence"] = float(face["confidence"])
                            if face.get("landmarks") is not None:
                                value["local_landmarks"] = face["landmarks"]
                if requests and self.stats["detector_calls"] > before:
                    self.history[requests[0]["history_key"]] = frame_idx
            else:
                self.stats["budget_or_cadence_skips"] += 1
            if detections:
                self.bundles[(track_id, frame_idx, int(candidate["direction"]))] = {
                    "candidates": values,
                    "detections": detections,
                    # Keep a reference to every localization actually seen
                    # on this frame, including later overlapping ROI queries.
                    # A weak extra face can veto deletion without being strong
                    # enough to support a positive ownership claim itself.
                    "additional_detections": self.local_detections.setdefault(frame_idx, {}),
                }

    def resolve(
        self,
        observations: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        aliases: dict[str, str],
        endpoint_candidates: list[dict[str, Any]] | None = None,
    ) -> set[tuple[str, int, int]]:
        """Return duplicate endpoints, never cancel tracking or core coverage."""
        suppressed: set[tuple[str, int, int]] = set()
        if not self.enabled:
            return suppressed
        started = time.perf_counter()
        core = {(str(item["track_id"]), int(item["frame_idx"])): item for item in observations}
        accepted_evidence = {(str(item["track_id"]), int(item["frame_idx"])): item for item in evidence}
        final_endpoints = None if endpoint_candidates is None else {
            (str(item["track_id"]), int(item["frame_idx"]), int(item["direction"])): item
            for item in endpoint_candidates
        }
        for (original_id, frame_idx, direction), bundle in sorted(self.bundles.items()):
            endpoint_id = aliases.get(original_id, original_id)
            final_endpoint = None if final_endpoints is None else final_endpoints.get((endpoint_id, frame_idx, direction))
            if final_endpoints is not None and final_endpoint is None:
                continue
            values = []
            owners = set()
            for item in bundle["candidates"]:
                value = dict(item)
                original_peer_id = str(item["track_id"])
                peer_id = aliases.get(original_peer_id, original_peer_id)
                value["track_id"] = peer_id
                if value.get("endpoint_repair") and final_endpoint is not None:
                    value["box"] = list(final_endpoint.get("motion_box", final_endpoint["box"]))
                    latest_review = final_endpoint.get("_review_measurement", {})
                    if latest_review.get("local_box") is not None:
                        value["local_box"] = latest_review["local_box"]
                        value["local_confidence"] = latest_review.get("local_confidence")
                        value["local_landmarks"] = latest_review.get("local_landmarks")
                if not value.get("endpoint_repair"):
                    current = core.get((peer_id, frame_idx))
                    if current is None or peer_id == endpoint_id:
                        continue
                    # Refresh the render geometry after admission. Retain the
                    # independently measured localization from collection.
                    value["box"] = list(current["box"])
                    owners.add(peer_id)
                    final_review = accepted_evidence.get((peer_id, frame_idx), {})
                    if value.get("local_box") is None and final_review.get("local_box") is not None:
                        value["local_box"] = final_review["local_box"]
                        value["local_confidence"] = final_review.get("local_confidence")
                        value["local_landmarks"] = final_review.get("local_landmarks")
                values.append(value)
            if not owners:
                continue
            decisions = match_endpoint_candidates(
                values, bundle["detections"], valid_owner_track_ids=owners,
                settings=self.settings, frame_idx=frame_idx,
                candidate_match_settings=self.candidate_match_settings,
                additional_detections=list(bundle.get("additional_detections", {}).values()),
            )
            for decision in decisions:
                if str(decision.get("track_id")) != endpoint_id:
                    continue
                self.stats[str(decision["status"])] += 1
                self.stats["reason_" + str(decision["reason"])] += 1
                if self.audit:
                    self.decisions.append({
                        **decision, "frame_idx": frame_idx, "direction": direction,
                        "localizations": [
                            {key: value.get(key) for key in (
                                "track_id", "box", "local_box", "local_confidence", "local_landmarks",
                                "localization_selection", "localization_candidates",
                            )}
                            for value in values
                        ],
                        "detections": bundle["detections"],
                        "additional_detections": list(bundle.get("additional_detections", {}).values()),
                    })
                if decision["status"] == "duplicate":
                    key = (endpoint_id, frame_idx, direction)
                    suppressed.add(key)
                    localized = next(value for value in values if value.get("endpoint_repair"))
                    self.coverage_references[key] = {
                        "track_id": str(decision["owner_track_id"]),
                        "frame_idx": frame_idx,
                        "box": list(localized["local_box"]),
                        "source": "local_scrfd",
                        "admission_scope": "endpoint_conflict_coverage",
                        "replaced_endpoint_track_id": endpoint_id,
                    }
                    # A second instance may allow suppression because another
                    # accepted core already covers it. Preserve that coverage
                    # through smoothing too, not just the primary face claim.
                    # The matched coarse canonical box is only an instance ID
                    # and is deliberately not reintroduced as a reference.
                    extra_references = []
                    seen = {(str(decision["owner_track_id"]), tuple(localized["local_box"]))}
                    known_faces = [*bundle["detections"], *bundle.get("additional_detections", {}).values()]
                    core_values = sorted(
                        (value for value in values if not value.get("endpoint_repair") and str(value["track_id"]) in owners),
                        key=lambda value: (str(value["track_id"]) != str(decision["owner_track_id"]), str(value["track_id"])),
                    )
                    for face in known_faces:
                        if face.get("frame_idx") is not None and int(face["frame_idx"]) != frame_idx:
                            continue
                        if face.get("scene_segment_id") is not None and localized.get("scene_segment_id") is not None and face["scene_segment_id"] != localized["scene_segment_id"]:
                            continue
                        if face.get("detection_id") == decision["detection_id"]:
                            continue
                        try:
                            face_box = np.asarray(face.get("box"), dtype=np.float64).reshape(4).tolist()
                            valid = math.isfinite(float(face.get("confidence"))) and np.isfinite(face_box).all() and area(face_box) > 0.0
                        except (TypeError, ValueError):
                            continue
                        if not valid:
                            continue
                        if intersection(face_box, localized["box"]) / max(area(face_box), 1e-9) < float(self.settings["match_min_iou"]):
                            continue
                        for owner in core_values:
                            owner_id = str(owner["track_id"])
                            if not covers_reference(
                                face_box, owner["box"], min_coverage=float(self.settings["equivalence_iou"]),
                                max_candidate_area_ratio=float(self.settings["match_max_area_ratio"]),
                            ):
                                continue
                            reference_key = (owner_id, tuple(face_box))
                            if reference_key not in seen:
                                seen.add(reference_key)
                                extra_references.append({
                                    "track_id": owner_id, "frame_idx": frame_idx, "box": list(face_box),
                                    "source": "local_scrfd", "admission_scope": "endpoint_conflict_coverage",
                                    "replaced_endpoint_track_id": endpoint_id,
                                })
                            break
                    self.additional_coverage_references[key] = extra_references
        self.stats["resolve_seconds"] += time.perf_counter() - started
        return suppressed

    def summary(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "angles": list(self.angles),
                "detector_call_limit": self.call_limit, **dict(self.stats)}
