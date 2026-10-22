"""One demux pass: detection, online association, ROI flow, and RGB review."""

from __future__ import annotations

import logging
import math
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .bidirectional_fusion import (
    interpolate_published_geometry,
    soft_fuse_bidirectional_sequence,
)
from .box_stabilization import stabilize_observations
from .endpoint_conflicts import EndpointConflictReview
from .geometry import (
    area_ratio,
    clip,
    containment,
    covers_reference,
    iou,
    normalized_center_distance,
)
from .model_catalog import DETECTION_TASK, RECOGNITION_TASK, VERIFICATION_TASK
from .models import (
    detector_max_detections,
    make_face_analysis,
    make_review_face_analysis,
)
from .packet_cache import DecodedFrameStore, EncodedPacketCache, iter_cached_frames
from .recognition import (
    GALLERY_DETECTOR_CONFIDENCE_THRESHOLD,
    GALLERY_DETECTOR_INPUT_SIZES,
    LOCAL_LANDMARK_CONFIDENCE_THRESHOLD,
    LOCAL_LANDMARK_MAX_AREA_RATIO,
    LOCAL_LANDMARK_MAX_CENTER_DISTANCE,
    LOCAL_LANDMARK_MIN_CONTAINMENT,
    LOCAL_LANDMARK_MIN_IOU,
    SINGLE_FRAME_SIMILARITY_OFFSET,
    TEMPORAL_EVIDENCE_MIN_ADJACENT_GAP_SECONDS,
    TEMPORAL_EVIDENCE_MIN_SELECTED_FRAMES,
    TEMPORAL_EVIDENCE_SIMILARITY_OFFSET,
    RecognitionCandidate,
    arcface_align_112,
    create_recognition_engine,
    detect_gallery_faces_upright,
    local_landmark_box_agreement,
    recognition_candidate_quality,
    select_temporally_distributed,
)
from .revalidation import LocalReviewer, finalize_precomputed
from .roi_flow import AffineEndpointState, ROIFlowState
from .scan import ScanRunner
from .scene_cut import SceneCutDetector
from .tracker import _select_track_frame_candidates
from .video import probe_video

logger = logging.getLogger(__name__)

# Identity crops are biometric data and used to live until end-of-video without
# a process-wide ceiling.  Keep this internal rather than adding another user
# tuning knob: exceeding the ceiling makes the affected track UNKNOWN, which is
# privacy-safe for both selective policies.
_RECOGNITION_CANDIDATE_MAX_BYTES = 64 * 1024 * 1024
_RECOGNITION_CANDIDATE_PRUNE_RATIO = 0.85
_ENDPOINT_VERIFIER_HZ = 4.0
_ENDPOINT_LOCAL_REVIEW_HZ = 1.0

# ``max_analysis_fps`` is deliberately a soft ceiling. Real-world CFR files
# commonly report values such as 30.02 for nominal 30 FPS media; treating that
# tiny container-level difference as a reason to halve the detector cadence is
# both surprising and wasteful. Keep the tolerance internal so the public
# configuration has one sampling control.
ANALYSIS_FPS_TOLERANCE_MULTIPLIER = 1.05
_FINAL_CROSS_TRACK_MIN_COVERAGE = 0.80
_FINAL_CROSS_TRACK_MAX_AREA_RATIO = 2.50


def _raise_if_cancelled(is_cancelled: Callable[[], bool] | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise InterruptedError("PrivateFrame operation was cancelled")


def resolve_analysis_stride(
    source_fps: float,
    max_analysis_fps: float,
    *,
    tolerance_multiplier: float = ANALYSIS_FPS_TOLERANCE_MULTIPLIER,
) -> int:
    """Return the smallest uniform integer stride under the soft FPS ceiling."""

    source = float(source_fps)
    maximum = float(max_analysis_fps)
    tolerance = float(tolerance_multiplier)
    if not math.isfinite(source) or source <= 0.0:
        raise ValueError("source_fps must be positive and finite")
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_analysis_fps must be positive and finite")
    if not math.isfinite(tolerance) or tolerance < 1.0:
        raise ValueError("tolerance_multiplier must be finite and at least 1")
    return max(1, math.ceil(source / (maximum * tolerance)))


@dataclass
class ObjectState:
    track: dict[str, Any]
    flow: ROIFlowState
    last_detection_frame: int
    last_detection_box: np.ndarray
    # Rank among frames on which the full-frame detector actually ran. This is
    # deliberately independent of ``frame_idx``: regular stride, adaptive
    # bursts, scene cuts, and the forced EOF scan all contribute real scan
    # opportunities, while sampled-out video frames do not.
    last_detection_scan_rank: int | None = None
    last_consensus_anchor_box: np.ndarray | None = None
    last_detection_geometry_source: str = ""
    pending: dict[int, dict[str, Any]] = field(default_factory=dict)
    active: bool = True
    pre_roll_extension: int = 0


@dataclass(frozen=True)
class ReviewPlan:
    """Neural review work scheduled for one provisional track observation."""

    run_local_scrfd: bool
    run_verifier: bool
    local_review_reason: str


@dataclass(frozen=True)
class RecognitionLandmarkProposal:
    """One internally consistent box/landmark source for video recognition."""

    source: str
    box: Any
    landmarks: Any
    confidence: Any
    scan_angle_degrees: int
    require_reference_agreement: bool


def _evidence_boxes_by_track_frame(
    evidence: list[dict[str, Any]] | None,
) -> dict[tuple[str, int], list[list[float]]]:
    references: dict[tuple[str, int], list[list[float]]] = {}
    for item in evidence or []:
        value = item.get("box")
        if value is None:
            continue
        key = (str(item["track_id"]), int(item["frame_idx"]))
        box = [float(component) for component in value]
        if box not in references.setdefault(key, []):
            references[key].append(box)
    return references


def _validate_final_track_geometry(
    observations: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep one evidence-covering final box per track and frame.

    Stabilization may move a box away from the reviewed geometry. Prefer the
    stabilized value, fall back to its raw output box when that safely covers
    all same-key evidence, and omit an unsafe observation so the final
    coverage invariant can report the missing track/frame. Historical records
    without same-key evidence retain their prior behavior.
    """

    def rank(value: dict[str, Any]) -> tuple[bool, bool, float]:
        return (
            value.get("local_confidence") is not None,
            value.get("source") == "detector",
            float(value.get("local_confidence") or value.get("confidence") or 0.0),
        )

    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for item in observations:
        key = (str(item["track_id"]), int(item["frame_idx"]))
        prior = selected.get(key)
        if prior is None or rank(item) > rank(prior):
            selected[key] = item

    evidence_boxes = _evidence_boxes_by_track_frame(evidence)
    output: list[dict[str, Any]] = []
    for key, item in selected.items():
        references = evidence_boxes.get(key, [])
        if not references or all(
            covers_reference(
                reference,
                item["box"],
                min_coverage=_FINAL_CROSS_TRACK_MIN_COVERAGE,
                max_candidate_area_ratio=_FINAL_CROSS_TRACK_MAX_AREA_RATIO,
            )
            for reference in references
        ):
            output.append(item)
            continue
        raw_box = item.get("raw_output_box")
        if raw_box is None or not all(
            covers_reference(
                reference,
                raw_box,
                min_coverage=_FINAL_CROSS_TRACK_MIN_COVERAGE,
                max_candidate_area_ratio=_FINAL_CROSS_TRACK_MAX_AREA_RATIO,
            )
            for reference in references
        ):
            continue
        fallback = dict(item)
        fallback["box"] = list(raw_box)
        fallback["motion_box"] = list(raw_box)
        fallback["box_stabilization"] = "raw_evidence_fallback"
        output.append(fallback)
    return sorted(
        output,
        key=lambda item: (int(item["frame_idx"]), str(item["track_id"])),
    )


def _deduplicate_final_cross_tracks(
    observations: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    *,
    recognition_mode: str,
) -> list[dict[str, Any]]:
    """Suppress duplicate tracks only after their final boxes are stable.

    Selective recognition decisions are track-specific and therefore never
    share geometry across tracks. In ``all`` mode, a kept box may replace a
    different track only when the two final boxes mutually cover one another
    and the kept box also safely covers every evidence reference associated
    with the dropped track/frame.
    """

    ordered = sorted(
        _validate_final_track_geometry(observations, evidence),
        key=lambda item: (
            int(item["frame_idx"]),
            0 if item.get("source") == "detector" else 1,
            -float(item.get("confidence") or 0.0),
            str(item["track_id"]),
        ),
    )
    if str(recognition_mode) != "all":
        return sorted(
            ordered,
            key=lambda item: (int(item["frame_idx"]), str(item["track_id"])),
        )

    evidence_boxes = _evidence_boxes_by_track_frame(evidence)
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for item in ordered:
        by_frame.setdefault(int(item["frame_idx"]), []).append(item)

    output: list[dict[str, Any]] = []
    for frame_idx in sorted(by_frame):
        kept: list[dict[str, Any]] = []
        for item in by_frame[frame_idx]:
            item_box = item["box"]
            references = evidence_boxes.get((str(item["track_id"]), frame_idx), [])
            duplicate = any(
                str(prior["track_id"]) != str(item["track_id"])
                and covers_reference(
                    item_box,
                    prior["box"],
                    min_coverage=_FINAL_CROSS_TRACK_MIN_COVERAGE,
                    max_candidate_area_ratio=_FINAL_CROSS_TRACK_MAX_AREA_RATIO,
                )
                and covers_reference(
                    prior["box"],
                    item_box,
                    min_coverage=_FINAL_CROSS_TRACK_MIN_COVERAGE,
                    max_candidate_area_ratio=_FINAL_CROSS_TRACK_MAX_AREA_RATIO,
                )
                and bool(references)
                and all(
                    covers_reference(
                        reference,
                        prior["box"],
                        min_coverage=_FINAL_CROSS_TRACK_MIN_COVERAGE,
                        max_candidate_area_ratio=_FINAL_CROSS_TRACK_MAX_AREA_RATIO,
                    )
                    for reference in references
                )
                for prior in kept
            )
            if not duplicate:
                kept.append(item)
        output.extend(kept)
    return sorted(
        output,
        key=lambda item: (int(item["frame_idx"]), str(item["track_id"])),
    )


def _accepted_interval_coverage(
    tracks: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    evidence: list[dict[str, Any]] | None = None,
    *,
    allow_cross_track_coverage: bool = False,
) -> tuple[int, int, int]:
    """Return expected, uncovered, and deduplicated-but-covered interval frames."""

    expected_by_track: dict[str, set[int]] = {}
    for track in tracks:
        if not bool(track.get("accepted", False)):
            continue
        expected = expected_by_track.setdefault(str(track["track_id"]), set())
        for first, last in track.get("accepted_intervals", []):
            expected.update(range(int(first), int(last) + 1))
    observations_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    observations_by_frame: dict[int, list[dict[str, Any]]] = {}
    for item in observations:
        frame_idx = int(item["frame_idx"])
        observations_by_key.setdefault(
            (str(item["track_id"]), frame_idx),
            [],
        ).append(item)
        observations_by_frame.setdefault(frame_idx, []).append(item)
    evidence_boxes = _evidence_boxes_by_track_frame(evidence)

    def covers_evidence(candidate: dict[str, Any], references: list[list[float]]) -> bool:
        return all(
            covers_reference(
                reference,
                candidate["box"],
                min_coverage=_FINAL_CROSS_TRACK_MIN_COVERAGE,
                max_candidate_area_ratio=_FINAL_CROSS_TRACK_MAX_AREA_RATIO,
            )
            for reference in references
        )

    def geometrically_covered(track_id: str, frame_idx: int) -> tuple[bool, bool]:
        references = evidence_boxes.get((track_id, frame_idx), [])
        exact = observations_by_key.get((track_id, frame_idx), [])
        if exact and (not references or any(covers_evidence(item, references) for item in exact)):
            return True, False
        if not allow_cross_track_coverage or not references:
            return False, False
        cross_track = [
            item
            for item in observations_by_frame.get(frame_idx, [])
            if str(item["track_id"]) != track_id
        ]
        covered = any(covers_evidence(item, references) for item in cross_track)
        return covered, covered

    expected_frames = sum(len(values) for values in expected_by_track.values())
    hole_frames = 0
    cross_track_coverage_frames = 0
    for track_id, values in expected_by_track.items():
        for frame_idx in values:
            covered, cross_track = geometrically_covered(track_id, frame_idx)
            if covered and cross_track:
                cross_track_coverage_frames += 1
            elif not covered:
                hole_frames += 1
    return expected_frames, hole_frames, cross_track_coverage_frames


def _detector_pipeline_depth(
    configured_depth: int,
    frame_stride: int,
    frame_bytes: int,
    byte_limit: int,
) -> int:
    """Preserve scan concurrency without unbounded decoded-frame growth."""

    configured_depth = max(1, int(configured_depth))
    frame_stride = max(1, int(frame_stride))
    target = configured_depth * min(frame_stride, 3)
    if frame_stride == 1 or frame_bytes <= 0 or byte_limit <= 0:
        return target
    byte_limited_depth = max(configured_depth, int(byte_limit) // int(frame_bytes))
    return min(target, byte_limited_depth)


def _decoded_frame_store_target(
    config: dict[str, Any],
    fps: float,
) -> int:
    """Return the configured or policy-derived decoded-frame horizon.

    ``None`` keeps the Base profile adaptive: retain enough frame indices for
    the longest configured historical consumer, while the independent byte
    capacity remains the hard memory bound.  An explicit integer preserves a
    deterministic operator override; zero disables decoded-frame retention.
    """

    configured = config["streaming"].get("recent_frame_cache_frames")
    if configured is not None:
        return max(0, int(configured))
    history = config["streaming"]
    tracking = config.get("tracking", {})
    flow = tracking.get("kalman_optical_flow", {})
    fusion = flow.get("bidirectional_fusion", {})
    time_horizon = math.ceil(
        max(0.0, float(history.get("max_retroactive_seconds", 0.0)))
        * max(float(fps), 1.0)
    )
    return max(
        1,
        time_horizon + 1,
        int(fusion.get("max_gap_frames", 0)) + 2,
        int(tracking.get("reliable_pre_roll_extension", 0)) + 1,
        int(tracking.get("reliable_endpoint_extension", 0)) + 1,
    )


def _detector_scan_burst_frames(rule_gate: dict[str, Any]) -> int:
    """Use enough real consecutive scans for both anchor and persistence gates."""

    return max(
        1,
        int(rule_gate.get("strong_anchor_window_frames", 1)),
        int(rule_gate.get("min_detector_frames", 1)),
    )


def _interpolate_tracking_enabled(mode: str, frame_stride: int) -> bool:
    """Use interpolation only when regular detector frames are sampled out."""

    return mode == "interpolate" and frame_stride > 1


def _box_area(value: np.ndarray) -> float:
    size = np.maximum(0.0, value[2:4] - value[0:2])
    return float(size[0] * size[1])


def _touched_frame_edges(
    value: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    epsilon: float,
) -> frozenset[str]:
    edges: set[str] = set()
    if float(value[0]) <= epsilon:
        edges.add("left")
    if float(value[1]) <= epsilon:
        edges.add("top")
    if float(value[2]) >= frame_width - epsilon:
        edges.add("right")
    if float(value[3]) >= frame_height - epsilon:
        edges.add("bottom")
    return frozenset(edges)


def _bidirectional_geometry_bridge_decision(
    *,
    left_consensus: np.ndarray,
    right_consensus: np.ndarray,
    left_publication: np.ndarray,
    right_publication: np.ndarray,
    left_geometry_source: str,
    right_geometry_source: str,
    forward_results: dict[int, dict[str, Any]],
    reverse_results: dict[int, dict[str, Any]],
    frame_width: int,
    frame_height: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Gate endpoint-size repair without turning geometry into face truth."""

    bridge = settings["geometry_bridge"]
    if not bool(bridge["enabled"]):
        return {"accepted": False, "reason": "disabled"}
    epsilon = float(bridge["edge_epsilon_pixels"])
    minimum_ratio = float(bridge["min_edge_expansion_ratio"])

    def endpoint(
        consensus: np.ndarray,
        publication: np.ndarray,
        source: str,
    ) -> dict[str, Any]:
        raw_area = _box_area(consensus)
        publication_area = _box_area(publication)
        ratio = publication_area / raw_area if raw_area > 0.0 else math.inf
        edges = _touched_frame_edges(
            consensus,
            frame_width=frame_width,
            frame_height=frame_height,
            epsilon=epsilon,
        )
        locally_trusted = source in {
            "local_scrfd",
            "local_scrfd_filtered",
        }
        edge_expansion = bool(source == "detector_center_motion_size" and edges and ratio >= minimum_ratio)
        return {
            "usable": bool(locally_trusted or edge_expansion),
            "locally_trusted": locally_trusted,
            "edge_expansion": edge_expansion,
            "edges": sorted(edges),
            "publication_to_consensus_area_ratio": float(ratio),
        }

    left = endpoint(
        np.asarray(left_consensus, dtype=np.float64),
        np.asarray(left_publication, dtype=np.float64),
        left_geometry_source,
    )
    right = endpoint(
        np.asarray(right_consensus, dtype=np.float64),
        np.asarray(right_publication, dtype=np.float64),
        right_geometry_source,
    )
    edge_endpoints = [item for item in (left, right) if bool(item["edge_expansion"])]
    if not edge_endpoints:
        return {
            "accepted": False,
            "reason": "no_edge_expansion",
            "left": left,
            "right": right,
        }
    if not bool(left["usable"] and right["usable"]):
        return {
            "accepted": False,
            "reason": "endpoint_geometry_unusable",
            "left": left,
            "right": right,
        }
    common_edges = set(edge_endpoints[0]["edges"])
    for endpoint_evidence in edge_endpoints[1:]:
        common_edges.intersection_update(endpoint_evidence["edges"])
    if not common_edges:
        return {
            "accepted": False,
            "reason": "edge_mismatch",
            "left": left,
            "right": right,
        }

    ordered = sorted(set(forward_results) & set(reverse_results))
    count = len(ordered)
    both_trusted = 0
    mutually_consistent = 0
    for frame_idx in ordered:
        forward = forward_results[frame_idx]
        reverse = reverse_results[frame_idx]
        if bool(forward.get("flow_measurement_valid", False)) and bool(
            reverse.get("flow_measurement_valid", False)
        ):
            both_trusted += 1
        if iou(forward["box"], reverse["box"]) >= float(
            settings["mutual_min_iou"]
        ) or normalized_center_distance(forward["box"], reverse["box"]) <= float(
            settings["mutual_max_center_distance"]
        ):
            mutually_consistent += 1
    trusted_fraction = both_trusted / count if count else 0.0
    consistent_fraction = mutually_consistent / count if count else 0.0
    accepted = bool(
        trusted_fraction >= float(bridge["min_both_trusted_fraction"])
        and consistent_fraction >= float(bridge["min_mutual_consistent_fraction"])
    )
    return {
        "accepted": accepted,
        "reason": "accepted" if accepted else "gap_flow_evidence",
        "left": left,
        "right": right,
        "common_edges": sorted(common_edges),
        "both_trusted_fraction": float(trusted_fraction),
        "mutual_consistent_fraction": float(consistent_fraction),
    }


def _fragment_aliases(
    tracks: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    settings: dict[str, Any] | None,
) -> dict[str, str]:
    """Find split tracks whose local SCRFD measurements repeatedly agree."""

    identifiers = [str(track["track_id"]) for track in tracks]
    track_map = {str(track["track_id"]): track for track in tracks}
    parent = {track_id: track_id for track_id in identifiers}
    if not settings or not bool(settings.get("enabled", False)):
        return parent

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: str, second: str) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    by_track: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for item in evidence:
        if item.get("local_box") is None:
            continue
        confidence = item.get("local_confidence")
        if confidence is None or float(confidence) < float(settings["min_local_confidence"]):
            continue
        by_track.setdefault(str(item["track_id"]), {}).setdefault(int(item["frame_idx"]), []).append(item)

    minimum_overlap = int(settings["min_overlap_frames"])
    minimum_agreement = int(settings["min_agreement_frames"])
    minimum_fraction = float(settings["min_agreement_fraction"])
    minimum_iou = float(settings["min_local_iou"])
    frame_sets = {track_id: frozenset(values) for track_id, values in by_track.items()}
    frame_ranges = {track_id: (min(values), max(values)) for track_id, values in frame_sets.items() if values}
    for first_index, first in enumerate(identifiers):
        first_values = by_track.get(first, {})
        if not first_values:
            continue
        for second in identifiers[first_index + 1 :]:
            second_values = by_track.get(second, {})
            if not second_values:
                continue
            first_scene = track_map[first].get("scene_segment_id")
            second_scene = track_map[second].get("scene_segment_id")
            if first_scene is not None and second_scene is not None and int(first_scene) != int(second_scene):
                # A cut is an absolute identity boundary.  Local boxes from
                # two shots must never make fragment stitching bridge it.
                continue
            first_range = frame_ranges[first]
            second_range = frame_ranges[second]
            if (
                min(first_range[1], second_range[1]) - max(first_range[0], second_range[0]) + 1
                < minimum_overlap
            ):
                continue
            common = sorted(frame_sets[first] & frame_sets[second])
            if len(common) < minimum_overlap:
                continue
            agreements = sum(
                max(
                    iou(first_item["local_box"], second_item["local_box"])
                    for first_item in first_values[frame_idx]
                    for second_item in second_values[frame_idx]
                )
                >= minimum_iou
                for frame_idx in common
            )
            if agreements < minimum_agreement:
                continue
            if agreements / len(common) < minimum_fraction:
                continue
            union(first, second)

    components: dict[str, list[str]] = {}
    for track_id in identifiers:
        components.setdefault(find(track_id), []).append(track_id)
    detector_counts = {str(track["track_id"]): len(track.get("detections", [])) for track in tracks}
    order = {track_id: index for index, track_id in enumerate(identifiers)}
    aliases: dict[str, str] = {}
    for members in components.values():
        canonical = max(
            members,
            key=lambda track_id: (
                detector_counts[track_id],
                -order[track_id],
            ),
        )
        aliases.update({track_id: canonical for track_id in members})
    return aliases


def _apply_fragment_aliases(
    tracks: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    aliases: dict[str, str],
    *,
    resolve_duplicate_candidates: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply track aliases after tracking without changing model evidence."""

    merged_ids = {
        canonical: sorted(track_id for track_id, value in aliases.items() if value == canonical)
        for canonical in set(aliases.values())
    }
    track_map = {str(track["track_id"]): track for track in tracks}
    for canonical, members in merged_ids.items():
        scene_ids = {
            int(track_map[member]["scene_segment_id"])
            for member in members
            if track_map[member].get("scene_segment_id") is not None
        }
        if len(scene_ids) > 1:
            raise ValueError(f"fragment aliases cannot cross a scene cut: {canonical}")
    for values in (detections, candidates, evidence):
        for item in values:
            original = str(item["track_id"])
            canonical = aliases.get(original, original)
            if canonical != original:
                item["original_track_id"] = original
                item["track_id"] = canonical

    merged_tracks: list[dict[str, Any]] = []
    for canonical in sorted(merged_ids):
        members = merged_ids[canonical]
        track = track_map[canonical]
        member_last_frames = {
            member: int(track_map[member]["detections"][-1]["frame_idx"])
            for member in members
            if track_map[member].get("detections")
        }
        by_frame: dict[int, dict[str, Any]] = {}
        for member in members:
            for detection in track_map[member].get("detections", []):
                frame_idx = int(detection["frame_idx"])
                prior = by_frame.get(frame_idx)
                if prior is None or float(detection.get("confidence", 0.0)) > float(
                    prior.get("confidence", 0.0)
                ):
                    by_frame[frame_idx] = detection
        track["detections"] = [by_frame[value] for value in sorted(by_frame)]
        track["stitched_track_ids"] = members
        track["starts_at_scene_cut"] = any(
            bool(track_map[member].get("starts_at_scene_cut", False)) for member in members
        )
        track["ends_at_scene_cut"] = any(
            bool(track_map[member].get("ends_at_scene_cut", False)) for member in members
        )
        start_boundaries = [
            int(track_map[member]["start_scene_cut_frame"])
            for member in members
            if track_map[member].get("start_scene_cut_frame") is not None
        ]
        end_boundaries = [
            int(track_map[member]["end_scene_cut_frame"])
            for member in members
            if track_map[member].get("end_scene_cut_frame") is not None
        ]
        track["start_scene_cut_frame"] = min(start_boundaries) if start_boundaries else None
        track["end_scene_cut_frame"] = max(end_boundaries) if end_boundaries else None
        chronological_last = max(
            members,
            key=lambda member: member_last_frames.get(member, -1),
        )
        track["close_reason"] = track_map[chronological_last].get("close_reason")
        track["close_boundary_frame_exclusive"] = track_map[chronological_last].get(
            "close_boundary_frame_exclusive"
        )
        merged_tracks.append(track)

    # Admission and output metadata need one review record per canonical
    # object/frame. Match the final detector's confidence-first selection so
    # the chosen box and evidence come from the same stitched fragment.
    # Detector ties retain input order like finalization; canonical origin and
    # local confidence remain the tie breakers for tracking-only evidence.
    def evidence_rank(item: dict[str, Any]) -> tuple[bool, float, bool, float]:
        detector = item.get("source") == "detector"
        if detector:
            return True, float(item.get("confidence") or 0.0), False, 0.0
        return (
            False,
            0.0,
            item.get("original_track_id", item["track_id"]) == item["track_id"],
            float(item.get("local_confidence") or 0.0),
        )

    selected_evidence: dict[tuple[str, int], dict[str, Any]] = {}
    for item in evidence:
        key = (str(item["track_id"]), int(item["frame_idx"]))
        prior = selected_evidence.get(key)
        if prior is None or evidence_rank(item) > evidence_rank(prior):
            selected_evidence[key] = item

    # A stitched object can have two independently tracked candidates for the
    # same frame.  Resolve that ambiguity before global NMS and box smoothing;
    # otherwise the smoother sees two geometries with one canonical track id
    # and can interpolate through the rejected fragment.  The independently
    # selected local-review record is the common geometric reference, so keep
    # the candidate that agrees with it best.  Candidate order is otherwise
    # preserved because it is part of the deterministic global-NMS tie break.
    stitched = (
        {canonical for canonical, members in merged_ids.items() if len(members) > 1}
        if resolve_duplicate_candidates
        else set()
    )
    candidate_groups: dict[tuple[str, int], list[int]] = {}
    for index, item in enumerate(candidates):
        key = (str(item["track_id"]), int(item["frame_idx"]))
        if key[0] in stitched:
            candidate_groups.setdefault(key, []).append(index)
    keep = set(range(len(candidates)))
    for key, indices in candidate_groups.items():
        if len(indices) < 2:
            continue
        reference = selected_evidence.get(key, {}).get("box")

        def candidate_rank(
            index: int, reference_box: list[float] | None = reference
        ) -> tuple[float, float, bool, float]:
            item = candidates[index]
            value = item.get("motion_box", item.get("box"))
            if reference_box is not None and value is not None:
                overlap = iou(value, reference_box)
                distance = normalized_center_distance(value, reference_box)
            else:
                overlap, distance = -1.0, math.inf
            return (
                overlap,
                -distance,
                item.get("original_track_id", item["track_id"]) == item["track_id"],
                float(item.get("quality") or 0.0),
            )

        selected_index = max(indices, key=candidate_rank)
        for index in indices:
            if index != selected_index:
                keep.discard(index)
        candidates[selected_index]["fragment_candidate_count"] = len(indices)
        candidates[selected_index]["fragment_geometry_selected"] = True
    candidates[:] = [item for index, item in enumerate(candidates) if index in keep]
    return merged_tracks, list(selected_evidence.values())


def _association_score(
    state: ObjectState,
    detection: dict[str, Any],
    frame_idx: int,
    settings: dict[str, Any],
    *,
    reference_box: np.ndarray | None = None,
    allow_long_gap_flow: bool = True,
) -> float | None:
    frame_gap = frame_idx - state.last_detection_frame
    if not 1 <= frame_gap <= int(settings["max_missed_frames"]):
        return None
    current_scan_rank = detection.get("detector_scan_rank")
    if current_scan_rank is not None and state.last_detection_scan_rank is not None:
        scan_gap = int(current_scan_rank) - int(state.last_detection_scan_rank)
        if scan_gap < 1:
            return None
    else:
        # Direct API/unit callers created before scan ranks existed retain the
        # every-frame behavior. Production detections always carry a rank.
        scan_gap = frame_gap
    fps = max(float(settings.get("source_fps", 30.0)), 1e-6)
    elapsed_seconds = frame_gap / fps
    long_gap = bool(
        scan_gap > int(settings["association_max_scan_gap"])
        or elapsed_seconds > float(settings["association_max_gap_seconds"])
    )
    reference = (
        state.flow.box
        if reference_box is None
        else np.asarray(reference_box, dtype=np.float64)
    )
    target = np.asarray(detection["box"], dtype=np.float64)
    ratio = area_ratio(reference, target)
    if ratio > float(settings["association_max_area_ratio"]):
        return None
    overlap = iou(reference, target)
    distance = normalized_center_distance(reference, target)
    strict_sparse_geometry = bool(
        not allow_long_gap_flow
        and elapsed_seconds
        > float(settings["association_strict_geometry_after_seconds"])
    )
    if strict_sparse_geometry and (
        overlap < float(settings.get("long_gap_min_iou", 0.15))
        and distance > float(settings.get("long_gap_max_center_distance", 0.50))
    ):
        return None
    if long_gap:
        if not allow_long_gap_flow:
            return None
        # Five seconds is a state-retention ceiling, not permission to join two
        # unrelated detections.  A long-gap re-anchor is allowed only while the
        # ROI tracker produced an uninterrupted path to this frame. A short
        # bounded coast is still a continuous state path; spatial gates below
        # remain mandatory before accepting the new detector hit.
        if bool(settings.get("long_gap_requires_continuous_flow", True)):
            allowed_coast = int(settings["kalman_optical_flow"]["max_coast_frames"])
            coast = int(getattr(state.flow, "coast", allowed_coast + 1))
            if len(state.pending) != frame_gap or coast > allowed_coast:
                return None
        if overlap < float(settings.get("long_gap_min_iou", 0.15)) and distance > float(
            settings.get("long_gap_max_center_distance", 0.50)
        ):
            return None
    # Normalize motion relaxation to physical time rather than source FPS. At
    # 30 FPS this preserves the established score exactly; at other rates an
    # equal elapsed duration receives the same geometric allowance.
    normalized_time_gap = max(1.0, elapsed_seconds * 30.0)
    relaxed = float(settings["association_max_center_distance"]) * math.sqrt(
        normalized_time_gap
    )
    limit = (
        min(relaxed, float(settings["association_sparse_max_center_distance"]))
        if normalized_time_gap > 1.0
        else relaxed
    )
    if overlap < float(settings["association_min_iou"]) and distance > limit:
        return None
    score = 2.5 * overlap + max(0.0, 1.0 - distance / max(limit, 1e-6))
    score -= 0.025 * (normalized_time_gap - 1.0) + 0.08 * abs(math.log(ratio))
    return score if score >= float(settings["association_min_score"]) else None


def _is_long_association_gap(
    state: ObjectState,
    detection: dict[str, Any],
    frame_idx: int,
    settings: dict[str, Any],
) -> bool:
    """Return whether an anchor exceeds either scan or elapsed-time budget."""

    frame_gap = int(frame_idx) - int(state.last_detection_frame)
    current_scan_rank = detection.get("detector_scan_rank")
    if current_scan_rank is not None and state.last_detection_scan_rank is not None:
        scan_gap = int(current_scan_rank) - int(state.last_detection_scan_rank)
    else:
        scan_gap = frame_gap
    fps = max(float(settings.get("source_fps", 30.0)), 1e-6)
    return bool(
        scan_gap > int(settings["association_max_scan_gap"])
        or frame_gap / fps > float(settings["association_max_gap_seconds"])
    )


def _association_rescue_decision(
    forward: dict[str, Any],
    reverse: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Confirm an endpoint rescue without privileging either time direction."""

    def directional(item: dict[str, Any]) -> tuple[bool, bool, bool]:
        area_passed = bool(float(item["area_ratio"]) <= float(settings["max_area_ratio"]))
        eligible = bool(item.get("trusted", False)) and area_passed
        strong_iou = bool(eligible and float(item["iou"]) >= float(settings["min_endpoint_iou"]))
        center_confirmed = bool(
            eligible and float(item["center_distance"]) <= float(settings["max_endpoint_center_distance"])
        )
        return area_passed, strong_iou, center_confirmed

    forward_area, forward_strong, forward_center = directional(forward)
    reverse_area, reverse_strong, reverse_center = directional(reverse)
    if forward_strong or reverse_strong:
        basis = "strong_iou"
    elif forward_center and reverse_center:
        basis = "bilateral_center"
    else:
        basis = "rejected"
    return {
        "accepted": basis != "rejected",
        "confirmation_basis": basis,
        "forward_area_passed": forward_area,
        "reverse_area_passed": reverse_area,
        "forward_strong_iou": forward_strong,
        "reverse_strong_iou": reverse_strong,
        "forward_center_confirmed": forward_center,
        "reverse_center_confirmed": reverse_center,
    }


def _candidate(state: ObjectState, frame_idx: int, result: dict[str, Any]) -> dict[str, Any]:
    reference = state.last_detection_box
    value = np.asarray(result["box"], dtype=np.float64)
    return {
        "frame_idx": frame_idx,
        "track_id": state.track["track_id"],
        "source": "kalman_optical_flow",
        "box": value.tolist(),
        "motion_box": value.tolist(),
        "direction": 1,
        "anchor_frame": state.last_detection_frame,
        "selected_points": int(result.get("selected", 0)),
        "inlier_points": int(result.get("inliers", 0)),
        "flow_inlier_fraction": float(result.get("inlier_fraction", 0.0)),
        "quality": float(result.get("quality", 0.0)),
        "flow_continuity": (
            "trusted_cycle_recovery"
            if bool(result.get("recovered_from_coast", False))
            else "direct"
            if bool(result.get("valid", False))
            else "provisional_coast"
        ),
        "flow_cycle_iou": result.get("cycle_iou"),
        "_flow_trusted": bool(result.get("valid", False)),
        "area_ratio": float(area_ratio(reference, value)),
        "center_distance": float(normalized_center_distance(reference, value)),
    }


def _corridor(boxes: list[list[float] | np.ndarray], expansion: float) -> tuple[int, int, int, int]:
    values = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    maximum_side = max(
        16.0, float(np.max(np.maximum(values[:, 2] - values[:, 0], values[:, 3] - values[:, 1])))
    )
    margin = maximum_side * expansion * 0.5
    return (
        math.floor(float(np.min(values[:, 0])) - margin),
        math.floor(float(np.min(values[:, 1])) - margin),
        math.ceil(float(np.max(values[:, 2])) + margin),
        math.ceil(float(np.max(values[:, 3])) + margin),
    )


def _bounded_corridor(
    value: list[float] | np.ndarray, expansion: float, maximum_side: int
) -> tuple[int, int, int, int]:
    raw = _corridor([value], expansion)
    width, height = raw[2] - raw[0], raw[3] - raw[1]
    if width <= maximum_side and height <= maximum_side:
        return raw
    box = np.asarray(value, dtype=np.float64)
    center = (box[:2] + box[2:]) * 0.5
    width, height = min(width, maximum_side), min(height, maximum_side)
    x1 = math.floor(float(center[0]) - width * 0.5)
    y1 = math.floor(float(center[1]) - height * 0.5)
    return (
        x1,
        y1,
        x1 + width,
        y1 + height,
    )


def _consensus_corridor(
    values: list[list[float] | np.ndarray],
    expansion: float,
    maximum_side: int,
) -> tuple[int, int, int, int] | None:
    """Build one direction-neutral crop, or decline rather than clip anchors."""

    raw = _corridor(values, expansion)
    width, height = raw[2] - raw[0], raw[3] - raw[1]
    if width <= maximum_side and height <= maximum_side:
        return raw
    return None


def _translate_box(value: list[float] | np.ndarray, x: float, y: float) -> list[float]:
    return (np.asarray(value, dtype=np.float64) + np.asarray([x, y, x, y])).tolist()


def _trusted_local_geometry(
    target: list[float] | np.ndarray,
    review: dict[str, Any],
    settings: dict[str, Any],
) -> np.ndarray | None:
    if not bool(settings.get("enabled", False)):
        return None
    value = review.get("local_box")
    confidence = review.get("local_confidence")
    if value is None or confidence is None:
        return None
    candidate = np.asarray(value, dtype=np.float64)
    reference = np.asarray(target, dtype=np.float64)
    if float(confidence) < float(settings["min_local_confidence"]):
        return None
    if area_ratio(candidate, reference) > float(settings["max_area_ratio"]):
        return None
    if normalized_center_distance(candidate, reference) > float(settings["max_center_distance"]):
        return None
    return candidate


def _trusted_reliable_recovery_geometry(
    target: list[float] | np.ndarray,
    review: dict[str, Any],
    settings: dict[str, Any],
    recovery: dict[str, Any],
) -> np.ndarray | None:
    value = review.get("local_box")
    confidence = review.get("local_confidence")
    if value is None or confidence is None:
        return None
    candidate = np.asarray(value, dtype=np.float64)
    reference = np.asarray(target, dtype=np.float64)
    if float(confidence) < float(recovery["min_local_confidence"]):
        return None
    if area_ratio(candidate, reference) > float(settings["max_area_ratio"]):
        return None
    if normalized_center_distance(candidate, reference) > float(recovery["max_center_distance"]):
        return None
    if iou(candidate, reference) < float(recovery["min_iou"]) and containment(candidate, reference) < float(
        recovery["min_containment"]
    ):
        return None
    return candidate


def _box_with_center_and_size(
    center_source: list[float] | np.ndarray,
    size_source: list[float] | np.ndarray,
) -> np.ndarray:
    center_box = np.asarray(center_source, dtype=np.float64)
    size_box = np.asarray(size_source, dtype=np.float64)
    center = (center_box[:2] + center_box[2:]) * 0.5
    size = np.maximum(2.0, size_box[2:] - size_box[:2])
    return np.concatenate((center - size * 0.5, center + size * 0.5))


def _filter_local_measurement(
    reference: list[float] | np.ndarray,
    measurement: list[float] | np.ndarray,
    confidence: float,
    settings: dict[str, Any] | None,
    *,
    recovery: bool = False,
) -> np.ndarray:
    """Fuse a local face box without allowing a one-frame size reset."""

    value = np.asarray(measurement, dtype=np.float64)
    if not settings or not bool(settings.get("enabled", False)):
        return value
    prior = np.asarray(reference, dtype=np.float64)
    prior_size = np.maximum(2.0, prior[2:] - prior[:2])
    value_size = np.maximum(2.0, value[2:] - value[:2])
    prior_center = (prior[:2] + prior[2:]) * 0.5
    value_center = (value[:2] + value[2:]) * 0.5
    low, high = float(settings["confidence_low"]), float(settings["confidence_high"])
    strength = max(0.0, min(1.0, (float(confidence) - low) / max(1e-9, high - low)))
    center_gain = float(settings["center_gain_low"]) + strength * (
        float(settings["center_gain_high"]) - float(settings["center_gain_low"])
    )
    if recovery:
        center_gain = max(center_gain, float(settings["recovery_center_gain"]))
    size_gain = float(settings["size_gain_low"]) + strength * (
        float(settings["size_gain_high"]) - float(settings["size_gain_low"])
    )
    center_delta = value_center - prior_center
    maximum_center = float(settings["max_center_step"]) * math.sqrt(float(prior_size[0] * prior_size[1]))
    center_length = float(np.linalg.norm(center_delta))
    if center_length > maximum_center:
        center_delta *= maximum_center / max(center_length, 1e-9)
    center = prior_center + center_gain * center_delta
    maximum_log_size = math.log(float(settings["max_size_ratio_per_update"]))
    log_delta = np.clip(
        np.log(value_size) - np.log(prior_size),
        -maximum_log_size,
        maximum_log_size,
    )
    size = np.exp(np.log(prior_size) + size_gain * log_delta)
    return np.concatenate((center - size * 0.5, center + size * 0.5))


def _measurement_filter_for_scope(refinement: dict[str, Any], phase: str) -> dict[str, Any] | None:
    """Return measurement settings only when the configured phase may use them."""

    settings = refinement.get("measurement_filter")
    if not isinstance(settings, dict) or not bool(settings.get("enabled", False)):
        return None
    scope = str(settings.get("scope", "all"))
    if scope == "tracking_only" and phase != "tracking":
        return None
    return settings


def _stabilize_reviewed_geometry(
    reviewed: list[tuple[dict[str, Any], dict[str, Any]]],
    reference: np.ndarray,
    max_gap_frames: int,
) -> None:
    """Fill only short gaps between direct local-SCRFD geometry anchors."""

    anchors = [
        index
        for index, (item, _review) in enumerate(reviewed)
        if str(item.get("geometry_source", "")).startswith("local_scrfd")
    ]
    if not anchors:
        return
    for index, (item, review) in enumerate(reviewed):
        if index in anchors:
            continue
        frame_idx = int(item["frame_idx"])
        prior = max((value for value in anchors if value < index), default=None)
        following = min((value for value in anchors if value > index), default=None)
        replacement: np.ndarray | None = None
        source: str | None = None
        if prior is not None and following is not None:
            prior_item, following_item = reviewed[prior][0], reviewed[following][0]
            prior_frame = int(prior_item["frame_idx"])
            following_frame = int(following_item["frame_idx"])
            if following_frame - prior_frame - 1 <= max_gap_frames:
                weight = (frame_idx - prior_frame) / max(1, following_frame - prior_frame)
                replacement = (
                    np.asarray(prior_item["motion_box"], dtype=np.float64) * (1.0 - weight)
                    + np.asarray(following_item["motion_box"], dtype=np.float64) * weight
                )
                source = "local_scrfd_interpolation"
        elif prior is not None:
            prior_item = reviewed[prior][0]
            if frame_idx - int(prior_item["frame_idx"]) <= max_gap_frames:
                replacement = np.asarray(prior_item["motion_box"], dtype=np.float64)
                source = "local_scrfd_coast"
        elif following is not None:
            following_item = reviewed[following][0]
            if int(following_item["frame_idx"]) - frame_idx <= max_gap_frames:
                replacement = np.asarray(following_item["motion_box"], dtype=np.float64)
                source = "local_scrfd_coast"
        if replacement is None or source is None:
            continue
        raw = np.asarray(item["motion_box"], dtype=np.float64)
        item["raw_motion_box"] = raw.tolist()
        item["box"] = replacement.tolist()
        item["motion_box"] = replacement.tolist()
        item["geometry_source"] = source
        item["area_ratio"] = float(area_ratio(reference, replacement))
        item["center_distance"] = float(normalized_center_distance(reference, replacement))
        review["raw_box"] = raw.tolist()
        review["box"] = replacement.tolist()
        review["geometry_source"] = source


class StreamingEngine:
    def __init__(
        self,
        source: Path,
        workdir: Path,
        config: dict[str, Any],
        detector: Any,
        *,
        face_analysis: Any | None = None,
        review_face_analysis: Any | None = None,
    ):
        self.source = source
        self.config = config
        self._collect_developer_audits = config.get("output", {}).get(
            "artifacts_level", "final"
        ) in {"audit", "debug"}
        self.metadata = probe_video(source)
        self.fps = float(self.metadata.fps)
        self.max_analysis_fps = float(config["scan"]["max_analysis_fps"])
        self.analysis_fps_tolerance_multiplier = ANALYSIS_FPS_TOLERANCE_MULTIPLIER
        self.allowed_regular_analysis_fps = (
            self.max_analysis_fps * self.analysis_fps_tolerance_multiplier
        )
        self.detector_frame_stride = resolve_analysis_stride(
            self.fps,
            self.max_analysis_fps,
            tolerance_multiplier=self.analysis_fps_tolerance_multiplier,
        )
        self.nominal_regular_analysis_fps = self.fps / self.detector_frame_stride
        self.between_scan_frames = str(
            config["tracking"].get("between_scan_frames", "interpolate")
        )
        self.interpolate_tracking = _interpolate_tracking_enabled(
            self.between_scan_frames,
            self.detector_frame_stride,
        )
        # Every automatically sampled cadence uses the same reduced-work
        # review policy, including less common source/ceiling ratios that
        # derive strides such as 3, 8, or 16.
        self.fast_review_mode = self.detector_frame_stride > 1
        self.local_review_stride = (
            self.detector_frame_stride if self.fast_review_mode else 1
        )
        self.local_review_phase = (
            self.local_review_stride // 2 if self.fast_review_mode else 0
        )
        self._verifier_scores: dict[
            tuple[str, int, tuple[float, ...]], float
        ] = {}
        self.local_review_attempts = 0
        self.local_review_sampled_out = 0
        self.local_review_forced = 0
        self.verifier_review_calls = 0
        self.verifier_review_cache_hits = 0
        rule_gate = (
            config.get("revalidation", {})
            .get("policy", {})
            .get("rule_gate", {})
        )
        self.detector_scan_burst_frames = _detector_scan_burst_frames(rule_gate)
        self.forced_detector_scan_reasons: dict[int, set[str]] = {}
        if self.detector_frame_stride > 1:
            self._force_detector_scan_range(
                1,
                self.detector_scan_burst_frames - 1,
                "video_start_burst",
            )
        self.settings = dict(config["tracking"])
        minimum_endpoint_extension = self.detector_frame_stride - 1
        for setting in ("endpoint_extension", "reliable_endpoint_extension"):
            self.settings[setting] = max(
                int(self.settings.get(setting, 0)),
                minimum_endpoint_extension,
            )
        self.settings["max_missed_frames"] = max(
            1, math.ceil(float(config["streaming"]["max_missed_seconds"]) * self.fps)
        )
        self.settings["source_fps"] = self.fps
        self.max_retroactive_frames = max(
            self.settings["max_missed_frames"],
            math.ceil(float(config["streaming"]["max_retroactive_seconds"]) * self.fps),
        )
        self.face_analysis = face_analysis or make_face_analysis(
            config,
        )
        self.detector = self.face_analysis.models[DETECTION_TASK]
        if detector is not None and self.detector is not detector:
            raise ValueError("injected detector does not match FaceAnalysis detection task model")
        self.scanner = ScanRunner(
            config,
            detector=detector,
            face_analysis=self.face_analysis,
        )
        self.review_face_analysis = review_face_analysis or make_review_face_analysis(config)
        self.reviewer = LocalReviewer(
            config,
            face_analysis=self.review_face_analysis,
            verifier=self.face_analysis.models[VERIFICATION_TASK],
        )
        recognition_started = time.perf_counter()
        recognition_settings = config.get("recognition", {"mode": "all"})
        try:
            if str(recognition_settings.get("mode", "all")) == "all":
                # Do not even look up the package recognizer in the default mode.
                self.recognition_engine = create_recognition_engine(
                    recognition_settings,
                    recognizer=None,
                    gallery_detector=None,
                )
            else:
                recognizer = self.face_analysis.models[RECOGNITION_TASK]
                self.recognition_engine = create_recognition_engine(
                    recognition_settings,
                    recognizer=recognizer,
                    gallery_detector=self._gallery_faces,
                )
        except Exception:
            self.scanner.close()
            raise
        self.recognition_setup_seconds = (
            time.perf_counter() - recognition_started if self.recognition_engine.enabled else 0.0
        )
        self.recognition_candidate_prepare_seconds = 0.0
        self.recognition_candidate_errors = 0
        self.recognition_candidate_rejections = 0
        self.recognition_candidates_prepared = 0
        self.recognition_candidates: dict[str, list[tuple[str, RecognitionCandidate]]] = {}
        self.recognition_candidate_bytes = 0
        self.recognition_candidate_peak_bytes = 0
        self.recognition_candidate_drops = 0
        self.recognition_candidate_overflow_tracks: set[str] = set()
        self.recognition_local_landmark_available = 0
        self.recognition_local_landmark_high_confidence = 0
        self.recognition_local_landmark_strict_agreement = 0
        self.recognition_local_landmark_selected = 0
        self.recognition_local_landmark_quality_rejections = 0
        self.recognition_local_detector_candidates = 0
        self.recognition_local_tracking_candidates = 0
        self.recognition_local_duplicate_candidates = 0
        self.recognition_local_detector_attempts = 0
        self.recognition_local_tracking_attempts = 0
        self.recognition_landmark_source_stats: dict[str, Counter[str]] = {
            "local_scrfd": Counter(),
            "global_scrfd": Counter(),
        }
        self.recognition_candidate_max_bytes = _RECOGNITION_CANDIDATE_MAX_BYTES
        self.recognition_pool_limit = max(8, 2 * int(self.recognition_engine.max_frames_per_track))
        self.cache = EncodedPacketCache(workdir / "encoded-packets.sqlite", source)
        self.decoded_frame_store = DecodedFrameStore(
            frame_target=_decoded_frame_store_target(config, self.fps),
            byte_capacity=int(
                config["streaming"].get("recent_frame_cache_max_bytes", 0)
            ),
        )
        self.decoded_frame_bytes = (
            int(getattr(self.metadata, "width", 0))
            * int(getattr(self.metadata, "height", 0))
            * 3
        )
        self.states: list[ObjectState] = []
        self.tracks: list[dict[str, Any]] = []
        self.detections: list[dict[str, Any]] = []
        self.audits: list[dict[str, Any]] = []
        self.candidates: list[dict[str, Any]] = []
        self.evidence: list[dict[str, Any]] = []
        self.endpoint_affine_candidates: list[dict[str, Any]] = []
        self.endpoint_affine_jobs = 0
        self.endpoint_affine_frames = 0
        self.endpoint_affine_published_frames = 0
        self.endpoint_verifier_checkpoints = 0
        self.endpoint_verifier_refinement_frames = 0
        self.endpoint_local_review_frames: dict[str, set[int]] = {}
        self.interpolate_endpoint_jobs = 0
        self.interpolate_endpoint_frames = 0
        self.interpolate_endpoint_published_frames = 0
        self.interpolate_endpoint_seconds = 0.0
        self.interpolate_endpoint_reason_counts: Counter[str] = Counter()
        self.scene_cut_detector = SceneCutDetector(config["scan"])
        self.scene_segment_id = 0
        self.reverse_jobs = 0
        self.reverse_frames = 0
        self.bidirectional_gap_jobs = 0
        self.bidirectional_gap_frames = 0
        self.bidirectional_accepted_frames = 0
        self.bidirectional_rejected_frames = 0
        self.bidirectional_review_resolutions = 0
        self.bidirectional_skipped_jobs = 0
        self.bidirectional_association_attempts = 0
        self.bidirectional_association_rescues = 0
        self.bidirectional_audits: list[dict[str, Any]] = []
        self.interpolation_jobs = 0
        self.interpolated_frames = 0
        self.long_gap_reanchors = 0
        self.detector_scan_opportunities = 0
        self.discarded_unanchored_tail_frames = 0
        self.endpoint_conflicts = EndpointConflictReview(
            config, self.fps, int(self.metadata.frame_count)
        )
        self._conflict_candidate_cursor = 0
        self._conflict_detection_cursor = 0
        self._conflict_evidence_cursor = 0
        self._endpoint_conflict_aliases: dict[str, str] = {}

    def _collect_endpoint_conflict_evidence(
        self,
        candidate: dict[str, Any],
        image: np.ndarray,
        origin: tuple[int, int],
    ) -> None:
        collector = getattr(self, "endpoint_conflicts", None)
        if collector is None or not collector.enabled:
            return
        # These lists only append during streaming. Index each record once,
        # rather than rescan the entire video for every endpoint frame.
        for item in self.candidates[self._conflict_candidate_cursor:]:
            frame_values = collector.core.setdefault(int(item["frame_idx"]), {})
            prior = frame_values.get(str(item["track_id"]), {})
            if prior.get("source") != "detector":
                frame_values[str(item["track_id"])] = item
        self._conflict_candidate_cursor = len(self.candidates)
        # Detector anchors are stored separately from propagated candidates.
        # A current scan may still be awaiting association inside process().
        # Leave the cursor at that item until its track id is available.
        for item in self.detections[self._conflict_detection_cursor:]:
            if item.get("track_id") is None:
                break
            collector.core.setdefault(int(item["frame_idx"]), {})[
                str(item["track_id"])
            ] = item
            self._conflict_detection_cursor += 1
        for item in self.evidence[self._conflict_evidence_cursor:]:
            collector.evidence[(str(item["track_id"]), int(item["frame_idx"]))] = item
        self._conflict_evidence_cursor = len(self.evidence)
        collector.collect(
            candidate, image=image, origin=origin,
            decode=self._decode_frames, reviewer=self.reviewer,
        )

    def _resolve_endpoint_conflicts(
        self,
        observations: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        endpoint_candidates: list[dict[str, Any]] | None = None,
    ) -> set[tuple[str, int, int]]:
        collector = getattr(self, "endpoint_conflicts", None)
        if collector is None:
            return set()
        return collector.resolve(
            observations, evidence, getattr(self, "_endpoint_conflict_aliases", {}),
            endpoint_candidates=endpoint_candidates,
        )

    def _force_detector_scan_range(self, first: int, last: int, reason: str) -> None:
        """Request real full-frame scans for a bounded future frame range."""

        if self.detector_frame_stride == 1 or last < first:
            return
        # The probed frame count is only a progress estimate until decoding
        # reaches EOF.  Do not let an under-reported value truncate a short,
        # bounded adaptive burst; entries beyond the real EOF are harmless and
        # disappear with this engine.
        for frame_idx in range(max(0, int(first)), int(last) + 1):
            self.forced_detector_scan_reasons.setdefault(frame_idx, set()).add(reason)

    def _detector_scan_reason(self, frame_idx: int) -> str | None:
        """Return why this frame needs the expensive multi-view full-frame scan."""

        if self.detector_frame_stride == 1:
            return "every_frame"
        if frame_idx % self.detector_frame_stride == 0:
            return "regular_stride"
        if frame_idx == int(self.metadata.frame_count) - 1:
            return "end_of_stream"
        reasons = self.forced_detector_scan_reasons.get(frame_idx)
        if not reasons:
            return None
        priority = (
            "scene_cut",
            "scene_cut_burst",
            "new_track_burst",
            "video_start_burst",
        )
        return next((reason for reason in priority if reason in reasons), sorted(reasons)[0])

    def _gallery_faces(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect Gallery faces at fixed upright 640/128 SCRFD views."""

        return detect_gallery_faces_upright(
            self.face_analysis,
            image,
            max_detections=detector_max_detections(self.config),
        )

    def _capture_recognition_candidate(self, frame: np.ndarray, detection: dict[str, Any]) -> None:
        local_review = detection.pop("_recognition_local_review", None)
        self._capture_best_recognition_candidate(
            frame,
            detection,
            local_review,
            identifier=str(detection["detection_id"]),
        )

    def _capture_tracking_recognition_candidate(
        self,
        frame: np.ndarray,
        item: dict[str, Any],
        local_review: dict[str, Any],
        origin: tuple[int, int] = (0, 0),
    ) -> None:
        if item.get("source") == "detector":
            return
        self._capture_best_recognition_candidate(
            frame,
            item,
            local_review,
            identifier=(f"tracking:{item['track_id']}:{int(item['frame_idx'])}"),
            origin=origin,
        )

    @staticmethod
    def _ordered_landmark_proposals(
        item: dict[str, Any],
        local_review: dict[str, Any] | None,
    ) -> list[RecognitionLandmarkProposal]:
        proposals: list[RecognitionLandmarkProposal] = []
        if isinstance(local_review, dict):
            proposals.append(
                RecognitionLandmarkProposal(
                    source="local_scrfd",
                    box=local_review.get("local_box"),
                    landmarks=local_review.get("local_landmarks"),
                    confidence=local_review.get("local_confidence"),
                    scan_angle_degrees=int(local_review.get("local_angle") or 0),
                    require_reference_agreement=True,
                )
            )
        if item.get("source") == "detector":
            proposals.append(
                RecognitionLandmarkProposal(
                    source="global_scrfd",
                    box=item.get("detector_box"),
                    landmarks=item.get("detector_landmarks"),
                    confidence=item.get("confidence"),
                    scan_angle_degrees=int(item.get("scan_angle_degrees") or 0),
                    require_reference_agreement=False,
                )
            )
        return proposals

    def _capture_best_recognition_candidate(
        self,
        frame: np.ndarray,
        item: dict[str, Any],
        local_review: dict[str, Any] | None,
        *,
        identifier: str,
        origin: tuple[int, int] = (0, 0),
    ) -> None:
        recognition_engine = getattr(self, "recognition_engine", None)
        if recognition_engine is None or not recognition_engine.enabled:
            return
        if item.get("source") == "detector":
            self.recognition_local_detector_attempts = (
                getattr(
                    self,
                    "recognition_local_detector_attempts",
                    0,
                )
                + 1
            )
        else:
            self.recognition_local_tracking_attempts = (
                getattr(
                    self,
                    "recognition_local_tracking_attempts",
                    0,
                )
                + 1
            )
        started = time.perf_counter()
        try:
            for proposal in self._ordered_landmark_proposals(item, local_review):
                source_stats = self.recognition_landmark_source_stats[proposal.source]
                source_stats["attempted"] += 1
                aligned: np.ndarray | None = None
                try:
                    if (
                        proposal.box is None
                        or proposal.landmarks is None
                        or proposal.confidence is None
                    ):
                        source_stats["pair_missing"] += 1
                        continue
                    source_stats["pair_available"] += 1
                    if proposal.source == "local_scrfd":
                        self.recognition_local_landmark_available += 1
                    confidence = float(proposal.confidence)
                    if confidence < LOCAL_LANDMARK_CONFIDENCE_THRESHOLD:
                        source_stats["confidence_rejected"] += 1
                        continue
                    source_stats["confidence_passed"] += 1
                    if proposal.source == "local_scrfd":
                        self.recognition_local_landmark_high_confidence += 1
                    if proposal.require_reference_agreement:
                        reference_box = item.get(
                            "raw_box",
                            item.get("motion_box", item["box"]),
                        )
                        agreement, _metrics = local_landmark_box_agreement(
                            reference_box,
                            proposal.box,
                        )
                        if not agreement:
                            source_stats["agreement_rejected"] += 1
                            continue
                        source_stats["agreement_passed"] += 1
                        self.recognition_local_landmark_strict_agreement += 1
                    frame_box = _translate_box(
                        proposal.box,
                        -origin[0],
                        -origin[1],
                    )
                    frame_landmarks = np.asarray(
                        proposal.landmarks,
                        dtype=np.float64,
                    ).reshape(5, 2)
                    frame_landmarks -= np.asarray(origin, dtype=np.float64)
                    aligned = arcface_align_112(frame, frame_landmarks)
                    quality, eligible, _details = recognition_candidate_quality(
                        aligned,
                        frame_box,
                        frame_landmarks,
                        confidence,
                        frame.shape,
                        scan_angle_degrees=proposal.scan_angle_degrees,
                    )
                    if not eligible:
                        source_stats["quality_rejected"] += 1
                        if proposal.source == "local_scrfd":
                            self.recognition_local_landmark_quality_rejections += 1
                        if aligned.flags.writeable:
                            aligned.fill(0)
                        continue
                    candidate = RecognitionCandidate(
                        frame_index=int(item["frame_idx"]),
                        quality=quality,
                        aligned_face=aligned,
                        landmark_source=proposal.source,
                    )
                    aligned = None
                    source_stats["selected"] += 1
                    if proposal.source == "local_scrfd":
                        self.recognition_local_landmark_selected += 1
                        if item.get("source") == "detector":
                            self.recognition_local_detector_candidates += 1
                        else:
                            self.recognition_local_tracking_candidates += 1
                    self.recognition_candidates_prepared += 1
                    self._store_recognition_candidate(
                        str(item["track_id"]),
                        identifier,
                        candidate,
                    )
                    return
                except (TypeError, ValueError, cv2.error):
                    source_stats["errors"] += 1
                    self.recognition_candidate_errors += 1
                    if aligned is not None and aligned.flags.writeable:
                        aligned.fill(0)
            self.recognition_candidate_rejections += 1
        finally:
            self.recognition_candidate_prepare_seconds += time.perf_counter() - started

    def _store_recognition_candidate(
        self,
        track_id: str,
        identifier: str,
        candidate: RecognitionCandidate,
    ) -> None:
        values = self.recognition_candidates.setdefault(track_id, [])
        for index, (prior_identifier, prior) in enumerate(values):
            if prior_identifier != identifier:
                continue
            self.recognition_local_duplicate_candidates += 1
            source_rank = {"local_scrfd": 1, "global_scrfd": 0}
            if (
                source_rank[candidate.landmark_source],
                float(candidate.quality),
            ) <= (
                source_rank[prior.landmark_source],
                float(prior.quality),
            ):
                if candidate.aligned_face.flags.writeable:
                    candidate.aligned_face.fill(0)
                return
            if prior.aligned_face.flags.writeable:
                prior.aligned_face.fill(0)
            self.recognition_candidate_bytes -= int(prior.aligned_face.nbytes)
            values[index] = (identifier, candidate)
            self.recognition_candidate_bytes += int(candidate.aligned_face.nbytes)
            self.recognition_candidate_peak_bytes = max(
                self.recognition_candidate_peak_bytes,
                self.recognition_candidate_bytes,
            )
            return
        values.append((identifier, candidate))
        self.recognition_candidate_bytes += int(candidate.aligned_face.nbytes)
        # Keep twice the final profile's call budget, distributed over time.
        # The extra headroom lets final quality selection survive stitching.
        if len(values) > 2 * self.recognition_pool_limit:
            chosen = select_temporally_distributed(
                [value for _identifier, value in values],
                self.recognition_pool_limit,
            )
            chosen_ids = {id(value) for value in chosen}
            retained = [(identifier, value) for identifier, value in values if id(value) in chosen_ids]
            retained_arrays = {id(value.aligned_face) for _identifier, value in retained}
            for _identifier, value in values:
                if (
                    id(value) not in chosen_ids
                    and id(value.aligned_face) not in retained_arrays
                    and value.aligned_face.flags.writeable
                ):
                    value.aligned_face.fill(0)
            removed = len(values) - len(retained)
            self.recognition_candidates[track_id] = retained
            self.recognition_candidate_bytes -= sum(
                int(value.aligned_face.nbytes) for identifier, value in values if id(value) not in chosen_ids
            )
            self.recognition_candidate_drops += removed
        self._enforce_recognition_candidate_budget()
        self.recognition_candidate_peak_bytes = max(
            self.recognition_candidate_peak_bytes,
            self.recognition_candidate_bytes,
        )

    def _enforce_recognition_candidate_budget(self) -> None:
        """Prune globally in batches and fail closed for every touched track."""

        maximum = int(self.recognition_candidate_max_bytes)
        if self.recognition_candidate_bytes <= maximum:
            return
        target = max(0, int(maximum * _RECOGNITION_CANDIDATE_PRUNE_RATIO))
        ranked: list[tuple[int, int, float, int, str, int, int]] = []
        for track_id, values in self.recognition_candidates.items():
            if not values:
                continue
            best_index = max(
                range(len(values)),
                key=lambda index: (
                    values[index][1].landmark_source == "local_scrfd",
                    float(values[index][1].quality),
                    -int(values[index][1].frame_index),
                ),
            )
            for index, (_identifier, candidate) in enumerate(values):
                ranked.append(
                    (
                        0 if index == best_index else 1,
                        0 if candidate.landmark_source == "local_scrfd" else 1,
                        -float(candidate.quality),
                        int(candidate.frame_index),
                        track_id,
                        index,
                        int(candidate.aligned_face.nbytes),
                    )
                )
        ranked.sort()
        keep: set[tuple[str, int]] = set()
        retained_bytes = 0
        for _tier, _source, _quality, _frame, track_id, index, size in ranked:
            if retained_bytes + size > target:
                continue
            keep.add((track_id, index))
            retained_bytes += size

        removed_tracks: set[str] = set()
        removed_count = 0
        retained_arrays = {
            id(values[index][1].aligned_face)
            for track_id, index in keep
            for values in (self.recognition_candidates[track_id],)
        }
        for track_id, values in list(self.recognition_candidates.items()):
            retained_values = [value for index, value in enumerate(values) if (track_id, index) in keep]
            if len(retained_values) != len(values):
                removed_tracks.add(track_id)
                removed_count += len(values) - len(retained_values)
                for index, (_identifier, candidate) in enumerate(values):
                    if (
                        (track_id, index) not in keep
                        and id(candidate.aligned_face) not in retained_arrays
                        and candidate.aligned_face.flags.writeable
                    ):
                        candidate.aligned_face.fill(0)
            if retained_values:
                self.recognition_candidates[track_id] = retained_values
            else:
                del self.recognition_candidates[track_id]
        self.recognition_candidate_bytes = retained_bytes
        self.recognition_candidate_drops += removed_count
        self.recognition_candidate_overflow_tracks.update(removed_tracks)

    def _clear_recognition_candidates(self) -> None:
        """Best-effort zero and release sensitive aligned crops after inference."""

        seen: set[int] = set()
        for values in self.recognition_candidates.values():
            for _identifier, candidate in values:
                array = candidate.aligned_face
                if id(array) in seen:
                    continue
                seen.add(id(array))
                if array.flags.writeable:
                    array.fill(0)
        self.recognition_candidates.clear()
        self.recognition_candidate_bytes = 0

    def _finalize_recognition(self, aliases: dict[str, str]) -> dict[str, Any]:
        """Finalize identity decisions and always erase retained crop pixels."""

        try:
            return self._finalize_recognition_impl(aliases)
        finally:
            self._clear_recognition_candidates()

    def _finalize_recognition_impl(self, aliases: dict[str, str]) -> dict[str, Any]:
        if not self.recognition_engine.enabled:
            if self.recognition_candidates:
                raise RuntimeError("all recognition policy retained unexpected candidates")
            return {"enabled": False, "reason": "policy_all"}

        inference_started = time.perf_counter()
        by_track: dict[str, list[tuple[str, RecognitionCandidate]]] = {}
        for fragment_id, values in self.recognition_candidates.items():
            canonical = aliases.get(fragment_id, fragment_id)
            by_track.setdefault(canonical, []).extend(values)
        overflow_tracks = {
            aliases.get(fragment_id, fragment_id)
            for fragment_id in getattr(self, "recognition_candidate_overflow_tracks", set())
        }

        track_results: dict[str, dict[str, Any]] = {}
        track_recognizer_calls = 0
        status_counts = {"CONFIRMED": 0, "UNKNOWN": 0, "CONFLICT": 0}
        accepted_considered = 0
        accepted_overflow_tracks = 0
        for track in self.tracks:
            if not bool(track.get("accepted", False)):
                continue
            accepted_considered += 1
            track_id = str(track["track_id"])
            detection_ids = {
                str(value["detection_id"])
                for value in track.get("detections", [])
                if value.get("detection_id") is not None
            }
            intervals = [(int(first), int(last)) for first, last in track.get("accepted_intervals", [])]
            filtered: list[RecognitionCandidate] = []
            for candidate_id, candidate in by_track.get(track_id, []):
                frame_index = int(candidate.frame_index)
                tracking_only = candidate_id.startswith("tracking:")
                if (not tracking_only and candidate_id not in detection_ids) or not any(
                    first <= frame_index <= last for first, last in intervals
                ):
                    continue
                filtered.append(candidate)
            if track_id in overflow_tracks:
                # Pruning could have removed the very frame that exposes an
                # identity switch.  Never exempt from partial evidence.
                decision = self.recognition_engine.unknown_decision("candidate_memory_overflow")
                accepted_overflow_tracks += 1
            else:
                # RecognitionEngine owns the one final sampling operation so
                # temporal-threshold evidence and the crops sent to ArcFace
                # can never diverge.
                # Model failures must fail the task, rather than being treated
                # as an ordinary unmatched person that blur_only would expose.
                decision = self.recognition_engine.identify_track(
                    filtered,
                    frames_per_second=self.fps,
                )
                track_recognizer_calls += int(decision.selected_frame_count or 0)
            decision_record = decision.to_dict()
            decision_record["original_track_ids"] = list(track.get("stitched_track_ids", [track_id]))
            track_results[track_id] = decision_record
            track["identity"] = decision_record
            status_counts[decision.status.value] += 1

        inference_seconds = time.perf_counter() - inference_started
        gallery = self.recognition_engine.gallery
        gallery_recognizer_calls = len(gallery.references)
        retained_candidates = sum(len(values) for values in self.recognition_candidates.values())
        retained_candidate_bytes = int(getattr(self, "recognition_candidate_bytes", 0))
        artifact = {
            "enabled": True,
            "mode": self.recognition_engine.mode,
            "profile": self.recognition_engine.profile.name,
            "max_frames_per_track": int(self.recognition_engine.max_frames_per_track),
            "similarity_threshold": float(self.recognition_engine.similarity_threshold),
            "temporal_threshold_policy": {
                "base_similarity_threshold": float(self.recognition_engine.similarity_threshold),
                "minimum_selected_frames": (TEMPORAL_EVIDENCE_MIN_SELECTED_FRAMES),
                "minimum_adjacent_gap_seconds": (TEMPORAL_EVIDENCE_MIN_ADJACENT_GAP_SECONDS),
                "correlated_evidence_similarity_offset": (TEMPORAL_EVIDENCE_SIMILARITY_OFFSET),
                "single_frame_similarity_offset": (SINGLE_FRAME_SIMILARITY_OFFSET),
            },
            "unknown_action": self.recognition_engine.unknown_action,
            "landmark_policy": {
                "mode": "local_then_global_scrfd",
                "priority": ["local_scrfd", "global_scrfd"],
                "local_rotation_selection": "highest_confidence",
                "confidence_threshold": (LOCAL_LANDMARK_CONFIDENCE_THRESHOLD),
                "candidate_sources": ["detector", "tracking"],
                "global_scope": "full_frame_detector_observations",
                "box_landmarks_pair_required": True,
                "strict_box_agreement": {
                    "max_center_distance": (LOCAL_LANDMARK_MAX_CENTER_DISTANCE),
                    "max_area_ratio": LOCAL_LANDMARK_MAX_AREA_RATIO,
                    "min_iou": LOCAL_LANDMARK_MIN_IOU,
                    "min_containment": LOCAL_LANDMARK_MIN_CONTAINMENT,
                },
            },
            "reference_detection_policy": {
                "input_sizes": list(GALLERY_DETECTOR_INPUT_SIZES),
                "angles": [0],
                "confidence_threshold": GALLERY_DETECTOR_CONFIDENCE_THRESHOLD,
                "landmark_source": "full_frame_scrfd",
                "local_revalidation": False,
                "face_selection": "largest",
            },
            "references": {
                "fingerprint": getattr(gallery, "fingerprint", None),
                "accepted_images": len(gallery.references),
                "skipped_images": len(gallery.rejections),
                "files": [
                    {"file": item.relative_file_name,
                     "detected_face_count": item.detected_face_count,
                     "selected_box": list(item.selected_box)}
                    for item in gallery.references
                ],
                "skipped": [
                    {"file": item.relative_file_name, "reason": item.reason}
                    for item in gallery.rejections
                ],
            },
            "tracks": track_results,
            "statistics": {
                "accepted_tracks_considered": accepted_considered,
                "candidate_crops_prepared": self.recognition_candidates_prepared,
                "candidate_crops_retained_before_inference": retained_candidates,
                "candidate_crop_bytes_retained_before_inference": (retained_candidate_bytes),
                "candidate_crop_peak_bytes": int(getattr(self, "recognition_candidate_peak_bytes", 0)),
                "candidate_crop_drops": int(getattr(self, "recognition_candidate_drops", 0)),
                "candidate_overflow_fragments": len(
                    getattr(self, "recognition_candidate_overflow_tracks", set())
                ),
                "accepted_candidate_overflow_tracks": (accepted_overflow_tracks),
                "candidate_prepare_errors": self.recognition_candidate_errors,
                "candidate_quality_rejections": (self.recognition_candidate_rejections),
                "landmark_sources": {
                    source: dict(values)
                    for source, values in getattr(
                        self,
                        "recognition_landmark_source_stats",
                        {},
                    ).items()
                },
                "local_landmark_available": (getattr(self, "recognition_local_landmark_available", 0)),
                "local_landmark_high_confidence": (
                    getattr(
                        self,
                        "recognition_local_landmark_high_confidence",
                        0,
                    )
                ),
                "local_landmark_strict_agreement": (
                    getattr(
                        self,
                        "recognition_local_landmark_strict_agreement",
                        0,
                    )
                ),
                "local_landmark_selected": (getattr(self, "recognition_local_landmark_selected", 0)),
                "local_detector_candidates": getattr(
                    self,
                    "recognition_local_detector_candidates",
                    0,
                ),
                "local_tracking_candidates": getattr(
                    self,
                    "recognition_local_tracking_candidates",
                    0,
                ),
                "local_detector_attempts": getattr(
                    self,
                    "recognition_local_detector_attempts",
                    0,
                ),
                "local_tracking_attempts": getattr(
                    self,
                    "recognition_local_tracking_attempts",
                    0,
                ),
                "local_duplicate_candidates": getattr(
                    self,
                    "recognition_local_duplicate_candidates",
                    0,
                ),
                "local_landmark_quality_rejections": (
                    getattr(
                        self,
                        "recognition_local_landmark_quality_rejections",
                        0,
                    )
                ),
                "track_recognizer_calls": track_recognizer_calls,
                "reference_recognizer_calls": gallery_recognizer_calls,
                "total_recognizer_calls": (track_recognizer_calls + gallery_recognizer_calls),
                "status_counts": status_counts,
                "setup_seconds": self.recognition_setup_seconds,
                "candidate_prepare_seconds": self.recognition_candidate_prepare_seconds,
                "track_inference_seconds": inference_seconds,
                "total_seconds": (
                    self.recognition_setup_seconds
                    + self.recognition_candidate_prepare_seconds
                    + inference_seconds
                ),
            },
        }
        if not status_counts.get("CONFIRMED", 0):
            logger.warning("No specified people were matched in the video.")
        return artifact

    def _remember_frame(self, frame_idx: int, frame: np.ndarray) -> None:
        self.decoded_frame_store.remember(frame_idx, frame)

    def _decode_frames(
        self,
        first_frame: int,
        last_frame: int,
        *,
        crop: tuple[int, int, int, int] | None = None,
        crops: dict[int, tuple[int, int, int, int]] | None = None,
    ) -> dict[int, np.ndarray]:
        """Read one range through the shared complete-frame store.

        Complete oriented BGR frames, rather than crop-specific results, are
        retained so overlapping history consumers can reuse one packet decode.
        If retention is disabled or a single source frame exceeds its byte
        capacity, preserve the former crop-at-decode path and memory behavior.
        """

        if crop is not None and crops is not None:
            raise ValueError("crop and crops are mutually exclusive")
        store = self.decoded_frame_store
        if (
            store.frame_target <= 0
            or store.byte_capacity <= 0
            or self.decoded_frame_bytes > store.byte_capacity
        ):
            return self.cache.decode_range(
                first_frame,
                last_frame,
                crop=crop,
                crops=crops,
            )
        return store.read_range(
            first_frame,
            last_frame,
            loader=lambda first, last: self.cache.decode_range(first, last),
            crop=crop,
            crops=crops,
        )

    def _review_plan(
        self,
        item: dict[str, Any],
        *,
        force_local: bool = False,
        local_review_reason: str | None = None,
    ) -> ReviewPlan:
        """Return the one shared review schedule for every pipeline context."""

        frame_idx = int(item["frame_idx"])
        if not self.fast_review_mode:
            return ReviewPlan(
                run_local_scrfd=True,
                run_verifier=item.get("source") == "detector",
                local_review_reason=(local_review_reason or "full_review"),
            )
        scheduled = frame_idx % self.local_review_stride == self.local_review_phase
        if force_local:
            reason = local_review_reason or "forced_geometry_review"
        elif scheduled:
            reason = "stride_phase"
        else:
            reason = "sampled_out"
        return ReviewPlan(
            run_local_scrfd=bool(force_local or scheduled),
            run_verifier=True,
            local_review_reason=reason,
        )

    @staticmethod
    def _empty_local_review() -> dict[str, Any]:
        return {
            "local_match_count": -1,
            "local_confidence": None,
            "local_box": None,
            "local_landmarks": None,
            "local_angle": None,
            "local_review_variant": None,
            "local_edge_shift": None,
        }

    def _verify_once(
        self,
        frame: np.ndarray,
        item: dict[str, Any],
        box: list[float],
        *,
        cache_box: list[float] | None = None,
    ) -> float:
        identifier = str(
            item.get("detection_id")
            or item.get("track_id")
            or "unassigned"
        )
        key = (
            identifier,
            int(item["frame_idx"]),
            tuple(
                round(float(component), 4)
                for component in (box if cache_box is None else cache_box)
            ),
        )
        scores = getattr(self, "_verifier_scores", None)
        if scores is None:
            scores = {}
            self._verifier_scores = scores
        cached = scores.get(key)
        if cached is not None:
            self.verifier_review_cache_hits = (
                getattr(self, "verifier_review_cache_hits", 0) + 1
            )
            return cached
        score = float(self.reviewer.verify(frame, [box])[0]["face_probability"])
        scores[key] = score
        self.verifier_review_calls = getattr(self, "verifier_review_calls", 0) + 1
        return score

    def _measure_review(
        self,
        frame: np.ndarray,
        item: dict[str, Any],
        origin: tuple[int, int] = (0, 0),
        candidate_selection: str = "confidence",
        *,
        force_local: bool = False,
        local_review_reason: str | None = None,
    ) -> dict[str, Any]:
        local_box = _translate_box(item["box"], -origin[0], -origin[1])
        plan = self._review_plan(
            item,
            force_local=force_local,
            local_review_reason=local_review_reason,
        )
        if plan.run_local_scrfd:
            review = self.reviewer.local_match(
                frame,
                local_box,
                candidate_selection=candidate_selection,
            )
            self.local_review_attempts += 1
            if force_local and self.fast_review_mode:
                self.local_review_forced += 1
        else:
            review = self._empty_local_review()
            self.local_review_sampled_out += 1
        if review.get("local_box") is not None:
            review["local_box"] = _translate_box(review["local_box"], origin[0], origin[1])
        if review.get("local_landmarks") is not None:
            local_landmarks = np.asarray(review["local_landmarks"], dtype=np.float64).reshape(-1, 2)
            local_landmarks += np.asarray(origin, dtype=np.float64)
            review["local_landmarks"] = local_landmarks.tolist()
        review["local_review_reason"] = plan.local_review_reason
        review["verifier_face_probability"] = (
            self._verify_once(
                frame,
                item,
                local_box,
                cache_box=list(item["box"]),
            )
            if plan.run_verifier
            else None
        )
        return review

    def _review(
        self,
        frame: np.ndarray,
        item: dict[str, Any],
        origin: tuple[int, int] = (0, 0),
        candidate_selection: str = "confidence",
        *,
        force_local: bool = False,
        local_review_reason: str | None = None,
    ) -> dict[str, Any]:
        # Detector measurements are computed before association so their local
        # SCRFD geometry can participate in matching.  Reuse that measurement
        # after the track id is known instead of running SCRFD/Verifier twice.
        cached = item.pop("_review_measurement", None)
        forced_cached_upgrade = bool(
            cached is not None
            and force_local
            and int(cached.get("local_match_count", -1)) < 0
        )
        reusable_cached = bool(
            cached is not None
            and origin == (0, 0)
            and candidate_selection == "confidence"
            and not forced_cached_upgrade
        )
        review = cached if reusable_cached else self._measure_review(
            frame,
            item,
            origin,
            candidate_selection=candidate_selection,
            force_local=force_local,
            local_review_reason=local_review_reason,
        )
        if forced_cached_upgrade:
            # The preliminary off-phase skip was replaced by a real forced
            # measurement for this same observation; keep the audit count in
            # terms of final review decisions.
            self.local_review_sampled_out = max(
                0,
                self.local_review_sampled_out - 1,
            )
        if (
            item.get("source") == "detector"
            and getattr(self, "recognition_engine", None) is not None
            and self.recognition_engine.enabled
        ):
            item["_recognition_local_review"] = {
                "local_box": review.get("local_box"),
                "local_landmarks": review.get("local_landmarks"),
                "local_confidence": review.get("local_confidence"),
                "local_angle": review.get("local_angle"),
            }
        evidence = {
            "track_id": item["track_id"],
            "frame_idx": int(item["frame_idx"]),
            "source": item["source"],
            "box": list(item["box"]),
            "admission_scope": item.get("admission_scope", "core"),
            **review,
        }
        if item.get("detector_box") is not None:
            evidence["detector_box"] = list(item["detector_box"])
        if item.get("detector_landmarks") is not None:
            evidence["detector_landmarks"] = item["detector_landmarks"]
        if item.get("confidence") is not None:
            evidence["confidence"] = float(item["confidence"])
        self.evidence.append(evidence)
        return evidence

    def _refine_detection_geometry(
        self,
        detection: dict[str, Any],
        review: dict[str, Any],
        motion_reference: np.ndarray | None,
    ) -> bool:
        if detection.get("geometry_source") == "local_scrfd":
            # This detector hit was already refined before association. Keep
            # the original full-frame box in raw_box and use the local box as
            # a confidence-weighted measurement when temporal filtering is on.
            if motion_reference is not None:
                local = np.asarray(detection["box"], dtype=np.float64)
                filter_settings = _measurement_filter_for_scope(
                    self.config["revalidation"]["geometry_refinement"],
                    "detection",
                )
                filtered = _filter_local_measurement(
                    motion_reference,
                    local,
                    float(review.get("local_confidence") or 0.0),
                    filter_settings,
                )
                if bool((filter_settings or {}).get("enabled", False)):
                    review["local_measurement_box"] = local.tolist()
                detection["box"] = filtered.tolist()
                detection["geometry_source"] = (
                    "local_scrfd_filtered"
                    if bool((filter_settings or {}).get("enabled", False))
                    else "local_scrfd"
                )
            review["box"] = list(detection["box"])
            review["geometry_source"] = detection["geometry_source"]
            return True
        raw = np.asarray(detection["box"], dtype=np.float64)
        settings = self.config["revalidation"].get("geometry_refinement", {})
        local = _trusted_local_geometry(raw, review, settings)
        detection["raw_box"] = raw.tolist()
        if local is not None:
            corrected = (
                _filter_local_measurement(
                    motion_reference,
                    local,
                    float(review.get("local_confidence") or 0.0),
                    settings.get("measurement_filter"),
                )
                if motion_reference is not None
                else local
            )
            if motion_reference is not None:
                review["local_measurement_box"] = local.tolist()
            source = (
                "local_scrfd_filtered"
                if motion_reference is not None
                and bool(settings.get("measurement_filter", {}).get("enabled", False))
                else "local_scrfd"
            )
            update_size = True
        elif motion_reference is not None:
            # The full-frame hit still provides a useful center measurement,
            # but an unconfirmed width/height must not inject scale drift.
            corrected = _box_with_center_and_size(raw, motion_reference)
            source = "detector_center_motion_size"
            update_size = False
        else:
            corrected = raw
            source = "raw_initial_detection"
            update_size = True
        detection["box"] = corrected.tolist()
        detection["geometry_source"] = source
        review["box"] = corrected.tolist()
        review["raw_box"] = raw.tolist()
        review["geometry_source"] = source
        return update_size

    def _refine_tracking_geometry(
        self,
        item: dict[str, Any],
        review: dict[str, Any],
        reference: np.ndarray,
        geometry_target: np.ndarray | None = None,
    ) -> None:
        settings = dict(self.config["revalidation"].get("geometry_refinement", {}))
        if review.get("local_review_variant") == "anchor_recovery":
            settings["max_center_distance"] = max(
                float(settings["max_center_distance"]),
                float(self.config["revalidation"]["match_max_center_distance"]),
            )
        if review.get("geometry_recovery_policy") == "reliable_pre_roll":
            recovery = settings["anchor_recovery"]
            settings["min_local_confidence"] = float(recovery["min_local_confidence"])
            settings["max_center_distance"] = float(recovery["max_center_distance"])
        local = _trusted_local_geometry(
            item["box"] if geometry_target is None else geometry_target,
            review,
            settings,
        )
        if local is None:
            return
        raw = np.asarray(item["box"], dtype=np.float64)
        local_measurement = local.copy()
        filter_settings = _measurement_filter_for_scope(settings, "tracking")
        local = _filter_local_measurement(
            raw,
            local,
            float(review.get("local_confidence") or 0.0),
            filter_settings,
            recovery=review.get("local_review_variant") == "anchor_recovery",
        )
        filter_enabled = filter_settings is not None
        if review.get("local_review_variant") == "anchor_recovery":
            source = (
                "local_scrfd_anchor_recovery_filtered" if filter_enabled else "local_scrfd_anchor_recovery"
            )
        else:
            source = "local_scrfd_tracking_filtered" if filter_enabled else "local_scrfd_tracking"
        item["raw_motion_box"] = raw.tolist()
        if filter_enabled:
            item["local_measurement_box"] = local_measurement.tolist()
        item["box"] = local.tolist()
        item["motion_box"] = local.tolist()
        item["geometry_source"] = source
        item["area_ratio"] = float(area_ratio(reference, local))
        item["center_distance"] = float(normalized_center_distance(reference, local))
        review["box"] = local.tolist()
        review["raw_box"] = raw.tolist()
        review["geometry_source"] = source

    def _recover_tracking_geometry(
        self,
        frame: np.ndarray,
        item: dict[str, Any],
        review: dict[str, Any],
        anchor: np.ndarray,
        origin: tuple[int, int],
        prefer_target_geometry: bool = False,
    ) -> np.ndarray | None:
        settings = self.config["revalidation"].get("geometry_refinement", {})
        recovery = settings.get("anchor_recovery", {})
        if not bool(recovery.get("enabled", False)):
            return None
        if _trusted_local_geometry(item["box"], review, settings) is not None:
            return None
        trigger_distance = float(settings["max_center_distance"])
        if normalized_center_distance(item["box"], anchor) <= trigger_distance:
            return None

        local_anchor = _translate_box(anchor, -origin[0], -origin[1])
        fallback = self.reviewer.local_match(
            frame,
            local_anchor,
            candidate_selection=(
                str(recovery["candidate_selection"]) if prefer_target_geometry else "confidence"
            ),
        )
        self.local_review_attempts += 1
        self.local_review_forced += 1
        if fallback.get("local_box") is not None:
            fallback["local_box"] = _translate_box(fallback["local_box"], origin[0], origin[1])
        if fallback.get("local_landmarks") is not None:
            local_landmarks = np.asarray(
                fallback["local_landmarks"],
                dtype=np.float64,
            ).reshape(-1, 2)
            local_landmarks += np.asarray(origin, dtype=np.float64)
            fallback["local_landmarks"] = local_landmarks.tolist()
        fallback_settings = dict(settings)
        fallback_settings["max_center_distance"] = max(
            float(settings["max_center_distance"]),
            float(self.config["revalidation"]["match_max_center_distance"]),
        )
        if prefer_target_geometry:
            if (
                _trusted_reliable_recovery_geometry(
                    anchor,
                    fallback,
                    fallback_settings,
                    recovery,
                )
                is None
            ):
                return None
        elif _trusted_local_geometry(anchor, fallback, fallback_settings) is None:
            return None
        if self.fast_review_mode:
            # The first Verifier score belongs to the drifted proposal that
            # triggered recovery. Pair the accepted recovery evidence with the
            # face box actually found around the trusted anchor instead.
            recovery_box = _translate_box(
                fallback["local_box"],
                -origin[0],
                -origin[1],
            )
            fallback["verifier_face_probability"] = self._verify_once(
                frame,
                item,
                recovery_box,
                cache_box=list(fallback["local_box"]),
            )
        review.update(fallback)
        review["local_review_variant"] = "anchor_recovery"
        review["local_review_reason"] = "geometry_recovery"
        review["geometry_recovery_anchor"] = anchor.tolist()
        if prefer_target_geometry:
            review["geometry_recovery_policy"] = "reliable_pre_roll"
        return anchor

    def _decode_pending(
        self, state: ObjectState, include_frame: int | None, include_box: np.ndarray | None
    ) -> tuple[dict[int, np.ndarray], dict[int, tuple[int, int, int, int]]]:
        known: dict[int, np.ndarray] = {
            int(frame_idx): np.asarray(item["motion_box"], dtype=np.float64)
            for frame_idx, item in state.pending.items()
        }
        if include_box is not None:
            assert include_frame is not None
            known[int(include_frame)] = np.asarray(include_box, dtype=np.float64)
        first = min(state.pending) if state.pending else int(include_frame or state.last_detection_frame)
        last = max(max(state.pending) if state.pending else first, int(include_frame or first))
        ordered = sorted(known)
        maximum_side = int(self.config["streaming"]["max_corridor_side_pixels"])
        expansion = float(self.config["streaming"]["corridor_expansion"])
        detector_anchors = [(state.last_detection_frame, state.last_detection_box)]
        if include_frame is not None and include_box is not None:
            detector_anchors.append((include_frame, include_box))
        recovery_distance = float(self.config["revalidation"]["geometry_refinement"]["max_center_distance"])
        crops: dict[int, tuple[int, int, int, int]] = {}
        for frame_idx in range(first, last + 1):
            nearest = min(ordered, key=lambda value: abs(value - frame_idx))
            value = known[nearest]
            anchor = min(detector_anchors, key=lambda pair: abs(pair[0] - frame_idx))[1]
            corridor_center = (
                anchor if normalized_center_distance(value, anchor) > recovery_distance else value
            )
            crops[frame_idx] = _bounded_corridor(corridor_center, expansion, maximum_side)
        return self._decode_frames(first, last, crops=crops), crops

    def _bidirectional_fusion_settings(self) -> dict[str, Any]:
        return self.config["tracking"]["kalman_optical_flow"]["bidirectional_fusion"]

    @staticmethod
    def _directional_flow_proposal(
        result: dict[str, Any],
        origin: tuple[int, int],
    ) -> dict[str, Any]:
        return {
            "box": _translate_box(result["box"], origin[0], origin[1]),
            "trusted": bool(result.get("flow_measurement_valid", False)),
            "quality": float(result.get("quality", 0.0)),
            "selected_points": int(result.get("selected", 0)),
            "inlier_points": int(result.get("inliers", 0)),
            "inlier_fraction": float(result.get("inlier_fraction", 0.0)),
            "coast": int(result.get("coast", 0)),
        }

    def _measure_consensus_target(
        self,
        frame: np.ndarray,
        *,
        frame_idx: int,
        track_id: str,
        target: np.ndarray,
        origin: tuple[int, int],
    ) -> dict[str, Any]:
        probe = {
            "track_id": track_id,
            "frame_idx": frame_idx,
            "source": "kalman_optical_flow",
            "box": target.tolist(),
        }
        return self._measure_review(
            frame,
            probe,
            origin,
            candidate_selection="target_geometry",
        )

    @staticmethod
    def _select_consensus_review(
        reviews: list[dict[str, Any]],
        forward_box: np.ndarray,
        reverse_box: np.ndarray,
        anchor_reference: np.ndarray,
    ) -> dict[str, Any]:
        usable = [item for item in reviews if item.get("local_box") is not None]
        if not usable:
            return (
                reviews[0]
                if reviews
                else {
                    "local_box": None,
                    "local_confidence": None,
                    "verifier_face_probability": None,
                }
            )

        def rank(item: dict[str, Any]) -> tuple[float, float, float, tuple[float, ...]]:
            value = np.asarray(item["local_box"], dtype=np.float64)
            agreement = max(iou(value, forward_box), iou(value, reverse_box))
            confidence = float(item.get("local_confidence") or 0.0)
            anchor_distance = normalized_center_distance(value, anchor_reference)
            # Coordinates are the final deterministic tie break; no direction
            # label or call order participates in the choice.
            coordinates = tuple(-float(component) for component in value)
            return agreement, confidence, -anchor_distance, coordinates

        return max(usable, key=rank)

    def _record_consensus_review(
        self,
        item: dict[str, Any],
        review: dict[str, Any],
    ) -> dict[str, Any]:
        evidence = {
            "track_id": item["track_id"],
            "frame_idx": int(item["frame_idx"]),
            "source": item["source"],
            "box": list(item["box"]),
            "admission_scope": item.get("admission_scope", "core"),
            **review,
        }
        self.evidence.append(evidence)
        return evidence

    def _emit_soft_consensus_sequence(
        self,
        state: ObjectState,
        *,
        decoded: dict[int, np.ndarray],
        origin: tuple[int, int],
        left_frame: int,
        right_frame: int,
        left_anchor: np.ndarray,
        right_anchor: np.ndarray,
        left_geometry_anchor: np.ndarray,
        right_geometry_anchor: np.ndarray,
        geometry_bridge_accepted: bool,
        forward_results: dict[int, dict[str, Any]],
        reverse_results: dict[int, dict[str, Any]],
        settings: dict[str, Any],
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Apply a centered local-evidence bias across one complete gap."""

        track_id = str(state.track["track_id"])
        contexts: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        for frame_idx in range(left_frame + 1, right_frame):
            fraction = (frame_idx - left_frame) / (right_frame - left_frame)
            anchor_reference = left_anchor * (1.0 - fraction) + right_anchor * fraction
            forward = self._directional_flow_proposal(forward_results[frame_idx], origin)
            reverse = self._directional_flow_proposal(reverse_results[frame_idx], origin)
            forward_box = np.asarray(forward["box"], dtype=np.float64)
            reverse_box = np.asarray(reverse["box"], dtype=np.float64)
            reviews = [
                self._measure_consensus_target(
                    decoded[frame_idx],
                    frame_idx=frame_idx,
                    track_id=track_id,
                    target=anchor_reference,
                    origin=origin,
                )
            ]
            ambiguous = bool(
                iou(forward_box, reverse_box) < float(settings["mutual_min_iou"])
                and normalized_center_distance(forward_box, reverse_box)
                > float(settings["mutual_max_center_distance"])
            )
            if ambiguous:
                unique_targets: dict[tuple[float, ...], np.ndarray] = {}
                for value in (forward_box, reverse_box):
                    unique_targets[tuple(float(component) for component in value)] = value
                for _key, target in sorted(unique_targets.items()):
                    reviews.append(
                        self._measure_consensus_target(
                            decoded[frame_idx],
                            frame_idx=frame_idx,
                            track_id=track_id,
                            target=target,
                            origin=origin,
                        )
                    )
            review = self._select_consensus_review(
                reviews,
                forward_box,
                reverse_box,
                anchor_reference,
            )
            records.append(
                {
                    "frame_idx": frame_idx,
                    "fraction": fraction,
                    "forward": forward,
                    "reverse": reverse,
                    "local_review": review,
                }
            )
            contexts.append(
                {
                    "frame_idx": frame_idx,
                    "anchor_reference": anchor_reference,
                    "forward": forward,
                    "reverse": reverse,
                    "review": review,
                }
            )

        outputs = soft_fuse_bidirectional_sequence(records, settings)
        reviewed: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for context, output in zip(contexts, outputs, strict=True):
            frame_idx = int(context["frame_idx"])
            anchor_reference = np.asarray(context["anchor_reference"], dtype=np.float64)
            forward = context["forward"]
            reverse = context["reverse"]
            review = context["review"]
            flow_seed = np.asarray(output["box"], dtype=np.float64)
            anchor_consistent = bool(
                normalized_center_distance(flow_seed, anchor_reference)
                <= float(settings["anchor_max_center_distance"])
                and area_ratio(flow_seed, anchor_reference) <= float(settings["anchor_max_area_ratio"])
            )
            if not anchor_consistent:
                self.bidirectional_rejected_frames += 1
                if self._collect_developer_audits:
                    self.bidirectional_audits.append(
                        {
                            "track_id": track_id,
                            "frame_idx": frame_idx,
                            "left_frame": left_frame,
                            "right_frame": right_frame,
                            "mode": "symmetric_local_soft",
                            "accepted": False,
                            "decision": "detector_anchor_conflict",
                            "forward_trusted": bool(forward["trusted"]),
                            "reverse_trusted": bool(reverse["trusted"]),
                            "pair_iou": iou(forward["box"], reverse["box"]),
                            "pair_center_distance": normalized_center_distance(forward["box"], reverse["box"]),
                        }
                    )
                continue
            geometry_box = (
                interpolate_published_geometry(
                    flow_seed,
                    left_geometry_anchor,
                    right_geometry_anchor,
                    float(output["fraction"]),
                )
                if geometry_bridge_accepted
                else flow_seed.copy()
            )
            local_biased = abs(float(output["raw_bias"])) > 0.0
            if local_biased:
                self.bidirectional_review_resolutions += 1
            decision_name = (
                "symmetric_local_soft"
                if local_biased or abs(float(output["bias"])) > 0.0
                else "symmetric_linear_no_local_bias"
            )
            item = {
                "frame_idx": frame_idx,
                "track_id": track_id,
                "source": "kalman_optical_flow",
                "box": geometry_box.tolist(),
                "motion_box": geometry_box.tolist(),
                "direction": 0,
                "anchor_frame": left_frame,
                "right_anchor_frame": right_frame,
                "selected_points": min(
                    int(forward["selected_points"]),
                    int(reverse["selected_points"]),
                ),
                "inlier_points": min(
                    int(forward["inlier_points"]),
                    int(reverse["inlier_points"]),
                ),
                "quality": 0.5 * (float(forward["quality"]) + float(reverse["quality"])),
                "flow_continuity": decision_name,
                "bidirectional_fusion_mode": "symmetric_local_soft",
                "bidirectional_decision": decision_name,
                "bidirectional_pair_iou": iou(forward["box"], reverse["box"]),
                "bidirectional_pair_center_distance": normalized_center_distance(
                    forward["box"], reverse["box"]
                ),
                "bidirectional_forward_trusted": bool(forward["trusted"]),
                "bidirectional_reverse_trusted": bool(reverse["trusted"]),
                "bidirectional_forward_quality": float(forward["quality"]),
                "bidirectional_reverse_quality": float(reverse["quality"]),
                "bidirectional_anchor_consistent": anchor_consistent,
                "bidirectional_reverse_weight": float(output["reverse_weight"]),
                "bidirectional_local_raw_bias": float(output["raw_bias"]),
                "bidirectional_local_smoothed_bias": float(output["bias"]),
                "bidirectional_flow_seed_box": flow_seed.tolist(),
                "bidirectional_geometry_box": geometry_box.tolist(),
                "bidirectional_geometry_bridge_accepted": bool(geometry_bridge_accepted),
                "bidirectional_size_interpolation": (
                    "endpoint_log" if geometry_bridge_accepted else "flow_seed"
                ),
                "area_ratio": float(area_ratio(anchor_reference, geometry_box)),
                "center_distance": float(normalized_center_distance(anchor_reference, geometry_box)),
            }
            evidence = self._record_consensus_review(item, review)
            self._refine_tracking_geometry(
                item,
                evidence,
                anchor_reference,
            )
            self._capture_tracking_recognition_candidate(
                decoded[frame_idx],
                item,
                evidence,
                origin,
            )
            reviewed.append((item, evidence))
            self.bidirectional_accepted_frames += 1
            if self._collect_developer_audits:
                self.bidirectional_audits.append(
                    {
                        "track_id": track_id,
                        "frame_idx": frame_idx,
                        "left_frame": left_frame,
                        "right_frame": right_frame,
                        "mode": "symmetric_local_soft",
                        "accepted": True,
                        "decision": decision_name,
                        "forward_trusted": bool(forward["trusted"]),
                        "reverse_trusted": bool(reverse["trusted"]),
                        "pair_iou": item["bidirectional_pair_iou"],
                        "pair_center_distance": item["bidirectional_pair_center_distance"],
                        "reverse_weight": float(output["reverse_weight"]),
                        "raw_bias": float(output["raw_bias"]),
                        "smoothed_bias": float(output["bias"]),
                        "flow_seed_box": flow_seed.tolist(),
                        "geometry_box": geometry_box.tolist(),
                        "geometry_bridge_accepted": bool(geometry_bridge_accepted),
                    }
                )
        return reviewed

    def _finish_pending_consensus(
        self,
        state: ObjectState,
        *,
        anchor_frame: int,
        right_consensus_anchor: np.ndarray,
        right_geometry_anchor: np.ndarray,
        right_geometry_source: str,
    ) -> bool:
        settings = self._bidirectional_fusion_settings()
        left_frame = int(state.last_detection_frame)
        right_frame = int(anchor_frame)
        if right_frame <= left_frame + 1:
            state.pending.clear()
            return True
        left_anchor = np.asarray(
            state.last_consensus_anchor_box
            if state.last_consensus_anchor_box is not None
            else state.last_detection_box,
            dtype=np.float64,
        )
        right_anchor = np.asarray(right_consensus_anchor, dtype=np.float64)
        left_geometry_anchor = np.asarray(state.last_detection_box, dtype=np.float64)
        right_geometry_anchor = np.asarray(right_geometry_anchor, dtype=np.float64)
        gap_frames = right_frame - left_frame - 1
        if gap_frames > int(settings["max_gap_frames"]):
            self.bidirectional_skipped_jobs += 1
            fallback_available = bool(state.pending)
            if self._collect_developer_audits:
                self.bidirectional_audits.append(
                    {
                        "track_id": str(state.track["track_id"]),
                        "left_frame": left_frame,
                        "right_frame": right_frame,
                        "status": ("pending_fallback" if fallback_available else "consensus_skipped"),
                        "fallback_available": fallback_available,
                        "reason": "gap_frame_limit",
                        "gap_frames": gap_frames,
                    }
                )
            return False
        corridor = _consensus_corridor(
            [left_anchor, right_anchor],
            float(settings["corridor_expansion"]),
            int(settings["max_corridor_side_pixels"]),
        )
        if corridor is None:
            self.bidirectional_skipped_jobs += 1
            fallback_available = bool(state.pending)
            if self._collect_developer_audits:
                self.bidirectional_audits.append(
                    {
                        "track_id": str(state.track["track_id"]),
                        "left_frame": left_frame,
                        "right_frame": right_frame,
                        "status": ("pending_fallback" if fallback_available else "consensus_skipped"),
                        "fallback_available": fallback_available,
                        "reason": "corridor_limit",
                        "gap_frames": gap_frames,
                    }
                )
            return False
        estimated_bytes = (corridor[2] - corridor[0]) * (corridor[3] - corridor[1]) * 3 * (gap_frames + 2)
        if estimated_bytes > int(settings["max_materialized_bytes"]):
            self.bidirectional_skipped_jobs += 1
            fallback_available = bool(state.pending)
            if self._collect_developer_audits:
                self.bidirectional_audits.append(
                    {
                        "track_id": str(state.track["track_id"]),
                        "left_frame": left_frame,
                        "right_frame": right_frame,
                        "status": ("pending_fallback" if fallback_available else "consensus_skipped"),
                        "fallback_available": fallback_available,
                        "reason": "materialized_byte_limit",
                        "gap_frames": gap_frames,
                        "estimated_bytes": int(estimated_bytes),
                    }
                )
            return False
        origin = (corridor[0], corridor[1])
        decoded = self._decode_frames(
            left_frame,
            right_frame,
            crop=corridor,
        )
        flow_settings = self.config["tracking"]["kalman_optical_flow"]
        forward_state = ROIFlowState(
            decoded[left_frame],
            np.asarray(
                _translate_box(left_anchor, -origin[0], -origin[1]),
                dtype=np.float64,
            ),
            flow_settings,
        )
        reverse_state = ROIFlowState(
            decoded[right_frame],
            np.asarray(
                _translate_box(right_anchor, -origin[0], -origin[1]),
                dtype=np.float64,
            ),
            flow_settings,
        )
        forward_results: dict[int, dict[str, Any]] = {}
        reverse_results: dict[int, dict[str, Any]] = {}
        for frame_idx in range(left_frame + 1, right_frame):
            forward_results[frame_idx] = forward_state.step(decoded[frame_idx])
        for frame_idx in range(right_frame - 1, left_frame, -1):
            reverse_results[frame_idx] = reverse_state.step(decoded[frame_idx])

        geometry_bridge = _bidirectional_geometry_bridge_decision(
            left_consensus=left_anchor,
            right_consensus=right_anchor,
            left_publication=left_geometry_anchor,
            right_publication=right_geometry_anchor,
            left_geometry_source=state.last_detection_geometry_source,
            right_geometry_source=right_geometry_source,
            forward_results=forward_results,
            reverse_results=reverse_results,
            frame_width=int(self.metadata.width),
            frame_height=int(self.metadata.height),
            settings=settings,
        )
        geometry_bridge_accepted = bool(geometry_bridge["accepted"])
        if self._collect_developer_audits:
            self.bidirectional_audits.append(
                {
                    "track_id": str(state.track["track_id"]),
                    "left_frame": left_frame,
                    "right_frame": right_frame,
                    "mode": "symmetric_local_soft",
                    "status": "geometry_bridge",
                    **geometry_bridge,
                }
            )

        reviewed = self._emit_soft_consensus_sequence(
            state,
            decoded=decoded,
            origin=origin,
            left_frame=left_frame,
            right_frame=right_frame,
            left_anchor=left_anchor,
            right_anchor=right_anchor,
            left_geometry_anchor=left_geometry_anchor,
            right_geometry_anchor=right_geometry_anchor,
            geometry_bridge_accepted=geometry_bridge_accepted,
            forward_results=forward_results,
            reverse_results=reverse_results,
            settings=settings,
        )
        self.candidates.extend(item for item, _review in reviewed)
        self.bidirectional_gap_jobs += 1
        self.bidirectional_gap_frames += max(0, right_frame - left_frame - 1)
        self.reverse_jobs += 1
        self.reverse_frames += max(0, right_frame - left_frame - 1)
        state.pending.clear()
        return True

    def _finish_pending_interpolation(
        self,
        state: ObjectState,
        *,
        anchor_frame: int,
        anchor_box: np.ndarray,
    ) -> None:
        """Publish a detector-bounded gap without decoding its frames.

        Centers move linearly between the two reviewed detector anchors while
        width and height move linearly in log space.  These observations are
        deliberately marked as reduced-assurance geometry: they extend an
        already admitted track, but do not pretend that Local SCRFD or the
        verifier ran on the interpolated frames.
        """

        left_frame = int(state.last_detection_frame)
        right_frame = int(anchor_frame)
        state.pending.clear()
        if right_frame <= left_frame + 1:
            return

        left_box = np.asarray(state.last_detection_box, dtype=np.float64)
        right_box = np.asarray(anchor_box, dtype=np.float64)
        track_id = str(state.track["track_id"])
        emitted = 0
        for frame_idx in range(left_frame + 1, right_frame):
            fraction = (frame_idx - left_frame) / (right_frame - left_frame)
            linear_box = left_box * (1.0 - fraction) + right_box * fraction
            geometry_box = clip(
                interpolate_published_geometry(
                    linear_box,
                    left_box,
                    right_box,
                    fraction,
                ),
                int(self.metadata.width),
                int(self.metadata.height),
            )
            geometry = geometry_box.tolist()
            candidate = {
                "frame_idx": frame_idx,
                "track_id": track_id,
                # Preserve the existing tracking-source contract consumed by
                # stabilization/rendering while making the policy explicit.
                "source": "kalman_optical_flow",
                "box": geometry,
                "motion_box": geometry,
                "direction": 0,
                "anchor_frame": left_frame,
                "right_anchor_frame": right_frame,
                "selected_points": 0,
                "inlier_points": 0,
                "flow_inlier_fraction": 0.0,
                "quality": 0.0,
                "flow_continuity": "detector_anchor_interpolation",
                "geometry_source": "detector_anchor_interpolation",
                "reduced_assurance": True,
                "interpolation_left_frame": left_frame,
                "interpolation_right_frame": right_frame,
                "interpolation_fraction": float(fraction),
                "area_ratio": float(area_ratio(left_box, geometry_box)),
                "center_distance": float(
                    normalized_center_distance(left_box, geometry_box)
                ),
            }
            review = self._empty_local_review()
            review.update(
                {
                    "local_review_reason": "detector_anchor_interpolation",
                    "verifier_face_probability": None,
                }
            )
            self.candidates.append(candidate)
            self.evidence.append(
                {
                    "track_id": track_id,
                    "frame_idx": frame_idx,
                    "source": candidate["source"],
                    "box": geometry,
                    "admission_scope": "core",
                    "geometry_source": candidate["geometry_source"],
                    "reduced_assurance": True,
                    "interpolation_left_frame": left_frame,
                    "interpolation_right_frame": right_frame,
                    "interpolation_fraction": float(fraction),
                    **review,
                }
            )
            emitted += 1
        self.interpolation_jobs = getattr(self, "interpolation_jobs", 0) + 1
        self.interpolated_frames = getattr(self, "interpolated_frames", 0) + emitted

    def _finish_pending(
        self,
        state: ObjectState,
        *,
        anchor_frame: int | None = None,
        anchor_box: np.ndarray | None = None,
        right_consensus_anchor: np.ndarray | None = None,
        right_geometry_source: str = "",
    ) -> None:
        if bool(getattr(self, "interpolate_tracking", False)):
            if anchor_frame is not None and anchor_box is not None:
                self._finish_pending_interpolation(
                    state,
                    anchor_frame=anchor_frame,
                    anchor_box=anchor_box,
                )
            else:
                state.pending.clear()
            return
        if (
            anchor_frame is not None
            and anchor_box is not None
            and right_consensus_anchor is not None
            and anchor_frame > state.last_detection_frame + 1
        ):
            handled = self._finish_pending_consensus(
                state,
                anchor_frame=anchor_frame,
                right_consensus_anchor=right_consensus_anchor,
                right_geometry_anchor=anchor_box,
                right_geometry_source=right_geometry_source,
            )
            if handled:
                return
        if not state.pending:
            return
        decoded, crops = self._decode_pending(state, anchor_frame, anchor_box)
        detector_anchors = [(state.last_detection_frame, state.last_detection_box)]
        if anchor_frame is not None and anchor_box is not None:
            detector_anchors.append((anchor_frame, anchor_box))
        reverse: dict[int, np.ndarray] = {}
        if anchor_frame is not None and anchor_box is not None:
            anchor_crop = crops[anchor_frame]
            origin_x, origin_y = anchor_crop[0], anchor_crop[1]
            anchor_local = np.asarray(_translate_box(anchor_box, -origin_x, -origin_y))
            reverse_state = ROIFlowState(
                decoded[anchor_frame],
                anchor_local,
                self.config["tracking"]["kalman_optical_flow"],
            )
            prior_origin = (origin_x, origin_y)
            for frame_idx in sorted(state.pending, reverse=True):
                crop = crops[frame_idx]
                origin_x, origin_y = crop[0], crop[1]
                reverse_state.rebase(prior_origin[0] - origin_x, prior_origin[1] - origin_y)
                result = reverse_state.step(decoded[frame_idx])
                reverse[frame_idx] = np.asarray(
                    _translate_box(result["box"], origin_x, origin_y), dtype=np.float64
                )
                prior_origin = (origin_x, origin_y)
            self.reverse_jobs += 1
            self.reverse_frames += len(reverse)
        first_anchor = state.last_detection_frame
        reviewed: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for frame_idx, forward in sorted(state.pending.items()):
            value = np.asarray(forward["motion_box"], dtype=np.float64)
            if frame_idx in reverse and anchor_frame is not None:
                weight = (frame_idx - first_anchor) / max(1, anchor_frame - first_anchor)
                value = value * (1.0 - weight) + reverse[frame_idx] * weight
            item = dict(forward)
            item.pop("_flow_trusted", None)
            item["motion_box"] = value.tolist()
            item["box"] = value.tolist()
            crop = crops[frame_idx]
            origin_x, origin_y = crop[0], crop[1]
            review = self._review(decoded[frame_idx], item, (origin_x, origin_y))
            recovery_anchor = min(detector_anchors, key=lambda pair: abs(pair[0] - frame_idx))[1]
            geometry_target = self._recover_tracking_geometry(
                decoded[frame_idx],
                item,
                review,
                recovery_anchor,
                (origin_x, origin_y),
            )
            self._refine_tracking_geometry(
                item,
                review,
                state.last_detection_box,
                geometry_target=geometry_target,
            )
            self._capture_tracking_recognition_candidate(
                decoded[frame_idx],
                item,
                review,
                (origin_x, origin_y),
            )
            reviewed.append((item, review))
        _stabilize_reviewed_geometry(
            reviewed,
            state.last_detection_box,
            int(self.config["tracking"]["kalman_optical_flow"]["max_coast_frames"]),
        )
        self.candidates.extend(item for item, _review in reviewed)
        state.pending.clear()

    @staticmethod
    def _promote_cycle_recovery(
        state: ObjectState,
        frame_idx: int,
        recovered_box: np.ndarray,
    ) -> None:
        """Commit a buffered coast only after the shared flow state reconnects."""

        trusted = [
            (int(index), np.asarray(item["motion_box"], dtype=np.float64))
            for index, item in state.pending.items()
            if bool(item.get("_flow_trusted", False))
        ]
        if trusted:
            trusted_frame, trusted_box = max(trusted, key=lambda pair: pair[0])
        else:
            trusted_frame = state.last_detection_frame
            trusted_box = state.last_detection_box
        span = max(1, frame_idx - trusted_frame)
        for pending_frame, item in state.pending.items():
            if pending_frame <= trusted_frame or bool(item.get("_flow_trusted", False)):
                continue
            weight = (pending_frame - trusted_frame) / span
            interpolated = trusted_box * (1.0 - weight) + recovered_box * weight
            item["box"] = interpolated.tolist()
            item["motion_box"] = interpolated.tolist()
            item["_flow_trusted"] = True
            item["flow_continuity"] = "trusted_bridge_interpolation"

    def _endpoint_affine_settings(self) -> dict[str, Any]:
        settings = self.config["tracking"]["kalman_optical_flow"].get("endpoint_affine_repair", {})
        return settings if isinstance(settings, dict) else {}

    def _run_endpoint_affine(
        self,
        *,
        track_id: str,
        anchor_frame: int,
        anchor_box: np.ndarray,
        target_frame: int,
        direction: int,
        run_neural_review: bool = True,
        sparse_neural_review: bool = False,
        emit_before: int | None = None,
        repair_reason: str = "ordinary_endpoint",
        boundary_reason: str | None = None,
        boundary_frame_exclusive: int | None = None,
    ) -> int:
        """Build isolated endpoint candidates without touching main state."""

        settings = self._endpoint_affine_settings()
        if not bool(settings.get("enabled", False)) or direction not in {
            -1,
            1,
        }:
            return 0
        if direction > 0 and target_frame <= anchor_frame:
            return 0
        if direction < 0 and target_frame >= anchor_frame:
            return 0
        first, last = sorted((anchor_frame, target_frame))
        oldest = self.cache.oldest_frame_index()
        if oldest is not None and first < oldest:
            first = oldest
            if direction < 0:
                target_frame = first
            elif anchor_frame < first:
                return 0
        corridor = _bounded_corridor(
            anchor_box,
            float(self.config["streaming"]["pre_roll_corridor_expansion"]),
            max(
                int(self.config["streaming"]["max_corridor_side_pixels"]),
                int(self._bidirectional_fusion_settings()["max_corridor_side_pixels"]),
            ),
        )
        frame_bytes = (
            max(0, corridor[2] - corridor[0]) * max(0, corridor[3] - corridor[1]) * 3
        )
        maximum_materialized_bytes = int(
            self._bidirectional_fusion_settings()["max_materialized_bytes"]
        )
        if frame_bytes <= 0 or frame_bytes > maximum_materialized_bytes:
            return 0
        materialized_frame_limit = max(
            1,
            maximum_materialized_bytes // frame_bytes,
        )
        nominal_verifier_interval = max(
            1,
            int(
                math.ceil(
                    max(float(getattr(self, "fps", 30.0)), 1.0) / _ENDPOINT_VERIFIER_HZ
                )
            ),
        )
        verifier_interval = (
            min(nominal_verifier_interval, materialized_frame_limit)
            if run_neural_review and sparse_neural_review
            else nominal_verifier_interval
        )
        configured_chunk_size = max(
            1,
            int(
                self.config["streaming"].get(
                    "pre_roll_decode_chunk_frames",
                    last - first + 1,
                )
            ),
        )
        chunk_size = max(
            1,
            min(
                configured_chunk_size,
                materialized_frame_limit,
            ),
        )
        anchor_decoded = self._decode_frames(
            anchor_frame,
            anchor_frame,
            crop=corridor,
        )
        if anchor_frame not in anchor_decoded:
            return 0
        ox, oy = corridor[0], corridor[1]
        state = AffineEndpointState(
            anchor_decoded[anchor_frame],
            np.asarray(_translate_box(anchor_box, -ox, -oy), dtype=np.float64),
            self.config["tracking"]["kalman_optical_flow"],
        )
        # AffineEndpointState retains only its normalized gray ROI. Release the
        # decoded anchor crop before materializing the first endpoint block.
        del anchor_decoded
        local_review_interval = max(
            1,
            int(
                math.ceil(
                    max(float(getattr(self, "fps", 30.0)), 1.0)
                    / _ENDPOINT_LOCAL_REVIEW_HZ
                )
            ),
        )
        rule_gate = (
            self.config.get("revalidation", {}).get("policy", {}).get("rule_gate", {})
        )
        short_track = rule_gate.get("short_track", {})
        verifier_pass = float(short_track.get("moderate_verifier_p50", 0.40))
        verifier_strong = max(
            verifier_pass,
            float(short_track.get("strong_verifier_p50", 0.80)),
        )
        endpoint_local_frames = getattr(
            self,
            "endpoint_local_review_frames",
            None,
        )
        if endpoint_local_frames is None:
            endpoint_local_frames = {}
            self.endpoint_local_review_frames = endpoint_local_frames
        track_local_frames = endpoint_local_frames.setdefault(track_id, set())
        pending_bucket: list[tuple[dict[str, Any], np.ndarray]] = []
        emitted = 0
        self.endpoint_affine_jobs += 1
        terminated = False
        prior_global_box = np.asarray(anchor_box, dtype=np.float64)

        def retain(candidate: dict[str, Any]) -> None:
            nonlocal emitted
            frame_idx = int(candidate["frame_idx"])
            if emit_before is not None and direction < 0 and frame_idx >= emit_before:
                return
            self.endpoint_affine_candidates.append(candidate)
            emitted += 1

        cursor = anchor_frame + direction
        while direction * (target_frame - cursor) >= 0 and not terminated:
            remaining = abs(target_frame - cursor) + 1
            # Historical decode scheduling and neural-review cadence are
            # independent concerns.  Materialize a normal execution block and
            # keep the sparse Verifier checkpoints inside that block.  The
            # pending prefix from a prior block still counts against the same
            # byte budget, so unusual chunk/checkpoint alignments remain
            # bounded without forcing every decode to restart at a checkpoint.
            available_materialized_frames = max(
                1,
                materialized_frame_limit - len(pending_bucket),
            )
            count = min(
                chunk_size,
                remaining,
                available_materialized_frames,
            )
            chunk_end = cursor + direction * (count - 1)
            chunk_first, chunk_last = sorted((cursor, chunk_end))
            decoded = self._decode_frames(
                chunk_first,
                chunk_last,
                crop=corridor,
            )
            if set(decoded) != set(range(chunk_first, chunk_last + 1)):
                break
            for frame_idx in range(cursor, chunk_end + direction, direction):
                image = decoded[frame_idx]
                result = state.step(image)
                if not bool(result["valid"]):
                    terminated = True
                    break
                global_box = np.asarray(
                    _translate_box(result["box"], ox, oy),
                    dtype=np.float64,
                )
                candidate = {
                    "frame_idx": frame_idx,
                    "track_id": track_id,
                    "source": "kalman_optical_flow",
                    "box": global_box.tolist(),
                    "motion_box": global_box.tolist(),
                    "direction": direction,
                    "anchor_frame": anchor_frame,
                    "selected_points": int(result.get("selected", 0)),
                    "inlier_points": int(result.get("inliers", 0)),
                    "quality": float(result.get("quality", 0.0)),
                    "flow_continuity": "affine_endpoint_repair",
                    "endpoint_repair": "affine_ransac",
                    "endpoint_repair_reason": repair_reason,
                    "endpoint_boundary_reason": boundary_reason,
                    "endpoint_boundary_frame_exclusive": (boundary_frame_exclusive),
                    "admission_scope": "reliable_endpoint_extension",
                    "reduced_assurance": (
                        not run_neural_review or sparse_neural_review
                    ),
                    "affine_scale": float(result["affine_scale"]),
                    "affine_rotation_degrees": float(result["affine_rotation_degrees"]),
                    "area_ratio": 1.0,
                    "center_distance": 0.0,
                }
                self.endpoint_affine_frames += 1
                if run_neural_review and not sparse_neural_review:
                    candidate["_review_measurement"] = self._measure_review(
                        image,
                        candidate,
                        (ox, oy),
                        local_review_reason="endpoint_affine",
                    )
                    self._collect_endpoint_conflict_evidence(candidate, image, (ox, oy))
                    retain(candidate)
                    prior_global_box = np.asarray(
                        candidate["motion_box"], dtype=np.float64
                    )
                    continue

                review = self._empty_local_review()
                review.update(
                    {
                        "local_review_reason": repair_reason,
                        "verifier_face_probability": None,
                    }
                )
                candidate["_review_measurement"] = review
                self._collect_endpoint_conflict_evidence(candidate, image, (ox, oy))
                if not run_neural_review:
                    retain(candidate)
                    prior_global_box = global_box
                    continue

                pending_bucket.append((candidate, image))
                checkpoint = bool(
                    frame_idx == target_frame
                    or abs(frame_idx - anchor_frame) % verifier_interval == 0
                )
                if not checkpoint:
                    prior_global_box = global_box
                    continue

                local_box = _translate_box(global_box, -ox, -oy)
                score = self._verify_once(
                    image,
                    candidate,
                    local_box,
                    cache_box=global_box.tolist(),
                )
                self.endpoint_verifier_checkpoints = (
                    getattr(self, "endpoint_verifier_checkpoints", 0) + 1
                )
                review["local_review_reason"] = "endpoint_verifier_checkpoint"
                review["verifier_face_probability"] = score
                candidate["_review_measurement"] = review

                weak_affine = bool(
                    float(result.get("quality", 0.0)) < 0.35
                    or int(result.get("inliers", 0))
                    < max(
                        int(
                            self.config["tracking"]["kalman_optical_flow"]["min_points"]
                        ),
                        math.ceil(0.5 * int(result.get("selected", 0))),
                    )
                )
                recognition_needs_candidate = False
                recognition_engine = getattr(self, "recognition_engine", None)
                if recognition_engine is not None and recognition_engine.enabled:
                    required_candidates = max(
                        1,
                        int(recognition_engine.max_frames_per_track),
                    )
                    candidate_frames = sorted(
                        {
                            int(value.frame_index)
                            for _identifier, value in getattr(
                                self,
                                "recognition_candidates",
                                {},
                            ).get(track_id, [])
                        }
                    )
                    independent_frames: list[int] = []
                    minimum_gap = max(
                        1,
                        int(math.ceil(float(getattr(self, "fps", 30.0)))),
                    )
                    for candidate_frame in candidate_frames:
                        if (
                            not independent_frames
                            or candidate_frame - independent_frames[-1] >= minimum_gap
                        ):
                            independent_frames.append(candidate_frame)
                    recognition_needs_candidate = (
                        len(independent_frames) < required_candidates
                    )
                local_allowed = bool(
                    not track_local_frames
                    or min(abs(frame_idx - prior) for prior in track_local_frames)
                    >= local_review_interval
                )
                if (
                    score < verifier_strong
                    or weak_affine
                    or recognition_needs_candidate
                ) and local_allowed:
                    local_reason = (
                        "endpoint_recognition_candidate"
                        if recognition_needs_candidate
                        and score >= verifier_strong
                        and not weak_affine
                        else "endpoint_affine_anomaly"
                    )
                    review = self._measure_review(
                        image,
                        candidate,
                        (ox, oy),
                        force_local=True,
                        local_review_reason=local_reason,
                    )
                    track_local_frames.add(frame_idx)
                    candidate["_review_measurement"] = review
                    self._refine_tracking_geometry(
                        candidate,
                        review,
                        prior_global_box,
                    )
                    score = float(review.get("verifier_face_probability") or 0.0)
                    corrected_box = np.asarray(
                        candidate["motion_box"], dtype=np.float64
                    )
                    if not np.allclose(corrected_box, global_box):
                        score = self._verify_once(
                            image,
                            candidate,
                            _translate_box(corrected_box, -ox, -oy),
                            cache_box=corrected_box.tolist(),
                        )
                        review["verifier_face_probability"] = score
                        state = AffineEndpointState(
                            image,
                            np.asarray(
                                _translate_box(corrected_box, -ox, -oy),
                                dtype=np.float64,
                            ),
                            self.config["tracking"]["kalman_optical_flow"],
                        )
                    self._capture_tracking_recognition_candidate(
                        image,
                        candidate,
                        review,
                        (ox, oy),
                    )

                if score >= verifier_pass:
                    for bucket_candidate, _bucket_image in pending_bucket:
                        retain(bucket_candidate)
                    pending_bucket.clear()
                    prior_global_box = np.asarray(
                        candidate["motion_box"], dtype=np.float64
                    )
                    continue

                # The checkpoint failed. Resolve only this short bucket at
                # frame granularity so a quarter-second sample does not move
                # the visible endpoint by the whole checkpoint interval.
                for bucket_candidate, bucket_image in pending_bucket:
                    if bucket_candidate is candidate:
                        bucket_score = score
                    else:
                        bucket_box = np.asarray(
                            bucket_candidate["motion_box"],
                            dtype=np.float64,
                        )
                        bucket_score = self._verify_once(
                            bucket_image,
                            bucket_candidate,
                            _translate_box(bucket_box, -ox, -oy),
                            cache_box=bucket_box.tolist(),
                        )
                        bucket_review = bucket_candidate["_review_measurement"]
                        bucket_review["local_review_reason"] = (
                            "endpoint_verifier_boundary_refinement"
                        )
                        bucket_review["verifier_face_probability"] = bucket_score
                        self.endpoint_verifier_refinement_frames = (
                            getattr(self, "endpoint_verifier_refinement_frames", 0) + 1
                        )
                    if bucket_score < verifier_pass:
                        terminated = True
                        break
                    retain(bucket_candidate)
                pending_bucket.clear()
                prior_global_box = np.asarray(candidate["motion_box"], dtype=np.float64)
                if terminated:
                    break
            del decoded
            cursor = chunk_end + direction
        if pending_bucket and run_neural_review and sparse_neural_review:
            # Affine may fail between scheduled checkpoints. Validate the last
            # valid geometry instead of discarding the entire still-unchecked
            # bucket merely because the following frame had insufficient LK
            # support.
            checkpoint_candidate, checkpoint_image = pending_bucket[-1]
            checkpoint_box = np.asarray(
                checkpoint_candidate["motion_box"],
                dtype=np.float64,
            )
            checkpoint_score = self._verify_once(
                checkpoint_image,
                checkpoint_candidate,
                _translate_box(checkpoint_box, -ox, -oy),
                cache_box=checkpoint_box.tolist(),
            )
            checkpoint_review = checkpoint_candidate["_review_measurement"]
            checkpoint_review["local_review_reason"] = "endpoint_verifier_affine_stop"
            checkpoint_review["verifier_face_probability"] = checkpoint_score
            self.endpoint_verifier_checkpoints = (
                getattr(self, "endpoint_verifier_checkpoints", 0) + 1
            )
            if checkpoint_score >= verifier_pass:
                for bucket_candidate, _bucket_image in pending_bucket:
                    retain(bucket_candidate)
            else:
                for bucket_candidate, bucket_image in pending_bucket:
                    if bucket_candidate is checkpoint_candidate:
                        bucket_score = checkpoint_score
                    else:
                        bucket_box = np.asarray(
                            bucket_candidate["motion_box"],
                            dtype=np.float64,
                        )
                        bucket_score = self._verify_once(
                            bucket_image,
                            bucket_candidate,
                            _translate_box(bucket_box, -ox, -oy),
                            cache_box=bucket_box.tolist(),
                        )
                        bucket_review = bucket_candidate["_review_measurement"]
                        bucket_review["local_review_reason"] = (
                            "endpoint_verifier_boundary_refinement"
                        )
                        bucket_review["verifier_face_probability"] = bucket_score
                        self.endpoint_verifier_refinement_frames = (
                            getattr(self, "endpoint_verifier_refinement_frames", 0) + 1
                        )
                    if bucket_score < verifier_pass:
                        break
                    retain(bucket_candidate)
            pending_bucket.clear()
        return emitted

    def _repair_interpolate_endpoint(
        self,
        state: ObjectState,
        *,
        direction: int = 1,
        extension: int | None = None,
        emit_before: int | None = None,
        boundary_frame_exclusive: int | None = None,
        close_reason: str = "natural",
    ) -> int:
        """Complete either unanchored endpoint with one symmetric policy.

        Detector-bounded gaps remain pure interpolation. Outside the first and
        last detector anchors, partial-affine motion supplies geometry and a
        sparse Verifier schedule commits short buckets. Final core admission
        still decides later whether any isolated endpoint may be published.
        """

        if not bool(getattr(self, "interpolate_tracking", False)) or direction not in {
            -1,
            1,
        }:
            return 0
        reliable = len(state.track["detections"]) >= int(
            self.settings["reliable_endpoint_min_detector_frames"]
        )
        if extension is None:
            extension = int(
                self.settings[
                    "reliable_endpoint_extension" if reliable else "endpoint_extension"
                ]
            )
        extension = max(0, int(extension))
        if direction < 0:
            first_detection = state.track["detections"][0]
            anchor_frame = int(first_detection["frame_idx"])
            anchor_box = np.asarray(first_detection["box"], dtype=np.float64)
            target_frame = max(0, anchor_frame - extension)
            oldest = self.cache.oldest_frame_index()
            if oldest is not None:
                target_frame = max(target_frame, int(oldest))
            prior_cuts = [
                int(item["frame_idx"])
                for item in self.audits
                if bool(item["scene_cut_from_previous"])
                and target_frame <= int(item["frame_idx"]) <= anchor_frame
            ]
            if prior_cuts:
                target_frame = max(target_frame, max(prior_cuts))
            boundary: int | None = None
        else:
            anchor_frame = int(state.last_detection_frame)
            anchor_box = np.asarray(state.last_detection_box, dtype=np.float64)
            if boundary_frame_exclusive is None:
                return 0
            boundary = int(boundary_frame_exclusive)
            if boundary <= anchor_frame + 1:
                return 0
            target_frame = min(anchor_frame + extension, boundary - 1)
        if target_frame == anchor_frame:
            return 0
        started = time.perf_counter()
        emitted = self._run_endpoint_affine(
            track_id=str(state.track["track_id"]),
            anchor_frame=anchor_frame,
            anchor_box=anchor_box,
            target_frame=target_frame,
            direction=direction,
            run_neural_review=True,
            sparse_neural_review=True,
            emit_before=emit_before,
            repair_reason="interpolate_unanchored_endpoint",
            boundary_reason=close_reason,
            boundary_frame_exclusive=boundary,
        )
        self.interpolate_endpoint_seconds = (
            getattr(self, "interpolate_endpoint_seconds", 0.0)
            + time.perf_counter()
            - started
        )
        if emitted:
            self.interpolate_endpoint_jobs = (
                getattr(self, "interpolate_endpoint_jobs", 0) + 1
            )
            self.interpolate_endpoint_frames = (
                getattr(self, "interpolate_endpoint_frames", 0) + emitted
            )
            reason_counts = getattr(
                self,
                "interpolate_endpoint_reason_counts",
                None,
            )
            if reason_counts is None:
                reason_counts = Counter()
                self.interpolate_endpoint_reason_counts = reason_counts
            reason_counts[close_reason] += 1
        return emitted

    def _repair_backward_endpoint(
        self,
        state: ObjectState,
        frame_idx: int,
        box: np.ndarray,
        *,
        extension: int,
    ) -> int:
        settings = self._endpoint_affine_settings()
        maximum = min(extension, int(settings.get("max_frames", 0)))
        if maximum <= 0 or frame_idx <= 0:
            return 0
        first = max(0, frame_idx - maximum)
        prior_cuts = [
            int(item["frame_idx"])
            for item in self.audits
            if bool(item["scene_cut_from_previous"]) and first <= int(item["frame_idx"]) <= frame_idx
        ]
        if prior_cuts:
            first = max(first, max(prior_cuts))
        by_frame = {
            int(item["frame_idx"]): item
            for item in self.candidates
            if str(item["track_id"]) == str(state.track["track_id"])
            and first <= int(item["frame_idx"]) < frame_idx
        }
        anchor = frame_idx
        anchor_value = np.asarray(box, dtype=np.float64)
        while anchor - 1 in by_frame:
            anchor -= 1
            anchor_value = np.asarray(by_frame[anchor]["motion_box"], dtype=np.float64)
        if anchor <= first:
            return 0
        return self._run_endpoint_affine(
            track_id=str(state.track["track_id"]),
            anchor_frame=anchor,
            anchor_box=anchor_value,
            target_frame=first,
            direction=-1,
        )

    def _repair_forward_endpoint(
        self,
        state: ObjectState,
        *,
        extension: int,
        reason: str,
    ) -> int:
        settings = self._endpoint_affine_settings()
        maximum = min(extension, int(settings.get("max_frames", 0)))
        if maximum <= 0 or reason == "scene_cut":
            return 0
        # Endpoint repair can only consume frames that have actually been
        # decoded and committed.  Container metadata is advisory (some MP4s
        # over-report their frame count), so never let it point the packet
        # cache at a future or non-existent frame.
        last_committed_frame = (
            int(self.audits[-1]["frame_idx"])
            if self.audits
            else int(state.last_detection_frame)
        )
        last = min(
            state.last_detection_frame + maximum,
            int(self.metadata.frame_count) - 1,
            last_committed_frame,
        )
        anchor = state.last_detection_frame
        anchor_value = np.asarray(state.last_detection_box, dtype=np.float64)
        while anchor + 1 in state.pending and bool(state.pending[anchor + 1].get("_flow_trusted", False)):
            anchor += 1
            anchor_value = np.asarray(state.pending[anchor]["motion_box"], dtype=np.float64)
        if anchor >= last:
            return 0
        return self._run_endpoint_affine(
            track_id=str(state.track["track_id"]),
            anchor_frame=anchor,
            anchor_box=anchor_value,
            target_frame=last,
            direction=1,
        )

    def _pre_roll(
        self,
        state: ObjectState,
        frame_idx: int,
        box: np.ndarray,
        *,
        extension: int,
        emit_before: int | None = None,
    ) -> int:
        first = max(0, frame_idx - extension)
        oldest_cached = self.cache.oldest_frame_index()
        if oldest_cached is not None:
            first = max(first, oldest_cached)
        prior_cuts = [
            int(item["frame_idx"])
            for item in self.audits
            if bool(item["scene_cut_from_previous"]) and first <= int(item["frame_idx"]) <= frame_idx
        ]
        if prior_cuts:
            first = max(first, max(prior_cuts))
        if first >= frame_idx:
            return 0
        corridor = _bounded_corridor(
            box,
            float(self.config["streaming"]["pre_roll_corridor_expansion"]),
            int(self.config["streaming"]["max_corridor_side_pixels"]),
        )
        ox, oy = corridor[0], corridor[1]
        flow_settings = self.config["tracking"]["kalman_optical_flow"]
        require_cycle = bool(flow_settings.get("require_cycle_consistency_after_coast", False))
        emitted = 0
        maximum_coast = int(flow_settings["max_coast_frames"])
        unresolved: list[tuple[int, dict[str, Any], np.ndarray]] = []
        trusted_frame = frame_idx
        trusted_global_box = np.asarray(box, dtype=np.float64)

        def emit_candidate(
            prior: int,
            result: dict[str, Any],
            global_box: np.ndarray,
            image: np.ndarray,
            *,
            continuity: str,
        ) -> None:
            nonlocal emitted
            # A reliable expansion replays the short pre-roll to rebuild the
            # optical-flow state, but emits only the newly exposed frames.
            if emit_before is not None and prior >= emit_before:
                return
            reliable_expansion = emit_before is not None
            item = {
                "frame_idx": prior,
                "track_id": state.track["track_id"],
                "source": "kalman_optical_flow",
                "box": global_box.tolist(),
                "motion_box": global_box.tolist(),
                "direction": -1,
                "anchor_frame": frame_idx,
                "selected_points": int(result.get("selected", 0)),
                "inlier_points": int(result.get("inliers", 0)),
                "quality": float(result.get("quality", 0.0)),
                "flow_continuity": continuity,
                "area_ratio": 1.0,
                "center_distance": 0.0,
                # This delayed endpoint is granted only after the core track
                # has enough detector anchors. It may extend a continuous
                # accepted interval, but must not dilute core admission.
                "admission_scope": ("reliable_endpoint_extension" if reliable_expansion else "core"),
            }
            review = self._review(
                image,
                item,
                (ox, oy),
                candidate_selection=(
                    str(
                        self.config["revalidation"]["geometry_refinement"]["anchor_recovery"][
                            "candidate_selection"
                        ]
                    )
                    if reliable_expansion
                    else "confidence"
                ),
            )
            geometry_target = self._recover_tracking_geometry(
                image,
                item,
                review,
                box,
                (ox, oy),
                prefer_target_geometry=reliable_expansion,
            )
            self._refine_tracking_geometry(
                item,
                review,
                box,
                geometry_target=geometry_target,
            )
            self._capture_tracking_recognition_candidate(
                image,
                item,
                review,
                (ox, oy),
            )
            self.candidates.append(item)
            self.reverse_frames += 1
            emitted += 1

        chunk_size = int(self.config["streaming"].get("pre_roll_decode_chunk_frames", frame_idx - first + 1))
        chunk_last = frame_idx
        reverse_state: ROIFlowState | None = None
        stopped = False
        while chunk_last >= first and not stopped:
            chunk_first = max(first, chunk_last - chunk_size + 1)
            frames = self._decode_frames(chunk_first, chunk_last, crop=corridor)
            if reverse_state is None:
                reverse_state = ROIFlowState(
                    frames[frame_idx],
                    np.asarray(_translate_box(box, -ox, -oy)),
                    flow_settings,
                )
            for prior in range(min(frame_idx - 1, chunk_last), chunk_first - 1, -1):
                image = frames[prior]
                result = reverse_state.step(image)
                global_box = np.asarray(_translate_box(result["box"], ox, oy), dtype=np.float64)
                if not require_cycle:
                    if not result["valid"] and int(result["coast"]) > int(flow_settings["max_coast_frames"]):
                        stopped = True
                        break
                    emit_candidate(
                        prior,
                        result,
                        global_box,
                        image,
                        continuity=("direct" if result["valid"] else "coast_prediction"),
                    )
                    continue

                if not result["valid"]:
                    unresolved.append((prior, result, image))
                    if len(unresolved) > maximum_coast:
                        # None of these predictions is connected to the last
                        # trusted face ROI. Do not publish them or reseed LK on
                        # unrelated texture in an older block.
                        stopped = True
                        break
                    continue

                recovered = bool(unresolved)
                if recovered:
                    span = max(1, trusted_frame - prior)
                    for missing_frame, missing_result, missing_image in unresolved:
                        weight = (trusted_frame - missing_frame) / span
                        interpolated = trusted_global_box * (1.0 - weight) + global_box * weight
                        emit_candidate(
                            missing_frame,
                            missing_result,
                            interpolated,
                            missing_image,
                            continuity="trusted_bridge_interpolation",
                        )
                    unresolved.clear()
                emit_candidate(
                    prior,
                    result,
                    global_box,
                    image,
                    continuity=("trusted_cycle_recovery" if recovered else "direct"),
                )
                trusted_frame = prior
                trusted_global_box = global_box
            chunk_last = chunk_first - 1
        self.reverse_jobs += 1
        return emitted

    def _expand_reliable_pre_roll(self, state: ObjectState) -> None:
        required = int(self.settings["reliable_endpoint_min_detector_frames"])
        extension = int(
            self.settings.get(
                "reliable_pre_roll_extension",
                self.settings["reliable_endpoint_extension"],
            )
        )
        if len(state.track["detections"]) < required:
            return
        if state.pre_roll_extension >= extension:
            return
        first_detection = state.track["detections"][0]
        frame_idx = int(first_detection["frame_idx"])
        emit_before = max(0, frame_idx - state.pre_roll_extension)
        if bool(getattr(self, "interpolate_tracking", False)):
            self._repair_interpolate_endpoint(
                state,
                direction=-1,
                extension=extension,
                emit_before=emit_before,
                close_reason="reliable_pre_roll",
            )
        else:
            self._pre_roll(
                state,
                frame_idx,
                np.asarray(first_detection["box"], dtype=np.float64),
                extension=extension,
                emit_before=emit_before,
            )
        state.pre_roll_extension = extension

    def _new_state(
        self,
        frame_idx: int,
        frame: np.ndarray,
        detection: dict[str, Any],
        *,
        starts_at_scene_cut: bool = False,
    ) -> ObjectState:
        track_id = f"t{len(self.tracks):05d}"
        detection["track_id"] = track_id
        review = self._review(
            frame,
            detection,
            force_local=self.fast_review_mode,
            local_review_reason="new_track_anchor",
        )
        self._refine_detection_geometry(detection, review, None)
        consensus_anchor = np.asarray(
            detection.pop("_consensus_anchor_box", detection["box"]),
            dtype=np.float64,
        )
        track = {
            "track_id": track_id,
            "detections": [detection],
            "scene_segment_id": int(getattr(self, "scene_segment_id", 0)),
            "starts_at_scene_cut": bool(starts_at_scene_cut),
            "ends_at_scene_cut": False,
            "start_scene_cut_frame": (frame_idx if starts_at_scene_cut else None),
            "end_scene_cut_frame": None,
            "close_reason": None,
            "close_boundary_frame_exclusive": None,
        }
        state = ObjectState(
            track,
            ROIFlowState(frame, detection["box"], self.config["tracking"]["kalman_optical_flow"]),
            frame_idx,
            np.asarray(detection["box"], dtype=np.float64),
            last_detection_scan_rank=(
                int(detection["detector_scan_rank"])
                if detection.get("detector_scan_rank") is not None
                else None
            ),
            last_consensus_anchor_box=consensus_anchor,
            last_detection_geometry_source=str(detection.get("geometry_source", "")),
        )
        self.tracks.append(track)
        self.states.append(state)
        initial_extension = int(self.settings["endpoint_extension"])
        if bool(getattr(self, "interpolate_tracking", False)):
            self._repair_interpolate_endpoint(
                state,
                direction=-1,
                extension=initial_extension,
                close_reason="initial_pre_roll",
            )
            state.pre_roll_extension = initial_extension
        else:
            self._pre_roll(
                state,
                frame_idx,
                state.last_detection_box,
                extension=initial_extension,
            )
            self._repair_backward_endpoint(
                state,
                frame_idx,
                state.last_detection_box,
                extension=initial_extension,
            )
            state.pre_roll_extension = initial_extension
        return state

    def _close(
        self,
        state: ObjectState,
        *,
        reason: str = "natural",
        boundary_frame: int | None = None,
    ) -> None:
        if reason not in {"natural", "scene_cut", "end_of_stream"}:
            raise ValueError(f"unsupported track close reason: {reason}")
        if reason == "scene_cut" and boundary_frame is None:
            raise ValueError("scene-cut closure requires its boundary frame")
        boundary_frame_exclusive = (
            int(boundary_frame)
            if boundary_frame is not None
            else int(self.audits[-1]["frame_idx"]) + 1
            if self.audits
            else int(state.last_detection_frame) + 1
        )
        if bool(getattr(self, "interpolate_tracking", False)):
            self._repair_interpolate_endpoint(
                state,
                boundary_frame_exclusive=boundary_frame_exclusive,
                close_reason=reason,
            )
            discarded = list(state.pending)
            state.pending.clear()
            self.discarded_unanchored_tail_frames += len(discarded)
        else:
            extension = int(
                self.settings["reliable_endpoint_extension"]
                if len(state.track["detections"])
                >= int(self.settings["reliable_endpoint_min_detector_frames"])
                else self.settings["endpoint_extension"]
            )
            self._repair_forward_endpoint(
                state,
                extension=extension,
                reason=reason,
            )
            cutoff = state.last_detection_frame + extension
            discarded = [
                frame_idx
                for frame_idx, item in state.pending.items()
                if frame_idx > cutoff or not bool(item.get("_flow_trusted", False))
            ]
            for frame_idx in discarded:
                state.pending.pop(frame_idx, None)
            self.discarded_unanchored_tail_frames += len(discarded)
        self._finish_pending(state)
        state.track["close_reason"] = reason
        state.track["close_boundary_frame_exclusive"] = boundary_frame_exclusive
        state.track["ends_at_scene_cut"] = reason == "scene_cut"
        state.track["end_scene_cut_frame"] = int(boundary_frame) if reason == "scene_cut" else None
        state.active = False

    def _publish_endpoint_affine_candidates(
        self,
        observations: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Attach isolated repairs only to already accepted endpoint paths."""

        track_map = {
            str(track["track_id"]): track for track in self.tracks if bool(track.get("accepted", False))
        }
        boxes = {
            (str(item["track_id"]), int(item["frame_idx"])): np.asarray(item["box"], dtype=np.float64)
            for item in observations
            if str(item["track_id"]) in track_map
        }
        unique: dict[tuple[str, int, int], dict[str, Any]] = {}
        for item in self.endpoint_affine_candidates:
            track_id = str(item["track_id"])
            if track_id not in track_map:
                continue
            key = (
                track_id,
                int(item["frame_idx"]),
                int(item["direction"]),
            )
            prior = unique.get(key)
            if prior is None or (
                int(item.get("inlier_points", 0)),
                float(item.get("quality", 0.0)),
            ) > (
                int(prior.get("inlier_points", 0)),
                float(prior.get("quality", 0.0)),
            ):
                unique[key] = item

        duplicate_endpoints = self._resolve_endpoint_conflicts(
            observations, evidence, list(unique.values())
        )
        continuity = self.config["revalidation"]["policy"]["continuity"]
        maximum_center = float(continuity["segment_max_center_jump"])
        maximum_area = float(continuity["segment_max_area_ratio"])
        published: list[dict[str, Any]] = []
        groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        for item in unique.values():
            groups.setdefault(
                (
                    str(item["track_id"]),
                    int(item["direction"]),
                    int(item["anchor_frame"]),
                ),
                [],
            ).append(item)
        for (track_id, direction, _anchor), values in sorted(groups.items()):
            ordered = sorted(
                values,
                key=lambda item: int(item["frame_idx"]),
                reverse=direction < 0,
            )
            for item in ordered:
                frame_idx = int(item["frame_idx"])
                key = (track_id, frame_idx)
                if key in boxes:
                    continue
                neighbor = (track_id, frame_idx - direction)
                if neighbor not in boxes:
                    break
                value = np.asarray(item["motion_box"], dtype=np.float64)
                reference = boxes[neighbor]
                if (
                    normalized_center_distance(reference, value) > maximum_center
                    or area_ratio(reference, value) > maximum_area
                ):
                    break
                # A confirmed duplicate still supplies geometric continuity
                # within this isolated repair. Do not let a suppressed frame
                # truncate later, independently supported endpoint frames.
                boxes[key] = value
                if (track_id, frame_idx, direction) in duplicate_endpoints:
                    collector = getattr(self, "endpoint_conflicts", None)
                    if collector is not None:
                        collector.stats["suppressed_frames"] += 1
                        coverage = collector.coverage_references.get((track_id, frame_idx, direction))
                        if coverage is not None:
                            # Admission is finished. Preserve the actual face
                            # localization through smoothing and final dedup,
                            # without treating the drifting endpoint box as
                            # an additional face-sized coverage requirement.
                            evidence.append(dict(coverage))
                        evidence.extend(
                            dict(reference)
                            for reference in collector.additional_coverage_references.get((track_id, frame_idx, direction), [])
                        )
                    continue
                final_item = dict(item)
                endpoint_review = final_item.pop(
                    "_review_measurement",
                    self._empty_local_review(),
                )
                final_item["box"] = value.tolist()
                final_item["motion_box"] = value.tolist()
                final_item.update(endpoint_review)
                observations.append(final_item)
                evidence.append(
                    {
                        "track_id": track_id,
                        "frame_idx": frame_idx,
                        "source": "kalman_optical_flow",
                        "box": value.tolist(),
                        "admission_scope": "reliable_endpoint_extension",
                        "endpoint_repair": str(
                            final_item.get("endpoint_repair", "affine_ransac")
                        ),
                        "endpoint_repair_reason": final_item.get(
                            "endpoint_repair_reason"
                        ),
                        "endpoint_boundary_reason": final_item.get(
                            "endpoint_boundary_reason"
                        ),
                        "endpoint_boundary_frame_exclusive": final_item.get(
                            "endpoint_boundary_frame_exclusive"
                        ),
                        "reduced_assurance": bool(
                            final_item.get("reduced_assurance", False)
                        ),
                        **endpoint_review,
                    }
                )
                boxes[key] = value
                published.append(final_item)
                if (
                    final_item.get("endpoint_repair_reason")
                    == "interpolate_unanchored_endpoint"
                ):
                    self.interpolate_endpoint_published_frames = (
                        getattr(self, "interpolate_endpoint_published_frames", 0)
                        + 1
                    )

                track = track_map[track_id]
                intervals = [list(interval) for interval in track.get("accepted_intervals", [])]
                intervals.append([frame_idx, frame_idx])
                merged: list[list[int]] = []
                for start, end in sorted(intervals):
                    if merged and int(start) <= merged[-1][1] + 1:
                        merged[-1][1] = max(merged[-1][1], int(end))
                    else:
                        merged.append([int(start), int(end)])
                track["accepted_intervals"] = merged
        self.endpoint_affine_published_frames += len(published)
        evidence.sort(key=lambda item: (str(item["track_id"]), int(item["frame_idx"])))
        return sorted(
            observations,
            key=lambda item: (int(item["frame_idx"]), str(item["track_id"])),
        )

    def _bidirectional_anchor_association_score(
        self,
        state: ObjectState,
        detection: dict[str, Any],
        frame_idx: int,
    ) -> float | None:
        """Retrospectively confirm a split association from the new endpoint."""

        settings = self._bidirectional_fusion_settings()
        rescue = settings.get("association_rescue", {})
        if not isinstance(rescue, dict) or not bool(rescue.get("enabled", False)):
            return None
        left_frame = int(state.last_detection_frame)
        gap_frames = frame_idx - left_frame - 1
        if gap_frames < 1 or gap_frames > int(settings["max_gap_frames"]):
            return None
        left_anchor = np.asarray(
            state.last_consensus_anchor_box
            if state.last_consensus_anchor_box is not None
            else state.last_detection_box,
            dtype=np.float64,
        )
        right_anchor = np.asarray(
            detection.get("_consensus_anchor_box", detection["box"]),
            dtype=np.float64,
        )
        if area_ratio(left_anchor, right_anchor) > float(rescue["max_area_ratio"]):
            return None
        endpoint_distance = normalized_center_distance(left_anchor, right_anchor)
        if endpoint_distance / max(1, frame_idx - left_frame) > float(rescue["max_endpoint_center_speed"]):
            return None
        corridor = _consensus_corridor(
            [left_anchor, right_anchor],
            float(settings["corridor_expansion"]),
            int(settings["max_corridor_side_pixels"]),
        )
        if corridor is None:
            return None
        estimated_bytes = (corridor[2] - corridor[0]) * (corridor[3] - corridor[1]) * 3 * (gap_frames + 2)
        if estimated_bytes > int(settings["max_materialized_bytes"]):
            return None

        self.bidirectional_association_attempts += 1
        origin = (corridor[0], corridor[1])
        decoded = self._decode_frames(
            left_frame,
            frame_idx,
            crop=corridor,
        )
        flow_settings = self.config["tracking"]["kalman_optical_flow"]
        forward_state = ROIFlowState(
            decoded[left_frame],
            np.asarray(
                _translate_box(left_anchor, -origin[0], -origin[1]),
                dtype=np.float64,
            ),
            flow_settings,
        )
        reverse_state = ROIFlowState(
            decoded[frame_idx],
            np.asarray(
                _translate_box(right_anchor, -origin[0], -origin[1]),
                dtype=np.float64,
            ),
            flow_settings,
        )
        forward_result: dict[str, Any] | None = None
        for following in range(left_frame + 1, frame_idx + 1):
            forward_result = forward_state.step(decoded[following])
        reverse_result: dict[str, Any] | None = None
        for prior in range(frame_idx - 1, left_frame - 1, -1):
            reverse_result = reverse_state.step(decoded[prior])
        assert forward_result is not None and reverse_result is not None
        forward_box = np.asarray(
            _translate_box(forward_result["box"], origin[0], origin[1]),
            dtype=np.float64,
        )
        reverse_box = np.asarray(
            _translate_box(reverse_result["box"], origin[0], origin[1]),
            dtype=np.float64,
        )
        forward_overlap = iou(forward_box, right_anchor)
        forward_distance = normalized_center_distance(forward_box, right_anchor)
        reverse_overlap = iou(reverse_box, left_anchor)
        reverse_distance = normalized_center_distance(reverse_box, left_anchor)

        forward_metrics = {
            "trusted": bool(forward_result.get("flow_measurement_valid", False)),
            "area_ratio": float(area_ratio(forward_box, right_anchor)),
            "iou": float(forward_overlap),
            "center_distance": float(forward_distance),
        }
        reverse_metrics = {
            "trusted": bool(reverse_result.get("flow_measurement_valid", False)),
            "area_ratio": float(area_ratio(reverse_box, left_anchor)),
            "iou": float(reverse_overlap),
            "center_distance": float(reverse_distance),
        }
        confirmation = _association_rescue_decision(
            forward_metrics,
            reverse_metrics,
            rescue,
        )
        accepted = bool(confirmation["accepted"])
        forward_confirmed = bool(
            confirmation["forward_strong_iou"] or confirmation["forward_center_confirmed"]
        )
        reverse_confirmed = bool(
            confirmation["reverse_strong_iou"] or confirmation["reverse_center_confirmed"]
        )
        if self.fast_review_mode and not (forward_confirmed and reverse_confirmed):
            accepted = False
            confirmation["confirmation_basis"] = "rejected_bilateral_required"
        best_overlap = max(forward_overlap, reverse_overlap)
        best_distance = min(forward_distance, reverse_distance)
        if self._collect_developer_audits:
            self.bidirectional_audits.append(
                {
                    "track_id": str(state.track["track_id"]),
                    "frame_idx": int(frame_idx),
                    "left_frame": left_frame,
                    "right_frame": int(frame_idx),
                    "mode": "symmetric_local_soft",
                    "status": "association_rescue",
                    "accepted": accepted,
                    "confirmation_rule": (
                        "bilateral_endpoint_confirmation"
                        if self.fast_review_mode
                        else "strong_iou_or_bilateral_center"
                    ),
                    "confirmation_basis": str(confirmation["confirmation_basis"]),
                    "decision": (
                        "bidirectional_endpoint_confirmed" if accepted else "bidirectional_endpoint_rejected"
                    ),
                    "forward_endpoint_confirmed": forward_confirmed,
                    "reverse_endpoint_confirmed": reverse_confirmed,
                    "forward_endpoint_area_passed": bool(confirmation["forward_area_passed"]),
                    "reverse_endpoint_area_passed": bool(confirmation["reverse_area_passed"]),
                    "forward_endpoint_strong_iou": bool(confirmation["forward_strong_iou"]),
                    "reverse_endpoint_strong_iou": bool(confirmation["reverse_strong_iou"]),
                    "forward_endpoint_center_confirmed": bool(confirmation["forward_center_confirmed"]),
                    "reverse_endpoint_center_confirmed": bool(confirmation["reverse_center_confirmed"]),
                    "forward_endpoint_iou": float(forward_overlap),
                    "reverse_endpoint_iou": float(reverse_overlap),
                    "forward_endpoint_area_ratio": float(forward_metrics["area_ratio"]),
                    "reverse_endpoint_area_ratio": float(reverse_metrics["area_ratio"]),
                    "forward_endpoint_center_distance": float(forward_distance),
                    "reverse_endpoint_center_distance": float(reverse_distance),
                    "forward_endpoint_flow_trusted": bool(forward_result.get("flow_measurement_valid", False)),
                    "reverse_endpoint_flow_trusted": bool(reverse_result.get("flow_measurement_valid", False)),
                    "endpoint_center_speed": float(endpoint_distance / max(1, frame_idx - left_frame)),
                }
            )
        if not accepted:
            return None
        # Keep rescue below a valid ordinary association score.  It competes
        # only among otherwise-unassigned states and detections.
        return 1.0 + best_overlap + max(0.0, 1.0 - best_distance)

    def _attach_detection(
        self,
        state: ObjectState,
        detection: dict[str, Any],
        *,
        frame_idx: int,
        frame: np.ndarray,
    ) -> None:
        long_gap = _is_long_association_gap(
            state,
            detection,
            frame_idx,
            self.settings,
        )
        if long_gap:
            self.long_gap_reanchors += 1
        detection["track_id"] = state.track["track_id"]
        force_anchor_local = bool(
            self.interpolate_tracking or (self.fast_review_mode and long_gap)
        )
        review = self._review(
            frame,
            detection,
            force_local=force_anchor_local,
            local_review_reason=(
                "interpolation_anchor"
                if self.interpolate_tracking
                else "long_gap_anchor"
                if long_gap
                else None
            ),
        )
        update_size = self._refine_detection_geometry(
            detection,
            review,
            state.last_detection_box if self.interpolate_tracking else state.flow.box,
        )
        state.pending.pop(frame_idx, None)
        current_box = np.asarray(detection["box"], dtype=np.float64)
        right_consensus_anchor = np.asarray(
            detection.get("_consensus_anchor_box", current_box),
            dtype=np.float64,
        )
        self._finish_pending(
            state,
            anchor_frame=frame_idx,
            anchor_box=current_box,
            right_consensus_anchor=right_consensus_anchor,
            right_geometry_source=str(detection.get("geometry_source", "")),
        )
        detection.pop("_consensus_anchor_box", None)
        state.track["detections"].append(detection)
        self._expand_reliable_pre_roll(state)
        if not self.interpolate_tracking:
            state.flow.correct(frame, current_box, update_size=update_size)
        state.last_detection_frame = frame_idx
        state.last_detection_scan_rank = (
            int(detection["detector_scan_rank"])
            if detection.get("detector_scan_rank") is not None
            else None
        )
        state.last_detection_box = current_box
        state.last_consensus_anchor_box = right_consensus_anchor
        state.last_detection_geometry_source = str(detection.get("geometry_source", ""))

    def process(
        self, frame_idx: int, frame: np.ndarray, detections: list[dict[str, Any]], scene_cut: bool
    ) -> None:
        if scene_cut:
            for state in self.states:
                if state.active:
                    self._close(
                        state,
                        reason="scene_cut",
                        boundary_frame=frame_idx,
                    )
            self.scene_segment_id = int(getattr(self, "scene_segment_id", 0)) + 1
        active = [state for state in self.states if state.active]
        if not self.interpolate_tracking:
            for state in active:
                result = state.flow.step(frame)
                maximum_coast = int(
                    self.config["tracking"]["kalman_optical_flow"]["max_coast_frames"]
                )
                if result["valid"]:
                    if bool(result.get("recovered_from_coast", False)):
                        self._promote_cycle_recovery(
                            state,
                            frame_idx,
                            np.asarray(result["box"], dtype=np.float64),
                        )
                    state.pending[frame_idx] = _candidate(state, frame_idx, result)
                elif int(result["coast"]) <= maximum_coast:
                    state.pending[frame_idx] = _candidate(state, frame_idx, result)

        # Use the same review scheduler before association and cache its full
        # result. Full mode runs Local SCRFD here; sampled modes normally use
        # the complementary local-review phase and force only safety anchors.
        for detection in detections:
            measurement = self._measure_review(
                frame,
                detection,
                force_local=self.interpolate_tracking,
                local_review_reason=(
                    "interpolation_anchor" if self.interpolate_tracking else None
                ),
            )
            if self.recognition_engine.enabled:
                detection["_recognition_local_review"] = {
                    "local_box": measurement.get("local_box"),
                    "local_landmarks": measurement.get("local_landmarks"),
                    "local_confidence": measurement.get("local_confidence"),
                    "local_angle": measurement.get("local_angle"),
                }
            detection["_review_measurement"] = measurement
            self._refine_detection_geometry(detection, measurement, None)
            # Capture the local-review-refined detector geometry before a
            # causal motion filter can pull it toward the online forward
            # state.  The two replay endpoints must be direction neutral.
            detection["_consensus_anchor_box"] = list(detection["box"])

        pairs: list[tuple[float, int, int]] = []
        for state_index, state in enumerate(active):
            for detection_index, detection in enumerate(detections):
                score = _association_score(
                    state,
                    detection,
                    frame_idx,
                    self.settings,
                    reference_box=(
                        state.last_detection_box
                        if self.interpolate_tracking
                        else None
                    ),
                    allow_long_gap_flow=not self.interpolate_tracking,
                )
                if score is not None:
                    pairs.append((score, state_index, detection_index))
        assigned_states: set[int] = set()
        assigned_detections: set[int] = set()
        for _score, state_index, detection_index in sorted(pairs, reverse=True):
            if state_index in assigned_states or detection_index in assigned_detections:
                continue
            state, detection = active[state_index], detections[detection_index]
            self._attach_detection(
                state,
                detection,
                frame_idx=frame_idx,
                frame=frame,
            )
            assigned_states.add(state_index)
            assigned_detections.add(detection_index)
        if not self.interpolate_tracking:
            rescue_pairs: list[tuple[float, int, int]] = []
            for state_index, state in enumerate(active):
                if state_index in assigned_states:
                    continue
                for detection_index, detection in enumerate(detections):
                    if detection_index in assigned_detections:
                        continue
                    score = self._bidirectional_anchor_association_score(
                        state,
                        detection,
                        frame_idx,
                    )
                    if score is not None:
                        rescue_pairs.append((score, state_index, detection_index))
            for _score, state_index, detection_index in sorted(
                rescue_pairs,
                reverse=True,
            ):
                if state_index in assigned_states or detection_index in assigned_detections:
                    continue
                self._attach_detection(
                    active[state_index],
                    detections[detection_index],
                    frame_idx=frame_idx,
                    frame=frame,
                )
                self.bidirectional_association_rescues += 1
                assigned_states.add(state_index)
                assigned_detections.add(detection_index)
        for index, detection in enumerate(detections):
            if index not in assigned_detections:
                self._new_state(
                    frame_idx,
                    frame,
                    detection,
                    starts_at_scene_cut=scene_cut,
                )
        for state in active:
            if state.active and frame_idx - state.last_detection_frame >= int(
                self.settings["max_missed_frames"]
            ):
                self._close(state)

    def run(
        self,
        *,
        progress: Callable[[int, int, str], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        pending: deque[list[Any]] = deque()
        finalized_audits: dict[int, dict[str, Any]] = {}
        configured_depth = max(1, int(self.config["scan"].get("pipeline_depth", 2)))
        # Sampling must not accidentally starve the existing scan worker pool:
        # keep approximately the same number of analyzed frames in flight,
        # while capping the additional decoded-frame memory on large strides.
        depth = _detector_pipeline_depth(
            configured_depth,
            self.detector_frame_stride,
            int(self.metadata.width) * int(self.metadata.height) * 3,
            int(self.config["streaming"].get("recent_frame_cache_max_bytes", 0)),
        )
        processed = 0
        detector_scan_reasons: Counter[str] = Counter()
        last_progress: tuple[int, int] | None = None

        def report_progress(current: int) -> None:
            nonlocal last_progress
            if progress is None:
                return
            update = (int(current), int(self.metadata.frame_count))
            if update == last_progress:
                return
            progress(update[0], update[1], "analysis")
            last_progress = update

        def submit_newly_forced_frames() -> None:
            for queued in pending:
                reason = self._detector_scan_reason(int(queued[0]))
                if reason is None:
                    continue
                queued[3] = reason
                if queued[2] is None:
                    queued[2] = self.scanner.submit(queued[1])

        def commit(entry: list[Any]) -> None:
            nonlocal processed
            _raise_if_cancelled(is_cancelled)
            frame_idx, frame, futures, scan_reason = entry
            audit = finalized_audits.pop(frame_idx)
            if not bool(audit.get("scene_cut_finalized", False)):
                raise RuntimeError(f"scene-cut audit {frame_idx} was not finalized")
            scene_cut = bool(audit["scene_cut_from_previous"])
            if self.detector_frame_stride > 1 and scene_cut:
                self._force_detector_scan_range(frame_idx, frame_idx, "scene_cut")
                self._force_detector_scan_range(
                    frame_idx + 1,
                    frame_idx + self.detector_scan_burst_frames - 1,
                    "scene_cut_burst",
                )
                submit_newly_forced_frames()
            resolved_scan_reason = self._detector_scan_reason(frame_idx)
            if resolved_scan_reason is not None:
                scan_reason = resolved_scan_reason
                if futures is None:
                    futures = self.scanner.submit(frame)
            detector_scan_performed = futures is not None
            audit["detector_scan_performed"] = detector_scan_performed
            audit["detector_scan_reason"] = scan_reason
            self.audits.append(audit)
            detector_scan_rank: int | None = None
            if detector_scan_performed:
                self.detector_scan_opportunities = (
                    getattr(self, "detector_scan_opportunities", 0) + 1
                )
                detector_scan_rank = self.detector_scan_opportunities
            selected = self.scanner.finish(futures) if futures is not None else []
            for local_index, detection in enumerate(selected):
                detection["frame_idx"] = frame_idx
                detection["detection_id"] = f"d{frame_idx:06d}_{local_index:03d}"
                detection["source"] = "detector"
                detection["detector_scan_rank"] = detector_scan_rank
                self.detections.append(detection)
            collector = getattr(self, "endpoint_conflicts", None)
            if collector is not None and detector_scan_performed:
                collector.register_scan(frame_idx, selected)
            self._remember_frame(frame_idx, frame)
            prior_track_count = len(self.tracks)
            self.process(
                frame_idx,
                frame,
                selected,
                scene_cut,
            )
            if self.detector_frame_stride > 1:
                if len(self.tracks) > prior_track_count:
                    self._force_detector_scan_range(
                        frame_idx + 1,
                        frame_idx + self.detector_scan_burst_frames - 1,
                        "new_track_burst",
                    )
                submit_newly_forced_frames()
            for detection in selected:
                self._capture_recognition_candidate(frame, detection)
            detector_scan_reasons[scan_reason or "sampled_out"] += 1
            self.forced_detector_scan_reasons.pop(frame_idx, None)
            processed += 1
            report_progress(processed)
            _raise_if_cancelled(is_cancelled)
            if (
                logger.isEnabledFor(logging.DEBUG)
                and frame_idx
                and frame_idx % int(self.config["streaming"]["progress_every_frames"]) == 0
            ):
                logger.debug(
                    "[stream/onnx] %s/%s frames detections=%s tracks=%s live_cache=%.1fMiB",
                    frame_idx,
                    self.metadata.frame_count,
                    len(self.detections),
                    len(self.tracks),
                    self.cache.live_payload_bytes() / 1048576,
                )
            if frame_idx % int(self.config["streaming"]["eviction_interval_frames"]) == 0:
                self.cache.evict_before_frame(frame_idx - self.max_retroactive_frames)
                if collector is not None and collector.enabled:
                    collector.prune_before(frame_idx - self.max_retroactive_frames)

        try:
            _raise_if_cancelled(is_cancelled)
            report_progress(0)
            _raise_if_cancelled(is_cancelled)
            for frame_idx, timestamp, frame in iter_cached_frames(self.source, self.cache):
                _raise_if_cancelled(is_cancelled)
                for audit in self.scene_cut_detector.observe(
                    frame_idx,
                    frame,
                    float(timestamp),
                ):
                    finalized_audits[int(audit["frame_idx"])] = audit
                scan_reason = self._detector_scan_reason(frame_idx)
                futures = self.scanner.submit(frame) if scan_reason is not None else None
                pending.append([frame_idx, frame, futures, scan_reason])
                if len(pending) >= depth and pending[0][0] in finalized_audits:
                    commit(pending.popleft())
            decoded_frame_count = processed + len(pending)
            if decoded_frame_count <= 0:
                raise RuntimeError("video decoder produced no frames")
            reported_frame_count = int(self.metadata.frame_count)
            if decoded_frame_count != reported_frame_count:
                self.metadata = replace(
                    self.metadata,
                    frame_count=decoded_frame_count,
                    duration=decoded_frame_count / float(self.metadata.fps),
                )
                logger.debug(
                    "[stream/onnx] decoded %s frames; container metadata reported %s; "
                    "using the decoded frame count",
                    decoded_frame_count,
                    reported_frame_count,
                )
                # Re-evaluate queued work after normalizing the end frame.  In
                # sampled modes this submits the real final frame as the
                # existing end_of_stream detector scan even when it is off the
                # regular stride phase.
                submit_newly_forced_frames()
            for audit in self.scene_cut_detector.flush():
                finalized_audits[int(audit["frame_idx"])] = audit
            while pending:
                _raise_if_cancelled(is_cancelled)
                if pending[0][0] not in finalized_audits:
                    raise RuntimeError(f"scene-cut audit {pending[0][0]} was not finalized at EOF")
                commit(pending.popleft())
            for state in self.states:
                _raise_if_cancelled(is_cancelled)
                if state.active:
                    self._close(state, reason="end_of_stream")
            report_progress(processed)
        finally:
            self.scanner.close()
            self.cache.commit()
        if processed != self.metadata.frame_count:
            raise RuntimeError(f"stream processed {processed}, expected {self.metadata.frame_count}")
        _raise_if_cancelled(is_cancelled)
        for track in self.tracks:
            track["first_frame"] = int(track["detections"][0]["frame_idx"])
            track["last_frame"] = int(track["detections"][-1]["frame_idx"])
            track["detector_observations"] = len(track["detections"])
        aliases = _fragment_aliases(
            self.tracks,
            self.evidence,
            self.config["tracking"].get("fragment_stitching"),
        )
        if any(track_id != canonical for track_id, canonical in aliases.items()):
            fragment_settings = self.config["tracking"].get("fragment_stitching", {})
            self.tracks, self.evidence = _apply_fragment_aliases(
                self.tracks,
                self.detections,
                self.candidates,
                self.evidence,
                aliases,
                resolve_duplicate_candidates=bool(
                    fragment_settings.get(
                        "resolve_duplicate_candidates_before_stabilization",
                        True,
                    )
                ),
            )
            for track in self.tracks:
                track["first_frame"] = int(track["detections"][0]["frame_idx"])
                track["last_frame"] = int(track["detections"][-1]["frame_idx"])
                track["detector_observations"] = len(track["detections"])
        for item in self.endpoint_affine_candidates:
            item["track_id"] = aliases.get(str(item["track_id"]), str(item["track_id"]))
        self._endpoint_conflict_aliases = aliases
        published = _select_track_frame_candidates(self.candidates)
        scan = {
            "metadata": self.metadata.to_dict(),
            "frame_count": processed,
            "frames": self.audits,
            "detections": self.detections,
        }
        tracking = {"observations": published}
        review = finalize_precomputed(
            scan,
            self.tracks,
            tracking,
            self.evidence,
            self.config,
            detector_frame_stride=self.detector_frame_stride,
        )
        _raise_if_cancelled(is_cancelled)
        # Endpoint candidates are isolated until core admission is final. Once
        # accepted, attach both directions before recognition and smoothing so
        # their sparse Local-SCRFD landmarks may participate in the existing
        # candidate path and their geometry receives the same output filter.
        review["observations"] = self._publish_endpoint_affine_candidates(
            review["observations"], review["evidence"]
        )
        recognition = self._finalize_recognition(aliases)
        _raise_if_cancelled(is_cancelled)
        review["observations"] = stabilize_observations(
            review["observations"],
            self.config["render"].get("box_stabilization"),
            scene_cut_frames={
                int(item["frame_idx"]) for item in self.audits if bool(item["scene_cut_from_previous"])
            },
            scene_mean_absdiff_by_frame={
                int(item["frame_idx"]): float(item["scene_mean_absdiff"]) for item in self.audits
            },
        )
        recognition_mode = str(
            self.config.get("recognition", {}).get("mode", "all")
        )
        review["observations"] = _deduplicate_final_cross_tracks(
            review["observations"],
            review["evidence"],
            recognition_mode=recognition_mode,
        )
        allow_cross_track_coverage = recognition_mode == "all"
        (
            accepted_interval_frames,
            accepted_interval_hole_frames,
            accepted_interval_cross_track_coverage_frames,
        ) = _accepted_interval_coverage(
            self.tracks,
            review["observations"],
            review["evidence"],
            allow_cross_track_coverage=allow_cross_track_coverage,
        )
        if accepted_interval_hole_frames:
            raise RuntimeError(
                "final geometry coverage invariant failed: "
                f"{accepted_interval_hole_frames} accepted-interval track frames "
                "have no safe render geometry"
            )
        tracking["observations"].extend(
            item
            for item in review["observations"]
            if item.get("endpoint_repair") == "affine_ransac"
        )
        tracking["observations"] = sorted(
            tracking["observations"],
            key=lambda item: (int(item["frame_idx"]), str(item["track_id"])),
        )
        analyzed_frame_indices = [
            int(item["frame_idx"])
            for item in self.audits
            if bool(item.get("detector_scan_performed", False))
        ]
        maximum_consecutive_skipped_frames = max(
            (
                following - prior - 1
                for prior, following in zip(
                    analyzed_frame_indices,
                    analyzed_frame_indices[1:],
                )
            ),
            default=0,
        )
        analyzed_frames = len(analyzed_frame_indices)
        regular_scan_frames = detector_scan_reasons["every_frame"] + detector_scan_reasons[
            "regular_stride"
        ]
        detector_sampling = {
            "source_fps": self.fps,
            "max_analysis_fps": self.max_analysis_fps,
            "tolerance_multiplier": self.analysis_fps_tolerance_multiplier,
            "allowed_regular_analysis_fps": self.allowed_regular_analysis_fps,
            "effective_frame_stride": self.detector_frame_stride,
            "nominal_regular_analysis_fps": self.nominal_regular_analysis_fps,
            # Retain the established result field for consumers while making
            # its runtime-derived nature explicit in the field above.
            "frame_stride": self.detector_frame_stride,
            "policy": (
                "every_frame"
                if self.detector_frame_stride == 1
                else "fixed_stride_adaptive_burst_v1"
            ),
            "reduced_assurance": self.detector_frame_stride > 1,
            "analyzed_frames": analyzed_frames,
            "regular_scan_frames": regular_scan_frames,
            "forced_scan_frames": analyzed_frames - regular_scan_frames,
            "skipped_scan_frames": processed - analyzed_frames,
            "effective_scan_fraction": analyzed_frames / max(1, processed),
            "maximum_consecutive_skipped_frames": maximum_consecutive_skipped_frames,
            "accepted_interval_frames": accepted_interval_frames,
            "accepted_interval_hole_frames": accepted_interval_hole_frames,
            "accepted_interval_cross_track_coverage_frames": (
                accepted_interval_cross_track_coverage_frames
            ),
            "effective_pipeline_depth": depth,
            "reason_counts": dict(sorted(detector_scan_reasons.items())),
        }
        interpolation_enabled = bool(
            getattr(self, "interpolate_tracking", False)
        )
        between_scan_frames = str(
            getattr(
                self,
                "between_scan_frames",
                self.config.get("tracking", {}).get(
                    "between_scan_frames",
                    "interpolate",
                ),
            )
        )
        local_review_sampling = {
            "mode": (
                "anchor_only"
                if interpolation_enabled
                else "sampled"
                if self.fast_review_mode
                else "full"
            ),
            "stride": self.local_review_stride,
            "phase": self.local_review_phase,
            "between_scan_frames": between_scan_frames,
            "effective_between_scan_frames": (
                "interpolate" if interpolation_enabled else "visual"
            ),
            "interpolation_jobs": getattr(self, "interpolation_jobs", 0),
            "interpolated_frames": getattr(self, "interpolated_frames", 0),
            "reduced_assurance": interpolation_enabled,
            "attempts": self.local_review_attempts,
            "sampled_out": self.local_review_sampled_out,
            "forced_attempts": self.local_review_forced,
            "verifier_calls": self.verifier_review_calls,
            "verifier_cache_hits": self.verifier_review_cache_hits,
            "independent_detection_session": (
                self.reviewer.detector is not self.detector
            ),
            "same_detection_model_file": (
                getattr(self.reviewer.detector, "model_file", None)
                == getattr(self.detector, "model_file", None)
            ),
        }
        return {
            "scan": scan,
            "tracks": self.tracks,
            "tracking": tracking,
            "review": review,
            "recognition": recognition,
            "detector_sampling": detector_sampling,
            "local_review_sampling": local_review_sampling,
            "endpoint_conflicts": (
                self.endpoint_conflicts.summary()
                if getattr(self, "endpoint_conflicts", None) is not None else {"enabled": False}
            ),
            "endpoint_conflict_decisions": (
                self.endpoint_conflicts.decisions
                if getattr(self, "endpoint_conflicts", None) is not None else []
            ),
            "seconds": time.perf_counter() - started,
            "cache": {
                "live_payload_bytes": self.cache.live_payload_bytes(),
                "evicted_packets": self.cache.evicted_packets,
                "evicted_bytes": self.cache.evicted_bytes,
                "historical_decode_requests": self.cache.historical_decode_requests,
                "historical_packets_read": self.cache.historical_packets_read,
                "peak_decode_range_bytes": self.cache.peak_decode_range_bytes,
                "recent_frame_hits": self.decoded_frame_store.hits,
                "recent_frame_count": self.decoded_frame_store.frame_count,
                "recent_frame_bytes": self.decoded_frame_store.live_bytes,
                "peak_recent_frame_bytes": self.decoded_frame_store.peak_bytes,
                "recent_frame_target_frames": self.decoded_frame_store.frame_target,
                "recent_frame_peak_limit_bytes": int(
                    self.decoded_frame_store.byte_capacity
                ),
            },
            "reverse_jobs": self.reverse_jobs,
            "reverse_frames": self.reverse_frames,
            "bidirectional_gap_jobs": self.bidirectional_gap_jobs,
            "bidirectional_gap_frames": self.bidirectional_gap_frames,
            "bidirectional_accepted_frames": self.bidirectional_accepted_frames,
            "bidirectional_rejected_frames": self.bidirectional_rejected_frames,
            "bidirectional_review_resolutions": self.bidirectional_review_resolutions,
            "bidirectional_skipped_jobs": self.bidirectional_skipped_jobs,
            "bidirectional_association_attempts": self.bidirectional_association_attempts,
            "bidirectional_association_rescues": self.bidirectional_association_rescues,
            "bidirectional_audits": self.bidirectional_audits,
            "long_gap_reanchors": self.long_gap_reanchors,
            "discarded_unanchored_tail_frames": self.discarded_unanchored_tail_frames,
            "endpoint_affine_jobs": self.endpoint_affine_jobs,
            "endpoint_affine_frames": self.endpoint_affine_frames,
            "endpoint_affine_published_frames": self.endpoint_affine_published_frames,
            "endpoint_verifier_checkpoints": getattr(
                self,
                "endpoint_verifier_checkpoints",
                0,
            ),
            "endpoint_verifier_refinement_frames": getattr(
                self,
                "endpoint_verifier_refinement_frames",
                0,
            ),
            "endpoint_local_review_frames": sum(
                len(values)
                for values in getattr(
                    self,
                    "endpoint_local_review_frames",
                    {},
                ).values()
            ),
            "interpolate_endpoint_jobs": getattr(
                self,
                "interpolate_endpoint_jobs",
                0,
            ),
            "interpolate_endpoint_frames": getattr(
                self,
                "interpolate_endpoint_frames",
                0,
            ),
            "interpolate_endpoint_published_frames": getattr(
                self,
                "interpolate_endpoint_published_frames",
                0,
            ),
            "interpolate_endpoint_seconds": getattr(
                self,
                "interpolate_endpoint_seconds",
                0.0,
            ),
            "interpolate_endpoint_reason_counts": dict(
                sorted(
                    getattr(
                        self,
                        "interpolate_endpoint_reason_counts",
                        {},
                    ).items()
                )
            ),
            "fragment_stitches": sum(
                max(0, len(track.get("stitched_track_ids", [])) - 1) for track in self.tracks
            ),
        }

    def close(self) -> None:
        self.cache.close()


def run_stream(
    source: Path,
    workdir: Path,
    config: dict[str, Any],
    detector: Any | None = None,
    *,
    face_analysis: Any | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    _raise_if_cancelled(is_cancelled)
    shared_analysis = face_analysis or make_face_analysis(
        config,
    )
    _raise_if_cancelled(is_cancelled)
    shared_detector = shared_analysis.models[DETECTION_TASK]
    if detector is not None and shared_detector is not detector:
        raise ValueError("injected detector does not match FaceAnalysis detection task model")
    engine = StreamingEngine(
        source,
        workdir,
        config,
        detector,
        face_analysis=shared_analysis,
    )
    try:
        callbacks = {}
        if progress is not None:
            callbacks["progress"] = progress
        if is_cancelled is not None:
            callbacks["is_cancelled"] = is_cancelled
        return engine.run(**callbacks)
    finally:
        engine._clear_recognition_candidates()
        engine.close()


__all__ = ["ObjectState", "StreamingEngine", "run_stream"]
