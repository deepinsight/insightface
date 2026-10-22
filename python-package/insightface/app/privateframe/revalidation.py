"""Local SCRFD review plus per-candidate face-verifier evidence."""

from __future__ import annotations

import logging
import math
from bisect import bisect_right
from collections.abc import Sequence
from typing import Any

import numpy as np

from .geometry import (
    area_ratio,
    containment,
    continuity_segments,
    inverse_cardinal_points,
    iou,
    median,
    nms_records,
    normalized_center_distance,
)
from .model_catalog import DETECTION_TASK, VERIFICATION_TASK
from .models import (
    detect_faces,
    detector_max_detections,
    make_face_analysis,
    padded_square_crop,
    rotate_image,
)

logger = logging.getLogger(__name__)


def _inverse_box(value: np.ndarray, angle: int, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = (float(item) for item in value)
    if angle == 0:
        return np.asarray([x1, y1, x2, y2])
    if angle == 90:
        return np.asarray([width - y2, x1, width - y1, x2])
    if angle == -90:
        return np.asarray([y1, height - x2, y2, height - x1])
    raise ValueError(f"unsupported local angle {angle}")


def _crop(frame: np.ndarray, target: list[float], expansion: float) -> tuple[np.ndarray, int, int]:
    return padded_square_crop(frame, target, expansion, minimum_side=8)


def _edge_fallback_shift(
    frame: np.ndarray,
    crop: np.ndarray,
    origin_x: int,
    origin_y: int,
    shift_fraction: float,
) -> tuple[float, float] | None:
    """Return one deterministic crop shift toward the overflowing image edge."""

    side = int(crop.shape[0])
    shift_x = 0.0
    shift_y = 0.0
    if origin_x < 0:
        shift_x = -shift_fraction * side
    elif origin_x + side > frame.shape[1]:
        shift_x = shift_fraction * side
    if origin_y < 0:
        shift_y = -shift_fraction * side
    elif origin_y + side > frame.shape[0]:
        shift_y = shift_fraction * side
    return (shift_x, shift_y) if shift_x or shift_y else None


class LocalReviewer:
    def __init__(
        self,
        config: dict[str, Any],
        detector: Any | None = None,
        verifier: Any | None = None,
        *,
        face_analysis: Any | None = None,
    ):
        settings = config["revalidation"]
        self.settings = settings
        self.region_scan_settings = dict(config.get("scan", {}))
        self.max_detections = detector_max_detections(config)
        self.face_analysis = face_analysis or make_face_analysis(
            config,
        )
        self.detector = self.face_analysis.models[DETECTION_TASK]
        # A local-review host may intentionally contain only detection.  Keep
        # that expensive SCRFD Session independent while reusing the verifier
        # from the primary FaceAnalysis host when it is supplied explicitly.
        self.verifier = (
            verifier
            if verifier is not None
            else self.face_analysis.models[VERIFICATION_TASK]
        )
        if detector is not None and self.detector is not detector:
            raise ValueError("injected detector does not match FaceAnalysis detection task model")

    def detect_region(
        self,
        image: np.ndarray,
        *,
        origin: tuple[int, int] = (0, 0),
        input_size: int | None = None,
        max_calls: int | None = None,
        nms_iou: float | None = None,
        angles: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Detect every face in an explicitly supplied shared image region.

        ``origin`` maps the region to the complete oriented video frame. No
        target-specific crop, geometric filter, or best-face selection is
        applied. ``angles`` selects this call's complete detection round;
        omitting it uses the original local-review angles without changing
        their configuration. Each selected angle consumes one invocation;
        partial angle coverage is explicitly marked incomplete. Detector
        exceptions propagate, so failure cannot masquerade as an empty region.
        """

        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or not image.shape[0] or not image.shape[1]:
            raise ValueError("region image must be a nonempty BGR image")
        if len(origin) != 2 or any(not math.isfinite(float(value)) for value in origin):
            raise ValueError("region origin must contain two finite coordinates")
        if max_calls is not None and (isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 0):
            raise ValueError("max_calls must be a nonnegative integer or None")
        size = self.settings["input_size"] if input_size is None else input_size
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("region input_size must be a positive integer")
        requested_angles = self.settings["angles"] if angles is None else angles
        if (not isinstance(requested_angles, Sequence)
                or isinstance(requested_angles, (str, bytes))
                or not requested_angles):
            raise ValueError("region angles must be a nonempty sequence of integers")
        if any(isinstance(angle, bool) or not isinstance(angle, (int, np.integer))
               or angle not in (0, 90, -90) for angle in requested_angles):
            raise ValueError("region angles must contain only integer values 0, 90, or -90")
        selected_angles = [int(angle) for angle in requested_angles]
        if len(set(selected_angles)) != len(selected_angles):
            raise ValueError("region angles must not contain duplicates")
        overlap_threshold = self.region_scan_settings["global_nms_iou"] if nms_iou is None else nms_iou
        containment_threshold = float(self.region_scan_settings["containment_threshold"])
        if not 0.0 < float(overlap_threshold) <= 1.0:
            raise ValueError("region nms_iou must be in (0, 1]")
        calls = 0
        candidates = []
        ox, oy = (float(value) for value in origin)
        for angle in selected_angles:
            if max_calls is not None and calls >= max_calls:
                break
            rotated = rotate_image(image, angle)
            calls += 1
            found = detect_faces(
                self.face_analysis,
                rotated,
                input_sizes=[size],
                confidence_threshold=float(self.settings["confidence_threshold"]),
                max_detections=self.max_detections,
            )
            for raw in found:
                value = _inverse_box(np.asarray(raw["box"]), angle, image.shape[1], image.shape[0])
                value += np.asarray([ox, oy, ox, oy])
                confidence = float(raw["confidence"])
                if not np.all(np.isfinite(value)) or value[2] <= value[0] or value[3] <= value[1] or not math.isfinite(confidence):
                    raise ValueError("region detector returned invalid face geometry or confidence")
                candidate = {"box": value.tolist(), "confidence": confidence, "angle": angle}
                if raw.get("landmarks") is not None:
                    landmarks = inverse_cardinal_points(raw["landmarks"], angle, image.shape[1], image.shape[0])
                    landmarks += np.asarray([ox, oy], dtype=np.float64)
                    if not np.all(np.isfinite(landmarks)):
                        raise ValueError("region detector returned invalid landmarks")
                    candidate["landmarks"] = landmarks.tolist()
                candidates.append(candidate)
        # Use the established full-scan suppression policy for duplicate views.
        # The stronger independent-localization equivalence threshold belongs
        # to endpoint matching, not to multi-angle detector NMS.
        selected = nms_records(candidates, float(overlap_threshold), containment_threshold)
        selected.sort(key=lambda item: (tuple(item["box"]), -item["confidence"], item["angle"]))
        for index, item in enumerate(selected):
            item["detection_id"] = f"region_face_{index:03d}"
        return {
            "detections": selected,
            "roi": [ox, oy, ox + image.shape[1], oy + image.shape[0]],
            "detector_calls": calls,
            "complete": calls == len(selected_angles),
            "input_size": size,
            "angles": selected_angles[:calls],
        }

    def _match_crop(
        self,
        crop: np.ndarray,
        origin_x: int,
        origin_y: int,
        target: list[float],
        input_size: int,
    ) -> list[dict[str, Any]]:
        candidates = []
        for angle in self.settings["angles"]:
            rotated = rotate_image(crop, int(angle))
            for raw in detect_faces(
                self.face_analysis,
                rotated,
                input_sizes=[input_size],
                confidence_threshold=float(self.settings["confidence_threshold"]),
                max_detections=self.max_detections,
            ):
                value = _inverse_box(np.asarray(raw["box"]), int(angle), crop.shape[1], crop.shape[0])
                value += np.asarray([origin_x, origin_y, origin_x, origin_y])
                ratio = area_ratio(value, target)
                overlap, inside = iou(value, target), containment(value, target)
                distance = normalized_center_distance(value, target)
                if (
                    ratio <= float(self.settings["match_max_area_ratio"])
                    and distance <= float(self.settings["match_max_center_distance"])
                    and (
                        overlap >= float(self.settings["match_min_iou"])
                        or inside >= float(self.settings["match_min_containment"])
                    )
                ):
                    candidate = {
                        "box": value.tolist(),
                        "confidence": float(raw["confidence"]),
                        "angle": int(angle),
                        "iou": overlap,
                        "containment": inside,
                        "center_distance": distance,
                    }
                    if raw.get("landmarks") is not None:
                        landmarks = inverse_cardinal_points(
                            raw["landmarks"],
                            int(angle),
                            crop.shape[1],
                            crop.shape[0],
                        )
                        landmarks += np.asarray([origin_x, origin_y], dtype=np.float64)
                        candidate["landmarks"] = landmarks.tolist()
                    candidates.append(candidate)
        return candidates

    def verify(
        self,
        frame: np.ndarray,
        boxes: list[list[float]],
    ) -> list[dict[str, Any]]:
        """Score detector proposals through the shared task-aware model host."""

        return self.verifier.verify(frame, boxes)

    def local_match(
        self,
        frame: np.ndarray,
        target: list[float],
        *,
        candidate_selection: str = "confidence",
    ) -> dict[str, Any]:
        explicit_passes = self.settings.get("passes")
        passes = explicit_passes or [
            {
                "name": "default",
                "input_size": int(self.settings["input_size"]),
                "crop_expansion": float(self.settings["crop_expansion"]),
            }
        ]
        candidates: list[dict[str, Any]] = []
        variant = "center"
        edge_shift: list[float] | None = None
        selected_pass: dict[str, Any] | None = None
        attempted_passes: list[str] = []
        fallback = self.settings.get("edge_fallback", {})
        for detector_pass in passes:
            expansion = float(detector_pass["crop_expansion"])
            input_size = int(detector_pass["input_size"])
            attempted_passes.append(str(detector_pass["name"]))
            crop, origin_x, origin_y = _crop(frame, target, expansion)
            candidates = self._match_crop(crop, origin_x, origin_y, target, input_size)
            variant = "center"
            edge_shift = None
            if not candidates and bool(fallback.get("enabled", False)):
                shift = _edge_fallback_shift(
                    frame,
                    crop,
                    origin_x,
                    origin_y,
                    float(fallback["shift_fraction"]),
                )
                if shift is not None:
                    shifted_target = np.asarray(target, dtype=np.float64) + np.asarray(
                        [shift[0], shift[1], shift[0], shift[1]],
                        dtype=np.float64,
                    )
                    shifted_crop, shifted_x, shifted_y = _crop(frame, shifted_target.tolist(), expansion)
                    candidates = self._match_crop(
                        shifted_crop,
                        shifted_x,
                        shifted_y,
                        target,
                        input_size,
                    )
                    variant = "edge_fallback"
                    edge_shift = [float(shift[0]), float(shift[1])]
            if candidates:
                selected_pass = detector_pass
                break
        if not candidates:
            result = {
                "local_match_count": 0,
                "local_confidence": None,
                "local_box": None,
                "local_landmarks": None,
                "local_review_variant": variant,
                "local_edge_shift": edge_shift,
            }
            if explicit_passes:
                result["local_scale_pass"] = None
                result["local_attempted_passes"] = attempted_passes
            return result
        if candidate_selection == "confidence":
            best = max(
                candidates,
                key=lambda item: (
                    item["confidence"],
                    item["iou"],
                    item["containment"],
                ),
            )
        elif candidate_selection == "target_geometry":
            best = max(
                candidates,
                key=lambda item: (
                    item["iou"],
                    item["containment"],
                    -item["center_distance"],
                    item["confidence"],
                ),
            )
        else:
            raise ValueError(f"unsupported local candidate selection: {candidate_selection}")
        result = {
            "local_match_count": len(candidates),
            "local_confidence": best["confidence"],
            "local_box": best["box"],
            "local_landmarks": best.get("landmarks"),
            "local_angle": best["angle"],
            "local_review_variant": variant,
            "local_edge_shift": edge_shift,
        }
        if explicit_passes:
            assert selected_pass is not None
            result["local_scale_pass"] = str(selected_pass["name"])
            result["local_input_size"] = int(selected_pass["input_size"])
            result["local_crop_expansion"] = float(selected_pass["crop_expansion"])
            result["local_attempted_passes"] = attempted_passes
        return result


def _detector_continuity_segments(
    values: list[dict[str, Any]],
    *,
    max_frame_gap: int,
    max_center_jump: float,
    max_area_ratio: float,
    detector_scan_rank: dict[int, int] | None = None,
    detector_hit_frames: set[int] | None = None,
) -> list[list[dict[str, Any]]]:
    """Segment attempted joint reviews on the actual detector cadence."""

    if not values:
        return []
    # A skipped video frame is neutral, but a frame on which the full-frame
    # detector actually ran and failed to produce this track is negative
    # continuity evidence.  Use the complete scan schedule rather than a rank
    # made only from successful Local SCRFD attempts; adaptive burst scans can
    # otherwise disappear between two stride-aligned successes.
    hit_frames = detector_hit_frames or set()
    missed_scan_frames = sorted(
        int(frame_idx)
        for frame_idx in (detector_scan_rank or {})
        if int(frame_idx) not in hit_frames
    )
    result: list[list[dict[str, Any]]] = [[values[0]]]
    for item in values[1:]:
        prior = result[-1][-1]
        prior_frame = int(prior["frame_idx"])
        item_frame = int(item["frame_idx"])
        frame_gap = item_frame - prior_frame
        missed_start = bisect_right(missed_scan_frames, prior_frame)
        missed_end = bisect_right(missed_scan_frames, item_frame)
        adjacent_scan = (
            1 <= frame_gap <= max_frame_gap
            and missed_start == missed_end
        )
        if (
            not adjacent_scan
            or normalized_center_distance(item["box"], prior["box"]) > max_center_jump
            or area_ratio(item["box"], prior["box"]) > max_area_ratio
        ):
            result.append([])
        result[-1].append(item)
    return result


def _joint_strong_anchor(
    values: list[dict[str, Any]],
    policy: dict[str, Any] | None,
    local_review_stride: int = 1,
    detector_scan_rank: dict[int, int] | None = None,
) -> float | None:
    if policy is None:
        return None
    settings = policy["rule_gate"]
    normalizers = settings["normalizers"]
    window_frames = int(settings["strong_anchor_window_frames"])
    continuity = policy["continuity"]
    attempted = sorted(
        (
            item
            for item in values
            if int(item.get("local_match_count", -1)) >= 0
        ),
        key=lambda item: int(item["frame_idx"]),
    )
    detector_hit_frames = {
        int(item["frame_idx"])
        for item in values
        if item.get("source") == "detector"
    }
    best = 0.0
    for segment in _detector_continuity_segments(
        attempted,
        max_frame_gap=max(1, int(local_review_stride)),
        max_center_jump=float(continuity["segment_max_center_jump"]),
        max_area_ratio=float(continuity["segment_max_area_ratio"]),
        detector_scan_rank=detector_scan_rank,
        detector_hit_frames=detector_hit_frames,
    ):
        if len(segment) < window_frames:
            continue
        joint = [
            min(
                _scale(
                    item.get("local_confidence"),
                    float(normalizers["local_confidence_low"]),
                    float(normalizers["local_confidence_high"]),
                ),
                _scale(
                    item.get("verifier_face_probability"),
                    float(normalizers["verifier_score_low"]),
                    float(normalizers["verifier_score_high"]),
                ),
            )
            for item in segment
        ]
        for index in range(len(joint) - window_frames + 1):
            # The weakest frame defines the window. One isolated score spike
            # therefore cannot validate a track.
            best = max(best, min(joint[index : index + window_frames]))
    return best


def _summary(
    values: list[dict[str, Any]],
    policy: dict[str, Any] | None = None,
    detector_frame_stride: int = 1,
    detector_scan_rank: dict[int, int] | None = None,
    local_review_stride: int | None = None,
) -> dict[str, Any]:
    detector = [item for item in values if item["source"] == "detector"]
    detector_frames = sorted({int(item["frame_idx"]) for item in detector})
    leading_consecutive_detector_frames = 0
    if detector_frames:
        prior_frame: int | None = None
        for frame_idx in detector_frames:
            if prior_frame is not None:
                prior_rank = (
                    detector_scan_rank.get(prior_frame)
                    if detector_scan_rank is not None
                    else None
                )
                frame_rank = (
                    detector_scan_rank.get(frame_idx)
                    if detector_scan_rank is not None
                    else None
                )
                frame_gap = frame_idx - prior_frame
                adjacent_scan = 1 <= frame_gap <= max(1, int(detector_frame_stride)) and (
                    frame_rank == prior_rank + 1
                    if prior_rank is not None and frame_rank is not None
                    else True
                )
                if not adjacent_scan:
                    break
            leading_consecutive_detector_frames += 1
            prior_frame = frame_idx
    detector_scores = [float(item["confidence"]) for item in detector if item.get("confidence") is not None]
    attempted_local = [
        item for item in values if int(item.get("local_match_count", -1)) >= 0
    ]
    local = [item for item in attempted_local if int(item["local_match_count"]) >= 1]
    local_scores = [float(item["local_confidence"]) for item in local]
    verifier = [
        float(item["verifier_face_probability"])
        for item in values
        if item.get("verifier_face_probability") is not None
    ]
    effective_local_review_stride = max(
        1,
        int(
            detector_frame_stride
            if local_review_stride is None
            else local_review_stride
        ),
    )
    # Every attempted review, including zero-match failures, remains in the
    # ordered sequence below.  A failure therefore contributes a zero to every
    # strong window crossing it.  Sampled-out frames remain neutral, while the
    # complete detector scan schedule separately breaks a window on a real
    # full-frame miss.
    frame_indices = [int(item["frame_idx"]) for item in values]
    attempted_local_count = len(attempted_local)
    sampled_out_local_count = len(values) - attempted_local_count
    return {
        "frames": len(values),
        "first_frame": min(frame_indices) if frame_indices else None,
        "last_frame": max(frame_indices) if frame_indices else None,
        "detector_source_frames": len(detector),
        "detector_source_fraction": len(detector) / max(1, len(values)),
        "first_detector_frame": detector_frames[0] if detector_frames else None,
        "last_detector_frame": detector_frames[-1] if detector_frames else None,
        "leading_consecutive_detector_frames": (leading_consecutive_detector_frames),
        "detector_confidence_p50": median(detector_scores),
        "local_review_stride": effective_local_review_stride,
        "local_review_attempted_frames": attempted_local_count,
        "local_review_sampled_out_frames": sampled_out_local_count,
        "local_review_attempt_fraction": attempted_local_count / max(1, len(values)),
        "local_review_matched_frames": len(local),
        "local_review_failed_frames": attempted_local_count - len(local),
        "local_match_fraction": len(local) / max(1, attempted_local_count),
        "confidence_p25": float(np.quantile(local_scores, 0.25)) if local_scores else None,
        "confidence_p50": median(local_scores),
        "confidence_p75": float(np.quantile(local_scores, 0.75)) if local_scores else None,
        "track_fraction_with_confidence_gte_025": (
            sum(value >= 0.25 for value in local_scores) / max(1, attempted_local_count)
        ),
        "track_fraction_with_confidence_gte_035": (
            sum(value >= 0.35 for value in local_scores) / max(1, attempted_local_count)
        ),
        "track_fraction_with_confidence_gte_045": (
            sum(value >= 0.45 for value in local_scores) / max(1, attempted_local_count)
        ),
        "aspect_p50": median(
            max(
                (item["box"][2] - item["box"][0]) / max(item["box"][3] - item["box"][1], 1e-6),
                (item["box"][3] - item["box"][1]) / max(item["box"][2] - item["box"][0], 1e-6),
            )
            for item in values
        ),
        "verifier_frames": len(verifier),
        "verifier_coverage_fraction": len(verifier) / max(1, len(values)),
        "verifier_p25": float(np.quantile(verifier, 0.25)) if verifier else None,
        "verifier_p50": median(verifier),
        "verifier_p75": float(np.quantile(verifier, 0.75)) if verifier else None,
        "verifier_fraction_gte_022": (
            sum(value >= 0.22 for value in verifier) / len(verifier) if verifier else None
        ),
        "verifier_fraction_gte_030": (
            sum(value >= 0.30 for value in verifier) / len(verifier) if verifier else None
        ),
        "verifier_fraction_gte_050": (
            sum(value >= 0.50 for value in verifier) / len(verifier) if verifier else None
        ),
        "joint_strong_anchor": _joint_strong_anchor(
            values,
            policy,
            effective_local_review_stride,
            detector_scan_rank=detector_scan_rank,
        ),
    }


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _scale(value: float | None, low: float, high: float) -> float:
    if value is None or high <= low:
        return 0.0
    return _clamp_unit((float(value) - low) / (high - low))


def _admission_decision(
    summary: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    settings = policy["rule_gate"]
    detector_frames = int(summary["detector_source_frames"])
    local_fraction = float(summary["local_match_fraction"])
    moderate_local_fraction = float(summary["track_fraction_with_confidence_gte_035"])
    joint_anchor = float(summary.get("joint_strong_anchor") or 0.0)
    verifier_fraction = float(summary["verifier_fraction_gte_030"] or 0.0)
    standard_conditions = {
        "persistent_detection": detector_frames >= int(settings["min_detector_frames"]),
        "local_coverage": local_fraction >= float(settings["min_local_match_fraction"]),
        "moderate_local_support": moderate_local_fraction
        >= float(settings["min_local_confidence_fraction_gte_035"]),
        "joint_anchor": joint_anchor >= float(settings["min_joint_strong_anchor"]),
        # A very strong continuous joint window is sufficient confirmation.
        # Otherwise require verifier support across the candidate frames too.
        "confirmation": (
            joint_anchor >= float(settings["strong_joint_anchor"])
            or verifier_fraction >= float(settings["min_verifier_pass_fraction"])
        ),
    }
    standard_accepted = all(standard_conditions.values())

    short_settings = settings.get("short_track", {"enabled": False})
    local_p50 = summary.get("confidence_p50")
    verifier_p50 = summary.get("verifier_p50")
    short_conditions = {
        "enabled": bool(short_settings.get("enabled", False)),
        "detector_frame_range": (
            detector_frames >= int(short_settings.get("min_detector_frames", 1))
            and detector_frames <= int(short_settings.get("max_detector_frames", 0))
            and detector_frames < int(settings["min_detector_frames"])
        ),
        "local_coverage": local_fraction >= float(short_settings.get("min_local_match_fraction", 1.0)),
        "moderate_local_confidence": _at_least(
            local_p50,
            float(short_settings.get("moderate_local_confidence_p50", 1.0)),
        ),
        "moderate_verifier": _at_least(
            verifier_p50,
            float(short_settings.get("moderate_verifier_p50", 1.0)),
        ),
        "strong_model_support": (
            _at_least(
                local_p50,
                float(short_settings.get("strong_local_confidence_p50", 1.0)),
            )
            or _at_least(
                verifier_p50,
                float(short_settings.get("strong_verifier_p50", 1.0)),
            )
        ),
    }
    short_accepted = all(short_conditions.values())
    video_start_settings = settings.get("video_start_short_track", {"enabled": False})
    video_start_minimum = int(video_start_settings.get("min_detector_frames", 1))
    video_start_conditions = {
        "enabled": bool(video_start_settings.get("enabled", False)),
        "true_video_start": summary.get("first_detector_frame") == 0,
        "detector_frame_range": (
            detector_frames >= video_start_minimum
            and detector_frames <= int(short_settings.get("max_detector_frames", 0))
            and detector_frames < int(settings["min_detector_frames"])
        ),
        "consecutive_leading_detections": int(summary.get("leading_consecutive_detector_frames", 0))
        >= video_start_minimum,
        "strong_detector_confidence": _at_least(
            summary.get("detector_confidence_p50"),
            float(video_start_settings.get("min_detector_confidence_p50", 1.0)),
        ),
        "local_coverage": local_fraction >= float(video_start_settings.get("min_local_match_fraction", 1.0)),
        "local_confidence": _at_least(
            local_p50,
            float(video_start_settings.get("min_local_confidence_p50", 1.0)),
        ),
    }
    video_start_accepted = all(video_start_conditions.values())
    admission_path = (
        "standard"
        if standard_accepted
        else "short_track"
        if short_accepted
        else "video_start_short_track"
        if video_start_accepted
        else "rejected"
    )
    standard_evidence = {
        "detector_frames": detector_frames,
        "local_match_fraction": local_fraction,
        "local_confidence_fraction_gte_035": moderate_local_fraction,
        "joint_strong_anchor": joint_anchor,
        "verifier_pass_fraction": verifier_fraction,
    }
    return {
        "mode": "rule_gate",
        "accepted": (standard_accepted or short_accepted or video_start_accepted),
        "admission_path": admission_path,
        # Preserve the previous top-level fields for audit consumers while
        # making the selected path explicit.
        "conditions": standard_conditions,
        "evidence": standard_evidence,
        "standard": {
            "accepted": standard_accepted,
            "conditions": standard_conditions,
            "evidence": standard_evidence,
        },
        "short_track": {
            "accepted": short_accepted,
            "conditions": short_conditions,
            "evidence": {
                "detector_frames": detector_frames,
                "local_match_fraction": local_fraction,
                "local_confidence_p50": local_p50,
                "verifier_p50": verifier_p50,
            },
        },
        "video_start_short_track": {
            "accepted": video_start_accepted,
            "conditions": video_start_conditions,
            "evidence": {
                "detector_frames": detector_frames,
                "first_detector_frame": summary.get("first_detector_frame"),
                "leading_consecutive_detector_frames": summary.get("leading_consecutive_detector_frames"),
                "detector_confidence_p50": summary.get("detector_confidence_p50"),
                "local_match_fraction": local_fraction,
                "local_confidence_p50": local_p50,
            },
        },
    }


def _admission_values(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return evidence allowed to decide whether the core track is valid.

    Reliable endpoint replay is scheduled only after a track has accumulated
    sufficient detector support. Replayed frames can extend a geometrically
    continuous accepted interval, but including them in the denominator would
    make endpoint recovery able to reject the core that authorized it.
    """

    return [item for item in values if item.get("admission_scope", "core") != "reliable_endpoint_extension"]


def _scene_confined_values(track: dict[str, Any], values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clip any anomalous replay strictly to this track's shot boundaries."""

    start = track.get("start_scene_cut_frame") if bool(track.get("starts_at_scene_cut", False)) else None
    end = track.get("end_scene_cut_frame") if bool(track.get("ends_at_scene_cut", False)) else None
    return [
        item
        for item in values
        if (start is None or int(item["frame_idx"]) >= int(start))
        and (end is None or int(item["frame_idx"]) < int(end))
    ]


def _inside(track: dict[str, Any], frame_idx: int) -> bool:
    return any(first <= frame_idx <= last for first, last in track.get("accepted_intervals", []))


def _finalize(
    tracks: list[dict[str, Any]],
    scan: dict[str, Any],
    tracking: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Publish one admitted observation per track and frame.

    This is intentionally only a same-track selection step.  Different tracks
    remain independent until their final stabilized boxes are available.
    """

    track_map = {track["track_id"]: track for track in tracks}
    evidence_map = {(item["track_id"], int(item["frame_idx"])): item for item in evidence}
    published = [
        item
        for item in tracking.get("observations", [])
        if track_map[item["track_id"]].get("accepted") and _inside(track_map[item["track_id"]], int(item["frame_idx"]))
    ]
    by_track_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for detection in scan["detections"]:
        track = track_map[detection["track_id"]]
        if not track.get("accepted") or not _inside(track, int(detection["frame_idx"])):
            continue
        review = evidence_map.get((detection["track_id"], int(detection["frame_idx"])), {})
        by_track_frame.setdefault((str(detection["track_id"]), int(detection["frame_idx"])), []).append(
            {
                "frame_idx": int(detection["frame_idx"]),
                "track_id": detection["track_id"],
                "source": "detector",
                "box": list(detection["box"]),
                "detector_box": list(detection.get("detector_box", detection["box"])),
                "detector_landmarks": detection.get("detector_landmarks"),
                "confidence": float(detection["confidence"]),
                "local_match_count": int(review.get("local_match_count", -1)),
                "local_confidence": review.get("local_confidence"),
                "local_review_reason": review.get("local_review_reason"),
                "verifier_face_probability": review.get("verifier_face_probability"),
            }
        )
    for item in published:
        review = evidence_map.get((item["track_id"], int(item["frame_idx"])), {})
        final_item = dict(item)
        final_item["box"] = list(item["motion_box"])
        final_item["local_match_count"] = int(review.get("local_match_count", -1))
        final_item["local_confidence"] = review.get("local_confidence")
        final_item["local_review_reason"] = review.get("local_review_reason")
        final_item["verifier_face_probability"] = review.get("verifier_face_probability")
        by_track_frame.setdefault((str(item["track_id"]), int(item["frame_idx"])), []).append(final_item)
    output: list[dict[str, Any]] = []
    for values in by_track_frame.values():
        output.append(
            min(
                values,
                key=lambda item: (
                    0 if item["source"] == "detector" else 1,
                    -float(item.get("confidence", 0.0)),
                    str(item.get("geometry_source", "")),
                ),
            )
        )
    return sorted(output, key=lambda item: (int(item["frame_idx"]), item["track_id"]))


def finalize_precomputed(
    scan: dict[str, Any],
    tracks: list[dict[str, Any]],
    tracking: dict[str, Any],
    evidence: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    detector_frame_stride: int,
) -> dict[str, Any]:
    """Apply unchanged admission/finalization rules to already-reviewed evidence."""

    evidence = sorted(evidence, key=lambda item: (item["track_id"], int(item["frame_idx"])))
    evidence_by_track: dict[str, list[dict[str, Any]]] = {}
    for item in evidence:
        evidence_by_track.setdefault(str(item["track_id"]), []).append(item)
    policy = config["revalidation"]["policy"]
    detector_frame_stride = max(1, int(detector_frame_stride))
    # Detector-anchor review follows the actual runtime scan cadence for every
    # derived stride. The scan-rank map below
    # still distinguishes skipped frames (neutral) from performed scans that
    # missed the target (a continuity break).
    local_review_stride = max(1, detector_frame_stride)
    detector_scan_rank = {
        frame_idx: index
        for index, frame_idx in enumerate(
            sorted(
                int(item["frame_idx"])
                for item in scan.get("frames", [])
                if bool(item.get("detector_scan_performed", True))
            )
        )
    }
    for track in tracks:
        values = _scene_confined_values(track, evidence_by_track.get(str(track["track_id"]), []))
        admission_values = _admission_values(values)
        summary = _summary(
            admission_values,
            policy,
            detector_frame_stride,
            detector_scan_rank,
            local_review_stride=local_review_stride,
        )
        track["revalidation_summary"] = summary
        track["admission_decision"] = _admission_decision(summary, policy)
        track["accepted"] = bool(track["admission_decision"]["accepted"])
        track["accepted_intervals"] = []
        if track["accepted"]:
            continuity = policy["continuity"]
            segments = continuity_segments(
                values,
                float(continuity["segment_max_center_jump"]),
                float(continuity["segment_max_area_ratio"]),
            )
            if track["admission_decision"].get("admission_path") in {
                "short_track",
                "video_start_short_track",
            }:
                # The short-track gate already evaluated this track's complete
                # detector/local/verifier evidence. Publish only its own geometric
                # continuity segments that contain a detector observation.
                track["accepted_intervals"] = [
                    [
                        int(segment[0]["frame_idx"]),
                        int(segment[-1]["frame_idx"]),
                    ]
                    for segment in segments
                    if segment and any(item.get("source") == "detector" for item in segment)
                ]
            else:
                for segment in segments:
                    segment_admission_values = _admission_values(segment)
                    if segment_admission_values and bool(
                        _admission_decision(
                            _summary(
                                segment_admission_values,
                                policy,
                                detector_frame_stride,
                                detector_scan_rank,
                                local_review_stride=local_review_stride,
                            ),
                            policy,
                        )["accepted"]
                    ):
                        track["accepted_intervals"].append(
                            [
                                int(segment[0]["frame_idx"]),
                                int(segment[-1]["frame_idx"]),
                            ]
                        )
            if (
                not track["accepted_intervals"]
                and track["admission_decision"].get("admission_path")
                not in {"short_track", "video_start_short_track"}
            ):
                # Aggregate evidence admitted the track even though no one
                # continuity segment independently met the standard gate.
                # Preserve that privacy-safe admission without manufacturing
                # geometry across an unobserved gap.
                track["accepted_intervals"] = [
                    [
                        int(segment[0]["frame_idx"]),
                        int(segment[-1]["frame_idx"]),
                    ]
                    for segment in segments
                    if segment
                ]
        logger.debug(
            "[admission/onnx] %s detector=%s local=%.3f conf=%s verifier=%s accepted=%s",
            track["track_id"],
            summary["detector_source_frames"],
            summary["local_match_fraction"],
            summary["confidence_p50"],
            summary["verifier_p50"],
            track["accepted"],
        )
    observations = _finalize(tracks, scan, tracking, evidence)
    return {
        "evidence": evidence,
        "observations": observations,
        "accepted_tracks": sum(bool(track["accepted"]) for track in tracks),
        "seconds": 0.0,
    }


__all__ = ["LocalReviewer", "finalize_precomputed"]
