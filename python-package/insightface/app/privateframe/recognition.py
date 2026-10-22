"""Fail-safe, track-level face recognition primitives.

The module deliberately keeps identity inference separate from detection,
tracking, admission, and rendering.  Streaming code can retain a bounded set
of aligned candidates and inject them into :class:`RecognitionEngine` only
after canonical tracks have been formed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np

from .geometry import area_ratio, containment, iou, normalized_center_distance
from .models import detect_faces

ARC_FACE_TEMPLATE_112 = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
ARC_FACE_TEMPLATE_112.setflags(write=False)

SUPPORTED_GALLERY_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
RECOGNITION_MODES = frozenset({"all", "blur_only", "exempt"})
UNKNOWN_ACTIONS = frozenset({"auto", "blur", "keep"})
_LOGGER = logging.getLogger(__name__)

# Gallery references are user-supplied still images, so they use a deliberately
# small, fixed upright-only detector policy instead of inheriting the much more
# expensive multi-angle video scan or local-revalidation policy. The 640 view
# supplies accurate landmarks for ordinary faces; 128 recovers very large
# faces. SCRFD performs global multi-scale NMS before PrivateFrame applies its
# stable output cap before the largest detected face is selected.
GALLERY_DETECTOR_INPUT_SIZES = (640, 128)
GALLERY_DETECTOR_CONFIDENCE_THRESHOLD = 0.50

# Video recognition alignment prefers already-computed local SCRFD landmarks
# and can fall back to the paired full-frame SCRFD box and landmarks. A single
# source gate applies to every sampling profile; profile differences concern
# only how many candidates are selected and how identity is decided. Gallery
# references follow the separate upright policy above.
LOCAL_LANDMARK_CONFIDENCE_THRESHOLD = 0.70
LOCAL_LANDMARK_MAX_CENTER_DISTANCE = 0.35
LOCAL_LANDMARK_MAX_AREA_RATIO = 2.0
LOCAL_LANDMARK_MIN_IOU = 0.30
LOCAL_LANDMARK_MIN_CONTAINMENT = 0.60

TEMPORAL_EVIDENCE_MIN_SELECTED_FRAMES = 3
TEMPORAL_EVIDENCE_MIN_ADJACENT_GAP_SECONDS = 1.0
TEMPORAL_EVIDENCE_SIMILARITY_OFFSET = 0.10
SINGLE_FRAME_SIMILARITY_OFFSET = 0.05

RECOGNITION_LANDMARK_SOURCES = frozenset({"local_scrfd", "global_scrfd"})


def detect_gallery_faces_upright(
    analysis: Any,
    image: np.ndarray,
    *,
    max_detections: int,
) -> list[dict[str, Any]]:
    """Detect Gallery faces at fixed 640/128 upright views without review."""

    result = detect_faces(
        analysis,
        image,
        input_sizes=GALLERY_DETECTOR_INPUT_SIZES,
        confidence_threshold=GALLERY_DETECTOR_CONFIDENCE_THRESHOLD,
        max_detections=max_detections,
    )
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        raise TypeError("Gallery SCRFD detector must return a sequence")
    if not all(isinstance(value, Mapping) for value in result):
        raise TypeError("Gallery SCRFD detections must be mappings")
    return [dict(value) for value in result]


def local_landmark_box_agreement(
    original_box: Any,
    local_box: Any,
) -> tuple[bool, dict[str, float]]:
    """Require a local-SCRFD face to be the same full-frame detection."""

    original = np.asarray(original_box, dtype=np.float64).reshape(4)
    local = np.asarray(local_box, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(original)) or not np.all(np.isfinite(local)):
        raise ValueError("recognition landmark boxes must be finite")
    overlap = float(iou(original, local))
    inside = max(
        float(containment(original, local)),
        float(containment(local, original)),
    )
    metrics = {
        "iou": overlap,
        "containment": inside,
        "center_distance": float(
            normalized_center_distance(original, local)
        ),
        "area_ratio": float(area_ratio(original, local)),
    }
    accepted = bool(
        metrics["center_distance"] <= LOCAL_LANDMARK_MAX_CENTER_DISTANCE
        and metrics["area_ratio"] <= LOCAL_LANDMARK_MAX_AREA_RATIO
        and (
            metrics["iou"] >= LOCAL_LANDMARK_MIN_IOU
            or metrics["containment"] >= LOCAL_LANDMARK_MIN_CONTAINMENT
        )
    )
    return accepted, metrics


def _landmark_array(value: Any) -> np.ndarray:
    landmarks = np.asarray(value, dtype=np.float64)
    if landmarks.shape != (5, 2):
        raise ValueError(f"ArcFace alignment requires five 2D landmarks, received {landmarks.shape}")
    if not np.all(np.isfinite(landmarks)):
        raise ValueError("ArcFace landmarks must be finite")
    centered = landmarks - np.mean(landmarks, axis=0, keepdims=True)
    if float(np.sum(centered * centered)) <= np.finfo(np.float64).eps:
        raise ValueError("ArcFace landmarks are degenerate")
    return landmarks


def estimate_arcface_similarity(landmarks: Any) -> np.ndarray:
    """Estimate the least-squares similarity mapping into the ArcFace 112 template."""

    source = _landmark_array(landmarks)
    destination = ARC_FACE_TEMPLATE_112.astype(np.float64)
    source_mean = np.mean(source, axis=0)
    destination_mean = np.mean(destination, axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean
    covariance = destination_centered.T @ source_centered / len(source)
    left, singular_values, right_transposed = np.linalg.svd(covariance)
    correction = np.eye(2, dtype=np.float64)
    if np.linalg.det(left) * np.linalg.det(right_transposed) < 0.0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_transposed
    variance = float(np.sum(source_centered * source_centered) / len(source))
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = destination_mean - scale * (rotation @ source_mean)
    matrix = np.column_stack((scale * rotation, translation)).astype(np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("ArcFace similarity transform is not finite")
    return matrix


def arcface_align_112(frame_bgr: np.ndarray, landmarks: Any) -> np.ndarray:
    """Return the conventional 112x112 ArcFace BGR alignment crop."""

    image = np.asarray(frame_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"ArcFace source image must be HWC BGR, received {image.shape}")
    if image.size == 0:
        raise ValueError("ArcFace source image must not be empty")
    matrix = estimate_arcface_similarity(landmarks)
    return cv2.warpAffine(
        image,
        matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def recognition_candidate_quality(
    aligned_bgr: np.ndarray,
    detector_box: Any,
    landmarks: Any,
    confidence: float,
    source_shape: Sequence[int],
    *,
    scan_angle_degrees: int = 0,
) -> tuple[float, bool, dict[str, float]]:
    """Rank identity anchors and reject geometrically unreliable alignments.

    These constants are an internal recognizer profile, not user-facing
    configuration. The gate is intentionally conservative because a rejected
    candidate makes a track UNKNOWN (handled by the selected policy), while a poor crop
    can create an unsafe false exemption.
    """

    aligned = np.asarray(aligned_bgr)
    if aligned.shape != (112, 112, 3):
        raise ValueError("recognition quality requires one aligned 112x112 BGR crop")
    if len(source_shape) < 2:
        raise ValueError("recognition source shape must contain height and width")
    height, width = int(source_shape[0]), int(source_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("recognition source dimensions must be positive")
    points = _landmark_array(landmarks)
    box = np.asarray(detector_box, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(box)):
        raise ValueError("recognition detector box must be finite")
    minimum_side = max(0.0, min(float(box[2] - box[0]), float(box[3] - box[1])))
    detector_confidence = float(np.clip(confidence, 0.0, 1.0))

    left_eye, right_eye, nose, left_mouth, right_mouth = points
    eye_left = float(np.linalg.norm(nose - left_eye))
    eye_right = float(np.linalg.norm(nose - right_eye))
    mouth_left = float(np.linalg.norm(nose - left_mouth))
    mouth_right = float(np.linalg.norm(nose - right_mouth))
    if min(eye_left, eye_right, mouth_left, mouth_right) <= 1e-6:
        pose = 0.0
    else:
        pose = min(
            min(eye_left, eye_right) / max(eye_left, eye_right),
            min(mouth_left, mouth_right) / max(mouth_left, mouth_right),
        )

    matrix = estimate_arcface_similarity(points)
    projected = np.column_stack((points, np.ones(5))) @ matrix.T
    residual = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (projected - ARC_FACE_TEMPLATE_112.astype(np.float64)) ** 2,
                    axis=1,
                )
            )
        )
    )
    inverse = cv2.invertAffineTransform(matrix)
    # A fixed 16x16 target grid estimates crop coverage to within far less than
    # the deliberately broad 0.55 gate, without adding a second full warp.
    axis = np.linspace(0.0, 111.0, 16, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis)
    target = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1), np.ones(grid_x.size)))
    source = target @ inverse.T
    coverage = float(
        np.mean(
            (source[:, 0] >= 0.0) & (source[:, 0] < width) & (source[:, 1] >= 0.0) & (source[:, 1] < height)
        )
    )

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    geometry_score = math.exp(-((residual / 8.0) ** 2))
    detail_score = float(np.clip(math.log1p(laplacian_variance) / math.log1p(500.0), 0.0, 1.0))
    size_score = float(np.clip(minimum_side / 112.0, 0.0, 1.0))
    view_score = 1.0 if int(scan_angle_degrees) == 0 else 0.75
    quality = max(
        1e-6,
        detector_confidence
        * coverage
        * (0.5 + 0.5 * pose)
        * (0.5 + 0.5 * geometry_score)
        * (0.7 + 0.3 * detail_score)
        * (0.7 + 0.3 * size_score)
        * view_score,
    )
    eligible = bool(
        detector_confidence >= 0.50
        and minimum_side >= 32.0
        and coverage >= 0.55
        and pose >= 0.20
        and residual <= 10.0
        and laplacian_variance >= 20.0
        and contrast >= 12.0
    )
    return (
        quality,
        eligible,
        {
            "confidence": detector_confidence,
            "minimum_side": minimum_side,
            "pose_score": pose,
            "alignment_residual": residual,
            "source_coverage": coverage,
            "laplacian_variance": laplacian_variance,
            "contrast": contrast,
            "view_score": view_score,
        },
    )


def l2_normalize_embedding(value: Any) -> np.ndarray:
    """Validate and L2-normalize one embedding vector."""

    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("face embedding must be a non-empty finite vector")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= np.finfo(np.float32).eps:
        raise ValueError("face embedding must have a non-zero L2 norm")
    result = np.ascontiguousarray(vector / norm, dtype=np.float32)
    result.setflags(write=False)
    return result


def _embed_aligned_face(recognizer: Any, aligned_bgr: np.ndarray) -> np.ndarray:
    """Infer and normalize one aligned crop through legacy ArcFaceONNX."""

    aligned = np.asarray(aligned_bgr)
    expected = (112, 112, 3)
    if aligned.shape != expected:
        raise ValueError(
            f"aligned ArcFace crop must have shape {expected}, received {aligned.shape}"
        )
    get_feat = getattr(recognizer, "get_feat", None)
    if not callable(get_feat):
        raise TypeError("ArcFace recognizer must provide get_feat(image)")
    input_size = tuple(getattr(recognizer, "input_size", ()))
    if input_size != (112, 112):
        raise RuntimeError(
            "PrivateFrame requires a 112x112 ArcFace recognizer, "
            f"received {input_size}"
        )
    output = np.asarray(get_feat(aligned))
    if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] <= 0:
        raise RuntimeError(
            "ArcFace get_feat() must return one non-empty embedding row, "
            f"received {output.shape}"
        )
    return l2_normalize_embedding(output[0])


@dataclass(frozen=True)
class RecognitionProfile:
    name: str
    max_frames_per_track: int
    minimum_cluster_similarity: float = 0.35


RECOGNITION_PROFILES: Mapping[str, RecognitionProfile] = MappingProxyType(
    {
        # A one-frame track cannot expose a switch; the single-frame decision
        # therefore compensates with a stricter cosine threshold.
        "fast": RecognitionProfile("fast", 1),
        "balanced": RecognitionProfile("balanced", 3),
        "accurate": RecognitionProfile("accurate", 5),
    }
)


def resolve_recognition_profile(
    name: str = "balanced",
    max_frames_per_track: int | None = None,
) -> RecognitionProfile:
    try:
        profile = RECOGNITION_PROFILES[str(name)]
    except KeyError as error:
        raise ValueError(f"recognition profile must be one of {sorted(RECOGNITION_PROFILES)}") from error
    if max_frames_per_track is None:
        return profile
    maximum = int(max_frames_per_track)
    if maximum <= 0:
        raise ValueError("max_frames_per_track must be positive")
    return replace(profile, max_frames_per_track=maximum)


@dataclass(frozen=True)
class RecognitionCandidate:
    """A bounded, aligned crop retained by Streaming for later inference."""

    frame_index: int
    quality: float
    aligned_face: np.ndarray
    landmark_source: str = "local_scrfd"

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("recognition candidate frame_index must be non-negative")
        if not math.isfinite(float(self.quality)) or float(self.quality) <= 0.0:
            raise ValueError("recognition candidate quality must be finite and positive")
        if np.asarray(self.aligned_face).shape != (112, 112, 3):
            raise ValueError("recognition candidate must contain a 112x112 BGR crop")
        if self.landmark_source not in RECOGNITION_LANDMARK_SOURCES:
            raise ValueError(
                "recognition candidate landmark_source must be local_scrfd or global_scrfd"
            )


@dataclass(frozen=True)
class TemporalThresholdEvidence:
    """Audit the temporal independence behind one track's cosine threshold."""

    selected_count: int
    span_seconds: float
    minimum_adjacent_gap_seconds: float | None
    decision_similarity_threshold: float
    effective_similarity_threshold: float
    threshold_reason: str

    def annotate(self, decision: IdentityDecision) -> IdentityDecision:
        """Attach the effective threshold evidence without changing identity."""

        return replace(
            decision,
            selected_frame_count=self.selected_count,
            effective_similarity_threshold=self.effective_similarity_threshold,
            temporal_evidence_span_seconds=self.span_seconds,
            minimum_adjacent_gap_seconds=self.minimum_adjacent_gap_seconds,
            threshold_reason=self.threshold_reason,
        )


def temporal_threshold_evidence(
    frame_indices: Sequence[int],
    *,
    frames_per_second: float,
    base_similarity_threshold: float,
) -> TemporalThresholdEvidence:
    """Choose the base threshold only for three independently timed samples."""

    fps = _finite_number(
        frames_per_second,
        field="frames_per_second",
        positive=True,
    )
    threshold = _cosine_value(
        base_similarity_threshold,
        field="base_similarity_threshold",
    )
    if threshold < 0.0:
        raise ValueError("base_similarity_threshold must be between 0 and 1")
    ordered = sorted(int(value) for value in frame_indices)
    if any(value < 0 for value in ordered):
        raise ValueError("temporal evidence frame indices must be non-negative")

    span_seconds = (
        float(ordered[-1] - ordered[0]) / fps if len(ordered) >= 2 else 0.0
    )
    minimum_gap = (
        min(
            float(current - previous) / fps
            for previous, current in pairwise(ordered)
        )
        if len(ordered) >= 2
        else None
    )
    independent = bool(
        len(ordered) >= TEMPORAL_EVIDENCE_MIN_SELECTED_FRAMES
        and minimum_gap is not None
        and minimum_gap >= TEMPORAL_EVIDENCE_MIN_ADJACENT_GAP_SECONDS
    )
    if independent:
        decision_threshold = threshold
        reason = "independent_temporal_evidence"
    else:
        decision_threshold = min(
            1.0,
            threshold + TEMPORAL_EVIDENCE_SIMILARITY_OFFSET,
        )
        reason = (
            "insufficient_selected_frames"
            if len(ordered) < TEMPORAL_EVIDENCE_MIN_SELECTED_FRAMES
            else "insufficient_temporal_separation"
        )

    # decide_track_identity keeps the existing stricter single-sample rule.
    # Record the final cosine gate here so the artifact states the threshold
    # that was actually applied, while passing the pre-singleton value below.
    effective_threshold = (
        min(1.0, decision_threshold + SINGLE_FRAME_SIMILARITY_OFFSET)
        if len(ordered) == 1
        else decision_threshold
    )
    if len(ordered) == 1:
        reason += "+single_frame_offset"
    return TemporalThresholdEvidence(
        selected_count=len(ordered),
        span_seconds=span_seconds,
        minimum_adjacent_gap_seconds=minimum_gap,
        decision_similarity_threshold=decision_threshold,
        effective_similarity_threshold=effective_threshold,
        threshold_reason=reason,
    )


def select_temporally_distributed(
    candidates: Sequence[RecognitionCandidate],
    max_frames: int,
) -> list[RecognitionCandidate]:
    """Choose one landmark proposal per time, distributed over the track.

    Local-SCRFD proposals take precedence inside a temporal bucket. A global
    SCRFD proposal is used when the bucket has no local proposal; quality then
    breaks ties within the preferred source. Collapsing same-frame proposals
    before bucketing prevents two crops from one instant from being counted as
    independent temporal evidence.
    """

    maximum = int(max_frames)
    if maximum <= 0:
        raise ValueError("max_frames must be positive")
    source_rank = {"local_scrfd": 0, "global_scrfd": 1}
    ordered_all = sorted(
        candidates,
        key=lambda item: (
            item.frame_index,
            source_rank[item.landmark_source],
            -item.quality,
        ),
    )
    ordered: list[RecognitionCandidate] = []
    for candidate in ordered_all:
        if ordered and candidate.frame_index == ordered[-1].frame_index:
            continue
        ordered.append(candidate)
    if len(ordered) <= maximum:
        return ordered
    first = ordered[0].frame_index
    last = ordered[-1].frame_index
    if first == last:
        return sorted(
            ordered,
            key=lambda item: (
                source_rank[item.landmark_source],
                -item.quality,
                item.frame_index,
            ),
        )[:maximum]
    buckets: list[list[RecognitionCandidate]] = [[] for _ in range(maximum)]
    span = last - first
    for candidate in ordered:
        bucket = min(
            maximum - 1,
            int((candidate.frame_index - first) * maximum / (span + 1)),
        )
        buckets[bucket].append(candidate)
    selected = [
        min(
            bucket,
            key=lambda item: (
                source_rank[item.landmark_source],
                -item.quality,
                item.frame_index,
            ),
        )
        for bucket in buckets
        if bucket
    ]
    selected_ids = {id(value) for value in selected}
    while len(selected) < maximum:
        remaining = [value for value in ordered if id(value) not in selected_ids]
        value = max(
            remaining,
            key=lambda item: (
                min(abs(item.frame_index - chosen.frame_index) for chosen in selected),
                -source_rank[item.landmark_source],
                item.quality,
                -item.frame_index,
            ),
        )
        selected.append(value)
        selected_ids.add(id(value))
    return sorted(selected, key=lambda item: item.frame_index)


@dataclass(frozen=True)
class EmbeddingSample:
    frame_index: int
    embedding: np.ndarray
    quality: float = 1.0

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("embedding frame_index must be non-negative")
        if not math.isfinite(float(self.quality)) or float(self.quality) <= 0.0:
            raise ValueError("embedding quality must be finite and positive")


def _normalized_matrix(values: Sequence[Any]) -> np.ndarray:
    vectors = [l2_normalize_embedding(value) for value in values]
    if not vectors:
        raise ValueError("at least one embedding is required")
    widths = {value.size for value in vectors}
    if len(widths) != 1:
        raise ValueError("all embeddings must have the same width")
    return np.stack(vectors)


def _complete_link_clusters(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    clusters = [[index] for index in range(len(vectors))]
    similarities = vectors @ vectors.T
    while True:
        best: tuple[float, int, int] | None = None
        for first in range(len(clusters)):
            for second in range(first + 1, len(clusters)):
                cross = similarities[np.ix_(clusters[first], clusters[second])]
                minimum = float(np.min(cross))
                if minimum < threshold:
                    continue
                candidate = (float(np.mean(cross)), first, second)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            break
        _score, first, second = best
        clusters[first] = sorted(clusters[first] + clusters[second])
        del clusters[second]
    return clusters


def _weighted_prototype(
    vectors: np.ndarray,
    indices: Sequence[int],
    weights: np.ndarray,
) -> np.ndarray:
    combined = np.average(vectors[list(indices)], axis=0, weights=weights[list(indices)])
    return l2_normalize_embedding(combined)


@dataclass(frozen=True)
class _TrackClusters:
    ordered: tuple[EmbeddingSample, ...]
    vectors: np.ndarray
    indices: tuple[tuple[int, ...], ...]
    prototypes: tuple[np.ndarray, ...]


def _cluster_track_embeddings(
    samples: Sequence[EmbeddingSample],
    minimum_cluster_similarity: float,
) -> _TrackClusters | None:
    if not samples:
        return None
    ordered = tuple(sorted(samples, key=lambda item: item.frame_index))
    vectors = _normalized_matrix([item.embedding for item in ordered])
    weights = _validated_weights([item.quality for item in ordered], len(ordered))
    threshold = _cosine_value(
        minimum_cluster_similarity,
        field="minimum_cluster_similarity",
    )
    clusters = _complete_link_clusters(vectors, threshold)
    clusters.sort(
        key=lambda values: (sum(weights[values]), len(values), -min(values)),
        reverse=True,
    )
    indices = tuple(tuple(values) for values in clusters)
    prototypes = tuple(
        _weighted_prototype(vectors, values, weights) for values in indices
    )
    return _TrackClusters(ordered, vectors, indices, prototypes)


class IdentityStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class IdentityDecision:
    """Track membership in the reference photo set, without named identities."""

    status: IdentityStatus
    matched_reference_files: tuple[str, ...] = ()
    similarity: float | None = None
    support: int = 0
    frame_indices: tuple[int, ...] = ()
    reason: str = ""
    selected_frame_count: int | None = None
    effective_similarity_threshold: float | None = None
    temporal_evidence_span_seconds: float | None = None
    minimum_adjacent_gap_seconds: float | None = None
    threshold_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-native reference-match evidence."""

        return {
            "status": self.status.value,
            "matched_reference_files": list(self.matched_reference_files),
            "similarity": self.similarity,
            "support": self.support,
            "frame_indices": list(self.frame_indices),
            "reason": self.reason,
            "selected_frame_count": self.selected_frame_count,
            "effective_similarity_threshold": self.effective_similarity_threshold,
            "temporal_evidence_span_seconds": self.temporal_evidence_span_seconds,
            "minimum_adjacent_gap_seconds": self.minimum_adjacent_gap_seconds,
            "threshold_reason": self.threshold_reason,
        }


def _reference_mapping(references: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    width: int | None = None
    for raw_name, embedding in references.items():
        name = _reference_file_name(raw_name)
        if name in result:
            raise ValueError(f"duplicate NFC-normalized reference filename: {name!r}")
        vector = l2_normalize_embedding(embedding)
        if width is None:
            width = vector.size
        elif vector.size != width:
            raise ValueError("reference embeddings must have the same width")
        result[name] = vector
    if not result:
        raise ValueError("at least one reference embedding is required")
    return result


def _best_reference_match(
    embedding: np.ndarray, references: Mapping[str, np.ndarray]
) -> tuple[str, float]:
    if any(value.size != embedding.size for value in references.values()):
        raise ValueError("track and reference embedding widths do not match")
    return min(
        ((name, float(embedding @ value)) for name, value in references.items()),
        key=lambda item: (-item[1], item[0]),
    )


def decide_track_identity(
    samples: Sequence[EmbeddingSample],
    prototypes: Mapping[str, np.ndarray],
    similarity_threshold: float,
    *,
    minimum_cluster_similarity: float = 0.35,
    single_frame_similarity_offset: float = SINGLE_FRAME_SIMILARITY_OFFSET,
) -> IdentityDecision:
    """Confirm reference-set membership only with unanimous temporal evidence.

    Every reference photo is an independent exemplar of the same selected set.
    No photo-to-photo runner-up margin is meaningful: two photos may show the
    same person. Different reference people are never averaged together.
    """

    threshold = _cosine_value(similarity_threshold, field="similarity_threshold")
    if threshold < 0.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    single_offset = _finite_number(
        single_frame_similarity_offset, field="single_frame_similarity_offset"
    )
    if not 0.0 <= single_offset <= 1.0:
        raise ValueError("single_frame_similarity_offset must be between 0 and 1")
    references = _reference_mapping(prototypes)
    frame_indices = [sample.frame_index for sample in samples]
    if len(set(frame_indices)) != len(frame_indices):
        return IdentityDecision(IdentityStatus.UNKNOWN, reason="duplicate_frame_samples")
    clustered = _cluster_track_embeddings(samples, minimum_cluster_similarity)
    if clustered is None:
        return IdentityDecision(IdentityStatus.UNKNOWN, reason="no_embedding")

    all_frames = tuple(item.frame_index for item in clustered.ordered)
    sample_threshold = min(1.0, threshold + single_offset) if len(samples) == 1 else threshold
    matches = [_best_reference_match(vector, references) for vector in clustered.vectors]
    confirmed = [(name, score) for name, score in matches if score >= sample_threshold]
    common = {
        "matched_reference_files": tuple(sorted({name for name, _score in confirmed})),
        "similarity": min(score for _name, score in matches),
        "support": len(confirmed),
        "frame_indices": all_frames,
    }
    if len(confirmed) != len(matches):
        mixed = bool(confirmed)
        return IdentityDecision(
            IdentityStatus.CONFLICT if mixed and len(clustered.indices) > 1 else IdentityStatus.UNKNOWN,
            reason=(
                "unconfirmed_track_cluster" if mixed and len(clustered.indices) > 1
                else "unconfirmed_track_sample" if mixed
                else "below_similarity_threshold"
            ),
            **common,
        )

    # Individual frames and each coherent appearance cluster must agree on
    # membership. This vetoes a weak pose/brief stranger hidden by averaging.
    # Different photos can independently support different video poses.
    cluster_scores = []
    for indices, prototype in zip(clustered.indices, clustered.prototypes):
        _name, score = _best_reference_match(prototype, references)
        cluster_scores.append(score)
        cluster_threshold = min(1.0, threshold + single_offset) if len(indices) == 1 else threshold
        if score < cluster_threshold:
            return IdentityDecision(
                IdentityStatus.CONFLICT if len(clustered.indices) > 1 else IdentityStatus.UNKNOWN,
                reason="unconfirmed_track_cluster",
                **{**common, "similarity": min(common["similarity"], score)},
            )
    return IdentityDecision(
        IdentityStatus.CONFIRMED,
        reason="confirmed_single_frame" if len(samples) == 1 else "confirmed_unanimous_support",
        **{**common, "similarity": min(common["similarity"], *cluster_scores)},
    )


@dataclass(frozen=True)
class GalleryImage:
    path: Path
    relative_file_name: str


@dataclass(frozen=True)
class GalleryFileFingerprint:
    """Non-biometric, location-independent reference photo content identity."""

    relative_file_name: str
    content_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "relative_file_name": self.relative_file_name,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class GalleryScan:
    root: Path
    images: tuple[GalleryImage, ...]


@dataclass(frozen=True)
class GalleryReference:
    file_name: str
    relative_file_name: str
    content_sha256: str
    embedding: np.ndarray
    quality: float
    detected_face_count: int
    selected_box: tuple[float, float, float, float]


@dataclass(frozen=True)
class GalleryRejection:
    file_name: str
    reason: str
    relative_file_name: str = ""
    content_sha256: str = ""


@dataclass(frozen=True)
class Gallery:
    root: Path
    prototypes: Mapping[str, np.ndarray]
    references: tuple[GalleryReference, ...]
    rejections: tuple[GalleryRejection, ...]
    file_fingerprints: tuple[GalleryFileFingerprint, ...]
    content_fingerprint: str

    @property
    def fingerprint(self) -> str:
        return self.content_fingerprint

    def fingerprint_dict(self) -> dict[str, Any]:
        """Return reference photo identity without paths, pixels, or embeddings."""

        return {
            "sha256": self.content_fingerprint,
            "files": [value.to_dict() for value in self.file_fingerprints],
        }


def scan_gallery(path: str | Path) -> GalleryScan:
    """Scan only direct, non-hidden, non-symlink reference images in a folder."""

    root = Path(path).expanduser()
    if root.is_symlink():
        raise ValueError("reference root must not be a symlink")
    if not root.is_dir():
        raise FileNotFoundError(f"reference directory does not exist: {root}")
    root = root.resolve()
    images: list[GalleryImage] = []
    normalized_names: dict[str, str] = {}
    for entry in sorted(root.iterdir(), key=lambda value: value.name):
        if entry.name.startswith(".") or entry.is_symlink() or not entry.is_file():
            continue
        if entry.suffix.lower() not in SUPPORTED_GALLERY_EXTENSIONS:
            continue
        normalized_name = _reference_file_name(entry.name)
        if normalized_name in normalized_names:
            raise ValueError(
                "reference filenames collide after Unicode NFC normalization: "
                f"{normalized_names[normalized_name]!r}, {entry.name!r}"
            )
        normalized_names[normalized_name] = entry.name
        images.append(GalleryImage(entry.resolve(), normalized_name))
    images.sort(key=lambda value: value.relative_file_name)
    return GalleryScan(root, tuple(images))


def _read_gallery_image(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gallery_content_fingerprint(values: Sequence[GalleryFileFingerprint]) -> str:
    payload = json.dumps(
        [value.to_dict() for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _detect_gallery_faces(detector: Any, image: np.ndarray) -> Sequence[Mapping[str, Any]]:
    if callable(detector):
        result = detector(image)
    elif callable(getattr(detector, "detect", None)):
        result = detector.detect(image)
    else:
        raise TypeError("reference detector must be callable or expose detect(image)")
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        raise TypeError("reference detector must return a sequence")
    if not all(isinstance(value, Mapping) for value in result):
        raise TypeError("reference detector results must be mappings")
    return result


def _detection_box(detection: Mapping[str, Any]) -> tuple[float, float, float, float]:
    box = np.asarray(detection.get("box"), dtype=np.float64)
    if box.shape != (4,) or not np.all(np.isfinite(box)):
        raise ValueError("face box must contain four finite coordinates")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("face box must have positive width and height")
    return tuple(float(value) for value in box)


def build_gallery(
    path: str | Path,
    detector: Any,
    recognizer: Any,
    *,
    image_loader: Callable[[Path], np.ndarray | None] = _read_gallery_image,
) -> Gallery:
    """Select the largest face from each reference photo, then validate it.

    A rejected largest face never falls back to another person in that photo.
    Detector and recognizer execution failures abort the task; unsuitable
    reference photographs are the only recoverable per-photo failures.
    """

    scanned = scan_gallery(path)
    references: list[GalleryReference] = []
    rejections: list[GalleryRejection] = []
    file_fingerprints = tuple(
        GalleryFileFingerprint(item.relative_file_name, _sha256_path(item.path))
        for item in scanned.images
    )
    fingerprint_by_relative_name = {value.relative_file_name: value for value in file_fingerprints}

    def reject(item: GalleryImage, reason: str) -> None:
        fingerprint = fingerprint_by_relative_name[item.relative_file_name]
        rejections.append(GalleryRejection(
            item.path.name, reason, item.relative_file_name, fingerprint.content_sha256
        ))
        _LOGGER.warning("Reference photo %s was not used: %s", item.path.name, reason)

    seen_content: set[str] = set()
    for item in scanned.images:
        fingerprint = fingerprint_by_relative_name[item.relative_file_name]
        if fingerprint.content_sha256 in seen_content:
            reject(item, "duplicate_content")
            continue
        seen_content.add(fingerprint.content_sha256)
        try:
            image = image_loader(item.path)
        except (OSError, ValueError, cv2.error):
            image = None
        image = None if image is None else np.asarray(image)
        if image is None or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            reject(item, "unreadable_image")
            continue
        try:
            detections = _detect_gallery_faces(detector, image)
        except MemoryError:
            raise
        except Exception as error:
            raise RuntimeError(f"Reference photo {item.path.name}: face detector inference failed") from error
        if not detections:
            reject(item, "no_face")
            continue
        # Validate boxes before ranking: with a malformed box we cannot know
        # which face is largest, so selecting another would be misleading.
        try:
            boxes = [_detection_box(value) for value in detections]
        except (TypeError, ValueError):
            reject(item, "invalid_face_box")
            continue
        selected_index = max(
            range(len(boxes)),
            key=lambda index: (boxes[index][2] - boxes[index][0]) * (boxes[index][3] - boxes[index][1]),
        )
        detection, selected_box = detections[selected_index], boxes[selected_index]
        if len(detections) > 1:
            _LOGGER.info(
                "Reference photo %s: detected %d faces; using only the largest face (box=%s), ignoring %d others",
                item.path.name, len(detections), selected_box, len(detections) - 1,
            )
        if "landmarks" not in detection:
            reject(item, "missing_landmarks")
            continue
        try:
            landmarks = _landmark_array(detection["landmarks"])
        except (TypeError, ValueError):
            reject(item, "invalid_landmarks")
            continue
        raw_confidence = detection.get("confidence")
        if (
            isinstance(raw_confidence, bool)
            or not isinstance(raw_confidence, (int, float))
            or not math.isfinite(float(raw_confidence))
            or not 0.0 < float(raw_confidence) <= 1.0
        ):
            reject(item, "invalid_confidence")
            continue
        aligned: np.ndarray | None = None
        try:
            aligned = arcface_align_112(image, landmarks)
            quality, eligible, _details = recognition_candidate_quality(
                aligned, selected_box, landmarks, float(raw_confidence), image.shape,
            )
        except (TypeError, ValueError, cv2.error):
            reject(item, "invalid_geometry")
            if aligned is not None and aligned.flags.writeable:
                aligned.fill(0)
            continue
        if not eligible:
            reject(item, "low_quality")
            if aligned.flags.writeable:
                aligned.fill(0)
            continue
        try:
            embedding = _embed_aligned_face(recognizer, aligned)
        except MemoryError:
            raise
        except Exception as error:
            raise RuntimeError(f"Reference photo {item.path.name}: recognizer inference failed") from error
        finally:
            if aligned.flags.writeable:
                aligned.fill(0)
        references.append(GalleryReference(
            item.path.name, item.relative_file_name, fingerprint.content_sha256,
            embedding, quality, len(detections), selected_box,
        ))

    _LOGGER.info(
        "Reference photos: read %d, used %d, skipped %d",
        len(scanned.images), len(references), len(rejections),
    )
    if not references:
        raise ValueError("No usable faces were found in the reference photos; choose clearer reference photos and try again")
    prototypes = _reference_mapping({
        reference.relative_file_name: reference.embedding for reference in references
    })
    return Gallery(
        root=scanned.root,
        prototypes=MappingProxyType(prototypes),
        references=tuple(references),
        rejections=tuple(rejections),
        file_fingerprints=file_fingerprints,
        content_fingerprint=_gallery_content_fingerprint(file_fingerprints),
    )


@dataclass(frozen=True)
class PolicyDecision:
    should_blur: bool
    reason: str


def _identity_from_artifact(value: IdentityDecision | Mapping[str, Any] | None) -> IdentityDecision:
    if isinstance(value, IdentityDecision):
        status, files = value.status, value.matched_reference_files
    elif isinstance(value, Mapping):
        status, files = value.get("status"), value.get("matched_reference_files")
    else:
        raise ValueError("missing or invalid reference-match artifact")
    try:
        status = IdentityStatus(status)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid reference-match artifact status") from error
    if not isinstance(files, (list, tuple)):
        raise ValueError("invalid reference-match artifact filenames")
    try:
        normalized_files = tuple(_reference_file_name(name) for name in files)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid reference-match artifact filenames") from error
    if len(set(normalized_files)) != len(normalized_files):
        raise ValueError("reference-match artifact filenames collide after Unicode NFC normalization")
    if status is IdentityStatus.CONFIRMED and not normalized_files:
        raise ValueError("confirmed reference-match artifact has no matched photos")
    if isinstance(value, IdentityDecision):
        return replace(value, status=status, matched_reference_files=normalized_files)
    return IdentityDecision(status, matched_reference_files=normalized_files, reason="artifact")


def resolve_unknown_action(mode: str, unknown_action: str = "auto") -> str:
    policy_mode = _recognition_mode(mode)
    if not isinstance(unknown_action, str) or unknown_action not in UNKNOWN_ACTIONS:
        raise ValueError(f"recognition unknown_action must be one of {sorted(UNKNOWN_ACTIONS)}")
    if policy_mode == "all":
        return "blur"
    if unknown_action == "auto":
        return "keep" if policy_mode == "blur_only" else "blur"
    return unknown_action


def apply_identity_policy(
    mode: str,
    identity: IdentityDecision | Mapping[str, Any] | None,
    unknown_action: str = "auto",
) -> PolicyDecision:
    """Map reference-match evidence to rendering without changing geometry."""

    policy_mode = _recognition_mode(mode)
    fallback = resolve_unknown_action(policy_mode, unknown_action)
    if policy_mode == "all":
        return PolicyDecision(True, "policy_all")
    decision = _identity_from_artifact(identity)
    if decision.status is not IdentityStatus.CONFIRMED:
        return PolicyDecision(fallback == "blur", f"{decision.status.value.lower()}_{fallback}")
    if not decision.matched_reference_files:
        raise ValueError("confirmed reference match has no matched photos")
    return PolicyDecision(policy_mode == "blur_only", "reference_match" if policy_mode == "blur_only" else "reference_exempt")


@dataclass(frozen=True)
class RecognitionEngine:
    enabled: bool
    mode: str
    profile: RecognitionProfile
    unknown_action: str = "blur"
    similarity_threshold: float | None = None
    recognizer: Any = None
    gallery: Gallery | Any = None

    @property
    def max_frames_per_track(self) -> int:
        return self.profile.max_frames_per_track

    @staticmethod
    def unknown_decision(reason: str) -> IdentityDecision:
        text = str(reason).strip()
        if not text:
            raise ValueError("unknown identity reason must not be empty")
        return IdentityDecision(IdentityStatus.UNKNOWN, reason=text)

    def identify_track(
        self,
        candidates: Sequence[RecognitionCandidate],
        *,
        frames_per_second: float,
    ) -> IdentityDecision:
        if not self.enabled:
            return replace(
                IdentityDecision(IdentityStatus.UNKNOWN, reason="policy_all"),
                selected_frame_count=0,
                threshold_reason="policy_all",
            )
        selected = select_temporally_distributed(candidates, self.profile.max_frames_per_track)
        temporal_evidence = temporal_threshold_evidence(
            [value.frame_index for value in selected],
            frames_per_second=frames_per_second,
            base_similarity_threshold=float(self.similarity_threshold),
        )
        # Model failures must escape this method. Treating them as UNKNOWN
        # would silently keep targets visible in blur_only mode.
        samples = [
            EmbeddingSample(
                frame_index=value.frame_index,
                embedding=_embed_aligned_face(self.recognizer, value.aligned_face),
                quality=value.quality,
            )
            for value in selected
        ]
        return temporal_evidence.annotate(decide_track_identity(
            samples,
            self.gallery.prototypes,
            temporal_evidence.decision_similarity_threshold,
            minimum_cluster_similarity=self.profile.minimum_cluster_similarity,
        ))


def create_recognition_engine(
    settings: Mapping[str, Any],
    *,
    recognizer: Any,
    gallery_detector: Any,
    gallery_builder: Callable[..., Any] = build_gallery,
) -> RecognitionEngine:
    """Create recognition lazily; mode all loads no model or reference photos."""

    mode = _recognition_mode(settings.get("mode", "all"))
    fallback = resolve_unknown_action(mode, settings.get("unknown_action", "auto"))
    if mode == "all":
        return RecognitionEngine(False, mode, RECOGNITION_PROFILES["balanced"], fallback)
    profile = resolve_recognition_profile(
        str(settings.get("profile", "balanced")), settings.get("max_frames_per_track"),
    )
    reference_dir = settings.get("reference_dir")
    if not isinstance(reference_dir, (str, Path)) or not str(reference_dir).strip():
        raise ValueError("selective recognition requires reference_dir")
    if "similarity_threshold" not in settings:
        raise ValueError("selective recognition requires similarity_threshold")
    threshold = _cosine_value(settings["similarity_threshold"], field="similarity_threshold")
    if threshold < 0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if not callable(getattr(recognizer, "get_feat", None)):
        raise TypeError("selective recognition requires an ArcFace recognizer with get_feat()")
    if gallery_detector is None:
        raise ValueError("selective recognition requires a reference detector")
    gallery = gallery_builder(reference_dir, gallery_detector, recognizer)
    if not isinstance(getattr(gallery, "prototypes", None), Mapping):
        raise TypeError("reference builder must return an object with reference embedding mappings")
    if not gallery.prototypes:
        raise ValueError("No usable faces were found in the reference photos")
    return RecognitionEngine(True, mode, profile, fallback, threshold, recognizer, gallery)


def _finite_number(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        suffix = " finite and positive" if positive else " finite"
        raise ValueError(f"{field} must be{suffix}")
    return result


def _cosine_value(value: Any, *, field: str) -> float:
    result = _finite_number(value, field=field)
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{field} must be between -1 and 1")
    return result


def _validated_weights(values: Sequence[float] | None, count: int) -> np.ndarray:
    if values is None:
        return np.ones(count, dtype=np.float64)
    weights = np.asarray(values, dtype=np.float64).reshape(-1)
    if weights.size != count:
        raise ValueError(f"expected {count} embedding weights, received {weights.size}")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("embedding weights must be finite and positive")
    return weights


def _reference_file_name(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("reference filename must be a string")
    result = unicodedata.normalize("NFC", value)
    if not result or result in {".", ".."} or any(character in result for character in ("/", "\\", "\x00")):
        raise ValueError("reference filename must be a non-empty direct filename")
    return result


def _recognition_mode(value: Any) -> str:
    mode = str(value)
    if mode not in RECOGNITION_MODES:
        raise ValueError(f"recognition mode must be one of {sorted(RECOGNITION_MODES)}")
    return mode


__all__ = [
    "ARC_FACE_TEMPLATE_112",
    "GALLERY_DETECTOR_CONFIDENCE_THRESHOLD",
    "GALLERY_DETECTOR_INPUT_SIZES",
    "LOCAL_LANDMARK_CONFIDENCE_THRESHOLD",
    "LOCAL_LANDMARK_MAX_AREA_RATIO",
    "LOCAL_LANDMARK_MAX_CENTER_DISTANCE",
    "LOCAL_LANDMARK_MIN_CONTAINMENT",
    "LOCAL_LANDMARK_MIN_IOU",
    "RECOGNITION_LANDMARK_SOURCES",
    "RECOGNITION_MODES",
    "UNKNOWN_ACTIONS",
    "RECOGNITION_PROFILES",
    "SINGLE_FRAME_SIMILARITY_OFFSET",
    "SUPPORTED_GALLERY_EXTENSIONS",
    "TEMPORAL_EVIDENCE_MIN_ADJACENT_GAP_SECONDS",
    "TEMPORAL_EVIDENCE_MIN_SELECTED_FRAMES",
    "TEMPORAL_EVIDENCE_SIMILARITY_OFFSET",
    "EmbeddingSample",
    "Gallery",
    "GalleryFileFingerprint",
    "GalleryImage",
    "GalleryReference",
    "GalleryRejection",
    "GalleryScan",
    "IdentityDecision",
    "IdentityStatus",
    "PolicyDecision",
    "RecognitionCandidate",
    "RecognitionEngine",
    "RecognitionProfile",
    "TemporalThresholdEvidence",
    "apply_identity_policy",
    "arcface_align_112",
    "build_gallery",
    "create_recognition_engine",
    "decide_track_identity",
    "detect_gallery_faces_upright",
    "estimate_arcface_similarity",
    "l2_normalize_embedding",
    "local_landmark_box_agreement",
    "recognition_candidate_quality",
    "resolve_recognition_profile",
    "resolve_unknown_action",
    "scan_gallery",
    "select_temporally_distributed",
    "temporal_threshold_evidence",
]
