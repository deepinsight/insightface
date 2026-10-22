"""Fast, model-free scene-cut detection with one-frame flash suppression."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import cv2
import numpy as np


def _spatial_correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = first.astype(np.float32).reshape(-1)
    right = second.astype(np.float32).reshape(-1)
    left -= float(left.mean())
    right -= float(right.mean())
    left_energy = float(np.dot(left, left))
    right_energy = float(np.dot(right, right))
    denominator = math.sqrt(left_energy * right_energy)
    if denominator <= 1e-9:
        return 1.0 if left_energy <= 1e-9 and right_energy <= 1e-9 else 0.0
    return float(np.dot(left, right) / denominator)


def _histogram_correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = cv2.calcHist([first], [0], None, [32], [0, 256])
    right = cv2.calcHist([second], [0], None, [32], [0, 256])
    return float(cv2.compareHist(left, right, cv2.HISTCMP_CORREL))


def _flow_continuity(
    first: np.ndarray,
    second: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, float | int]:
    points = cv2.goodFeaturesToTrack(
        first,
        maxCorners=int(settings["max_corners"]),
        qualityLevel=float(settings["quality_level"]),
        minDistance=float(settings["min_distance"]),
        blockSize=int(settings["block_size"]),
    )
    point_count = 0 if points is None else len(points)
    minimum = int(settings["min_corners"])
    unavailable = {
        "scene_flow_points": point_count,
        "scene_flow_valid_fraction": 0.0,
        "scene_flow_inlier_fraction": 0.0,
        "scene_flow_median_fb_error": None,
    }
    if points is None or point_count < minimum:
        return unavailable

    window = int(settings["lk_window_size"])
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(settings["lk_max_iterations"]),
        float(settings["lk_epsilon"]),
    )
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        first,
        second,
        points,
        None,
        winSize=(window, window),
        maxLevel=int(settings["lk_max_level"]),
        criteria=criteria,
        minEigThreshold=float(settings["lk_min_eigenvalue"]),
    )
    if forward is None or forward_status is None:
        return unavailable
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        second,
        first,
        forward,
        None,
        winSize=(window, window),
        maxLevel=int(settings["lk_max_level"]),
        criteria=criteria,
        minEigThreshold=float(settings["lk_min_eigenvalue"]),
    )
    if backward is None or backward_status is None:
        return unavailable

    original = points.reshape(-1, 2)
    projected = forward.reshape(-1, 2)
    returned = backward.reshape(-1, 2)
    errors = np.linalg.norm(original - returned, axis=1)
    valid = (
        (forward_status.reshape(-1) > 0)
        & (backward_status.reshape(-1) > 0)
        & np.isfinite(errors)
        & (errors <= float(settings["max_forward_backward_error"]))
    )
    valid_count = int(valid.sum())
    inlier_fraction = 0.0
    if valid_count >= minimum:
        _, inliers = cv2.estimateAffinePartial2D(
            original[valid],
            projected[valid],
            method=cv2.RANSAC,
            ransacReprojThreshold=float(settings["ransac_reprojection_threshold"]),
            maxIters=500,
            confidence=0.99,
            refineIters=5,
        )
        if inliers is not None:
            inlier_fraction = float(inliers.reshape(-1).sum()) / point_count
    finite = errors[np.isfinite(errors)]
    return {
        "scene_flow_points": point_count,
        "scene_flow_valid_fraction": valid_count / point_count,
        "scene_flow_inlier_fraction": inlier_fraction,
        "scene_flow_median_fb_error": (float(np.median(finite)) if finite.size else None),
    }


class SceneCutDetector:
    """Finalize each adaptive-LK boundary after one-frame lookahead."""

    def __init__(self, scan_settings: dict[str, Any]):
        self.settings = dict(scan_settings["scene_cut_detector"])
        self.history: deque[float] = deque(maxlen=int(self.settings["history_frames"]))
        self.previous_gray: np.ndarray | None = None
        self.before_previous_gray: np.ndarray | None = None
        self.pending: dict[str, Any] | None = None
        self.last_frame_idx: int | None = None
        self.flushed = False

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        side = int(self.settings["signature_size"])
        resized = cv2.resize(frame, (side, side), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _base_audit(frame_idx: int, timestamp: float) -> dict[str, Any]:
        return {
            "frame_idx": int(frame_idx),
            "time_seconds": float(timestamp),
            "scene_cut_from_previous": False,
            "scene_cut_finalized": False,
            "scene_cut_candidate": False,
            "scene_cut_flow_confirmed": False,
            "scene_cut_appearance_confirmed": False,
            "scene_cut_confirmed": False,
            "scene_cut_flash_suppressed": False,
            "scene_mean_absdiff": 0.0,
            "scene_histogram_correlation": 1.0,
            "scene_spatial_correlation": None,
            "scene_rolling_median_absdiff": 0.0,
            "scene_flow_points": 0,
            "scene_flow_valid_fraction": None,
            "scene_flow_inlier_fraction": None,
            "scene_flow_median_fb_error": None,
            "scene_skip1_mean_absdiff": None,
            "scene_skip1_spatial_correlation": None,
        }

    def _adaptive_audit(
        self,
        frame_idx: int,
        timestamp: float,
        gray: np.ndarray,
    ) -> dict[str, Any]:
        assert self.previous_gray is not None
        audit = self._base_audit(frame_idx, timestamp)
        difference = float(cv2.absdiff(self.previous_gray, gray).mean())
        histogram = _histogram_correlation(self.previous_gray, gray)
        if self.history:
            rolling = float(np.median(np.asarray(self.history, dtype=np.float32)))
            content_jump = (
                difference >= float(self.settings["min_mean_absdiff"])
                and difference >= rolling * float(self.settings["relative_multiplier"])
                and difference >= rolling + float(self.settings["relative_offset"])
            )
        else:
            rolling = 0.0
            content_jump = difference >= float(self.settings["bootstrap_min_mean_absdiff"])
        audit["scene_mean_absdiff"] = difference
        audit["scene_histogram_correlation"] = histogram
        audit["scene_rolling_median_absdiff"] = rolling
        if content_jump:
            audit["scene_cut_candidate"] = True
            spatial_correlation = _spatial_correlation(self.previous_gray, gray)
            audit["scene_spatial_correlation"] = spatial_correlation
            flow = _flow_continuity(self.previous_gray, gray, self.settings)
            audit.update(flow)
            audit["scene_cut_flow_confirmed"] = float(flow["scene_flow_inlier_fraction"]) <= float(
                self.settings["max_flow_inlier_fraction"]
            )
            appearance = self.settings.get("appearance_confirmation", {})
            if bool(appearance.get("enabled", False)):
                audit["scene_cut_appearance_confirmed"] = (
                    difference >= float(appearance["min_mean_absdiff"])
                    and histogram <= float(appearance["max_histogram_correlation"])
                    and spatial_correlation <= float(appearance["max_spatial_correlation"])
                    and float(flow["scene_flow_inlier_fraction"])
                    <= float(appearance["max_flow_inlier_fraction"])
                )
            audit["scene_cut_confirmed"] = bool(
                audit["scene_cut_flow_confirmed"] or audit["scene_cut_appearance_confirmed"]
            )
        # The rolling baseline represents continuous within-shot motion. A
        # confirmed discontinuity must not inflate it; a large jump explained
        # by coherent LK motion is safe to learn as normal motion.
        if not bool(audit["scene_cut_confirmed"]):
            self.history.append(difference)
        return audit

    def _flash_pair(
        self,
        prior: dict[str, Any],
        current: dict[str, Any],
        gray: np.ndarray,
    ) -> bool:
        flash = self.settings["flash_suppression"]
        if (
            not bool(flash["enabled"])
            or not bool(prior["scene_cut_confirmed"])
            or not bool(current["scene_cut_confirmed"])
            or bool(prior["scene_cut_flash_suppressed"])
            or self.before_previous_gray is None
        ):
            return False
        skip_difference = float(cv2.absdiff(self.before_previous_gray, gray).mean())
        skip_correlation = _spatial_correlation(self.before_previous_gray, gray)
        prior["scene_skip1_mean_absdiff"] = skip_difference
        prior["scene_skip1_spatial_correlation"] = skip_correlation
        transition_floor = min(
            float(prior["scene_mean_absdiff"]),
            float(current["scene_mean_absdiff"]),
        )
        return skip_difference <= min(
            float(flash["max_skip_mean_absdiff"]),
            float(flash["max_skip_to_transition_ratio"]) * transition_floor,
        ) and skip_correlation >= float(flash["min_skip_spatial_correlation"])

    @staticmethod
    def _finalize(audit: dict[str, Any]) -> dict[str, Any]:
        audit["scene_cut_from_previous"] = bool(
            audit["scene_cut_confirmed"] and not audit["scene_cut_flash_suppressed"]
        )
        audit["scene_cut_finalized"] = True
        return audit

    def observe(
        self,
        frame_idx: int,
        frame: np.ndarray,
        timestamp: float = 0.0,
    ) -> list[dict[str, Any]]:
        if self.flushed:
            raise RuntimeError("cannot observe frames after SceneCutDetector.flush()")
        if self.last_frame_idx is not None and frame_idx != self.last_frame_idx + 1:
            raise ValueError("SceneCutDetector frame indices must be contiguous")
        self.last_frame_idx = int(frame_idx)
        gray = self._gray(frame)

        if self.previous_gray is None:
            audit = self._base_audit(frame_idx, timestamp)
            audit["scene_cut_finalized"] = True
            self.previous_gray = gray
            return [audit]

        current = self._adaptive_audit(frame_idx, timestamp, gray)
        finalized: list[dict[str, Any]] = []
        if self.pending is not None:
            if self._flash_pair(self.pending, current, gray):
                self.pending["scene_cut_flash_suppressed"] = True
                current["scene_cut_flash_suppressed"] = True
            finalized.append(self._finalize(self.pending))
        self.pending = current
        self.before_previous_gray = self.previous_gray
        self.previous_gray = gray
        return finalized

    def flush(self) -> list[dict[str, Any]]:
        if self.flushed:
            return []
        self.flushed = True
        if self.pending is None:
            return []
        pending, self.pending = self.pending, None
        return [self._finalize(pending)]


__all__ = ["SceneCutDetector"]
