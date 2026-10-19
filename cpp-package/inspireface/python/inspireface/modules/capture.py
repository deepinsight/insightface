"""Pythonic wrappers for the synchronous InspireFace face-capture policy."""

import ctypes
import math
import numbers
import threading
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import List, Optional, Tuple

import numpy as np

from . import core as _native
from . import herror as errcode
from .exception import InvalidInputError, ResourceError, UnsupportedError, check_error
from .inspireface import FaceInformation, ImageStream, InspireFaceSession


_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


class FaceCaptureFilter(IntFlag):
    NONE = _native.HF_CAPTURE_FILTER_NONE
    FACE_COUNT = _native.HF_CAPTURE_FILTER_FACE_COUNT
    FACE_SIZE = _native.HF_CAPTURE_FILTER_FACE_SIZE
    FACE_POSITION = _native.HF_CAPTURE_FILTER_FACE_POSITION
    FACE_BOUNDARY = _native.HF_CAPTURE_FILTER_FACE_BOUNDARY
    STABILITY = _native.HF_CAPTURE_FILTER_STABILITY
    POSE = _native.HF_CAPTURE_FILTER_POSE
    QUALITY = _native.HF_CAPTURE_FILTER_QUALITY
    SHARPNESS = _native.HF_CAPTURE_FILTER_SHARPNESS
    BRIGHTNESS = _native.HF_CAPTURE_FILTER_BRIGHTNESS
    TRACK_COUNT = _native.HF_CAPTURE_FILTER_TRACK_COUNT


class FaceCaptureRejectReason(IntFlag):
    NONE = _native.HF_CAPTURE_REJECT_NONE
    NO_FACE = _native.HF_CAPTURE_REJECT_NO_FACE
    MULTIPLE_FACES = _native.HF_CAPTURE_REJECT_MULTIPLE_FACES
    FACE_TOO_SMALL = _native.HF_CAPTURE_REJECT_FACE_TOO_SMALL
    FACE_TOO_LARGE = _native.HF_CAPTURE_REJECT_FACE_TOO_LARGE
    FACE_OFF_CENTER = _native.HF_CAPTURE_REJECT_FACE_OFF_CENTER
    FACE_OUT_OF_BOUNDS = _native.HF_CAPTURE_REJECT_FACE_OUT_OF_BOUNDS
    UNSTABLE = _native.HF_CAPTURE_REJECT_UNSTABLE
    POSE = _native.HF_CAPTURE_REJECT_POSE
    QUALITY = _native.HF_CAPTURE_REJECT_QUALITY
    SHARPNESS = _native.HF_CAPTURE_REJECT_SHARPNESS
    BRIGHTNESS = _native.HF_CAPTURE_REJECT_BRIGHTNESS
    TRACK_COUNT_TOO_LOW = _native.HF_CAPTURE_REJECT_TRACK_COUNT_TOO_LOW


class FaceCaptureState(IntEnum):
    IDLE = _native.HF_CAPTURE_STATE_IDLE
    STABILIZING = _native.HF_CAPTURE_STATE_STABILIZING
    COLLECTING = _native.HF_CAPTURE_STATE_COLLECTING
    READY = _native.HF_CAPTURE_STATE_READY
    FINISHED = _native.HF_CAPTURE_STATE_FINISHED
    TRACK_LOST = _native.HF_CAPTURE_STATE_TRACK_LOST


def _require_native(name):
    function = getattr(_native, name, None)
    if function is None:
        raise UnsupportedError(
            "Face capture is not available in the loaded native library",
            errcode.HERR_UNSUPPORTED,
            missing_symbol=name,
        )
    return function


def _unsigned_integer(value, name, maximum=_UINT64_MAX):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise InvalidInputError(
            "{} must be an integer".format(name),
            errcode.HERR_INVALID_PARAM,
            **{name: value}
        )
    normalized = int(value)
    if normalized < 0 or normalized > maximum:
        raise InvalidInputError(
            "{} is outside its unsigned integer range".format(name),
            errcode.HERR_INVALID_PARAM,
            **{name: normalized}
        )
    return normalized


def _finite_float(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise InvalidInputError(
            "{} must be a finite number".format(name),
            errcode.HERR_CAPTURE_INVALID_CONFIG,
            **{name: value}
        )
    normalized = float(value)
    if not math.isfinite(normalized):
        raise InvalidInputError(
            "{} must be a finite number".format(name),
            errcode.HERR_CAPTURE_INVALID_CONFIG,
            **{name: normalized}
        )
    return normalized


@dataclass
class FaceCaptureConfig:
    """Mutable capture policy configuration with native-compatible defaults."""

    filter_mask: int = int(
        FaceCaptureFilter.FACE_COUNT
        | FaceCaptureFilter.FACE_SIZE
        | FaceCaptureFilter.FACE_POSITION
        | FaceCaptureFilter.FACE_BOUNDARY
        | FaceCaptureFilter.STABILITY
        | FaceCaptureFilter.TRACK_COUNT
    )
    output_count: int = 1
    min_track_count: int = 5
    stable_duration_ms: int = 300
    collect_duration_ms: int = 800
    max_collect_duration_ms: int = 3000
    track_lost_grace_ms: int = 300
    min_candidate_interval_ms: int = 150
    min_face_width_ratio: float = 0.12
    max_face_width_ratio: float = 0.75
    max_center_offset_x: float = 0.25
    max_center_offset_y: float = 0.25
    boundary_margin_ratio: float = 0.02
    max_center_motion_ratio: float = 0.025
    max_size_change_ratio: float = 0.08
    max_abs_yaw: float = 25.0
    max_abs_pitch: float = 25.0
    max_abs_roll: float = 20.0
    min_quality_score: float = 0.60
    min_sharpness_score: float = 0.03
    min_brightness_score: float = 0.15
    max_brightness_score: float = 0.90

    @classmethod
    def defaults(cls) -> "FaceCaptureConfig":
        """Read defaults from the native SDK when available."""
        function = getattr(_native, "HFGetDefaultFaceCaptureConfig", None)
        if function is None:
            return cls()
        value = _native.HFFaceCaptureConfig()
        check_error(function(ctypes.byref(value)), "Get default face capture config")
        return cls._from_native(value)

    @classmethod
    def _from_native(cls, value):
        return cls(
            filter_mask=int(value.filterMask),
            output_count=int(value.outputCount),
            min_track_count=int(value.minTrackCount),
            stable_duration_ms=int(value.stableDurationMs),
            collect_duration_ms=int(value.collectDurationMs),
            max_collect_duration_ms=int(value.maxCollectDurationMs),
            track_lost_grace_ms=int(value.trackLostGraceMs),
            min_candidate_interval_ms=int(value.minCandidateIntervalMs),
            min_face_width_ratio=float(value.minFaceWidthRatio),
            max_face_width_ratio=float(value.maxFaceWidthRatio),
            max_center_offset_x=float(value.maxCenterOffsetX),
            max_center_offset_y=float(value.maxCenterOffsetY),
            boundary_margin_ratio=float(value.boundaryMarginRatio),
            max_center_motion_ratio=float(value.maxCenterMotionRatio),
            max_size_change_ratio=float(value.maxSizeChangeRatio),
            max_abs_yaw=float(value.maxAbsYaw),
            max_abs_pitch=float(value.maxAbsPitch),
            max_abs_roll=float(value.maxAbsRoll),
            min_quality_score=float(value.minQualityScore),
            min_sharpness_score=float(value.minSharpnessScore),
            min_brightness_score=float(value.minBrightnessScore),
            max_brightness_score=float(value.maxBrightnessScore),
        )

    def _native(self):
        value = _native.HFFaceCaptureConfig()
        value.structSize = ctypes.sizeof(value)
        value.structVersion = _native.HF_FACE_CAPTURE_CONFIG_VERSION
        value.filterMask = _unsigned_integer(self.filter_mask, "filter_mask")
        value.outputCount = _unsigned_integer(self.output_count, "output_count", _UINT32_MAX)
        value.minTrackCount = _unsigned_integer(self.min_track_count, "min_track_count", _UINT32_MAX)
        value.stableDurationMs = _unsigned_integer(self.stable_duration_ms, "stable_duration_ms")
        value.collectDurationMs = _unsigned_integer(self.collect_duration_ms, "collect_duration_ms")
        value.maxCollectDurationMs = _unsigned_integer(
            self.max_collect_duration_ms, "max_collect_duration_ms"
        )
        value.trackLostGraceMs = _unsigned_integer(self.track_lost_grace_ms, "track_lost_grace_ms")
        value.minCandidateIntervalMs = _unsigned_integer(
            self.min_candidate_interval_ms, "min_candidate_interval_ms"
        )
        float_fields = (
            ("minFaceWidthRatio", "min_face_width_ratio"),
            ("maxFaceWidthRatio", "max_face_width_ratio"),
            ("maxCenterOffsetX", "max_center_offset_x"),
            ("maxCenterOffsetY", "max_center_offset_y"),
            ("boundaryMarginRatio", "boundary_margin_ratio"),
            ("maxCenterMotionRatio", "max_center_motion_ratio"),
            ("maxSizeChangeRatio", "max_size_change_ratio"),
            ("maxAbsYaw", "max_abs_yaw"),
            ("maxAbsPitch", "max_abs_pitch"),
            ("maxAbsRoll", "max_abs_roll"),
            ("minQualityScore", "min_quality_score"),
            ("minSharpnessScore", "min_sharpness_score"),
            ("minBrightnessScore", "min_brightness_score"),
            ("maxBrightnessScore", "max_brightness_score"),
        )
        for native_name, python_name in float_fields:
            setattr(value, native_name, _finite_float(getattr(self, python_name), python_name))
        return value


@dataclass(frozen=True)
class FaceCaptureMetrics:
    available_filters: FaceCaptureFilter
    face_width_ratio: float
    center_offset_x: float
    center_offset_y: float
    stability_score: float
    pose_score: float
    quality_score: float
    sharpness_score: float
    brightness_score: float


@dataclass(frozen=True)
class FaceCaptureProgress:
    state: FaceCaptureState
    candidate_count: int
    frame_id: int
    timestamp_ms: int
    track_id: int
    track_count: int
    evaluated_filters: FaceCaptureFilter
    reject_reasons: FaceCaptureRejectReason
    progress: float
    current_score: float
    metrics: FaceCaptureMetrics


@dataclass(frozen=True)
class FaceCaptureResult:
    frame_id: int
    timestamp_ms: int
    score: float
    face: FaceInformation
    metrics: FaceCaptureMetrics


def _metrics_from_native(value):
    return FaceCaptureMetrics(
        available_filters=FaceCaptureFilter(int(value.availableMetrics)),
        face_width_ratio=float(value.faceWidthRatio),
        center_offset_x=float(value.centerOffsetX),
        center_offset_y=float(value.centerOffsetY),
        stability_score=float(value.stabilityScore),
        pose_score=float(value.poseScore),
        quality_score=float(value.qualityScore),
        sharpness_score=float(value.sharpnessScore),
        brightness_score=float(value.brightnessScore),
    )


def _progress_from_native(value):
    return FaceCaptureProgress(
        state=FaceCaptureState(int(value.state)),
        candidate_count=int(value.candidateCount),
        frame_id=int(value.frameId),
        timestamp_ms=int(value.timestampMs),
        track_id=int(value.trackId),
        track_count=int(value.trackCount),
        evaluated_filters=FaceCaptureFilter(int(value.evaluatedFilters)),
        reject_reasons=FaceCaptureRejectReason(int(value.rejectReasons)),
        progress=float(value.progress),
        current_score=float(value.currentScore),
        metrics=_metrics_from_native(value.metrics),
    )


def _face_from_native(track_id, track_count, confidence, rect, roll, yaw, pitch, token):
    return FaceInformation(
        track_id=int(track_id),
        track_count=int(track_count),
        detection_confidence=float(confidence),
        location=(int(rect.x), int(rect.y), int(rect.x + rect.width), int(rect.y + rect.height)),
        roll=float(roll),
        yaw=float(yaw),
        pitch=float(pitch),
        _token=token,
    )


def _faces_from_view(view):
    count = int(view.detectedNum)
    if count < 0:
        raise ResourceError(
            "Native face detection snapshot returned a negative face count",
            errcode.HERR_SESS_FACE_DATA_ERROR,
            detected_count=count,
        )
    if count > 0:
        pointers = (
            view.trackIds,
            view.trackCounts,
            view.detConfidence,
            view.rects,
            view.angles.roll,
            view.angles.yaw,
            view.angles.pitch,
            view.tokens,
        )
        if not all(bool(pointer) for pointer in pointers):
            raise ResourceError(
                "Native face detection snapshot returned incomplete data",
                errcode.HERR_SESS_FACE_DATA_ERROR,
                detected_count=count,
            )
    faces = []
    for index in range(count):
        faces.append(
            _face_from_native(
                view.trackIds[index],
                view.trackCounts[index],
                view.detConfidence[index],
                view.rects[index],
                view.angles.roll[index],
                view.angles.yaw[index],
                view.angles.pitch[index],
                view.tokens[index],
            )
        )
    return tuple(faces)


class FaceDetectionSnapshot:
    """Owned face-detection result that remains valid across later session calls."""

    def __init__(self, session: InspireFaceSession, image):
        if not isinstance(session, InspireFaceSession) or session.closed:
            raise ResourceError("session must be open", errcode.HERR_INVALID_CONTEXT_HANDLE)
        self._handle = None
        self._lock = threading.RLock()
        self._faces: Tuple[FaceInformation, ...] = ()
        with session._managed_image_stream(image) as stream:
            handle = _native.HFFaceResultSnapshot()
            function = _require_native("HFExecuteFaceTrackSnapshot")
            check_error(
                function(session._sess, stream.handle, ctypes.byref(handle)),
                "Execute face tracking snapshot",
            )
            self._handle = handle
        try:
            view = _native.HFMultipleFaceData()
            function = _require_native("HFGetFaceResultSnapshotData")
            check_error(function(self._handle, ctypes.byref(view)), "Get face detection snapshot data")
            self._faces = _faces_from_view(view)
        except Exception:
            self.close()
            raise

    @property
    def faces(self) -> Tuple[FaceInformation, ...]:
        self._require_open()
        return self._faces

    @property
    def closed(self) -> bool:
        return self._handle is None

    def _require_open(self):
        if self._handle is None:
            raise ResourceError("Face detection snapshot is closed", errcode.HERR_INVALID_PARAM)

    def close(self):
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            self._handle = None
            function = _require_native("HFReleaseFaceResultSnapshot")
            check_error(function(handle), "Release face detection snapshot")

    release = close

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class FaceCaptureSession:
    """Synchronous, context-managed face-capture policy attached to a session."""

    def __init__(self, session: InspireFaceSession, config: Optional[FaceCaptureConfig] = None):
        if not isinstance(session, InspireFaceSession) or session.closed:
            raise ResourceError("session must be open", errcode.HERR_INVALID_CONTEXT_HANDLE)
        if config is None:
            config = FaceCaptureConfig.defaults()
        if not isinstance(config, FaceCaptureConfig):
            raise InvalidInputError("config must be FaceCaptureConfig", errcode.HERR_CAPTURE_INVALID_CONFIG)
        self._handle = None
        self._lock = threading.RLock()
        native_config = config._native()
        handle = _native.HFFaceCaptureSession()
        function = _require_native("HFCreateFaceCaptureSession")
        check_error(
            function(session._sess, ctypes.byref(native_config), ctypes.byref(handle)),
            "Create face capture session",
        )
        self._handle = handle

    @property
    def closed(self) -> bool:
        return self._handle is None

    def _require_open(self):
        if self._handle is None:
            raise ResourceError("Face capture session is closed", errcode.HERR_CAPTURE_INVALID_HANDLE)

    def update(
        self,
        image,
        frame_id: int,
        timestamp_ms: int,
        snapshot: Optional[FaceDetectionSnapshot] = None,
    ) -> FaceCaptureProgress:
        """Assess one frame; a supplied snapshot must belong to that same frame.

        Detection snapshots freeze ``track_count``. Continuous capture should
        create a new snapshot per frame, or omit ``snapshot`` to run tracking.
        """
        frame_id = _unsigned_integer(frame_id, "frame_id")
        timestamp_ms = _unsigned_integer(timestamp_ms, "timestamp_ms")
        if snapshot is not None and not isinstance(snapshot, FaceDetectionSnapshot):
            raise InvalidInputError("snapshot must be FaceDetectionSnapshot", errcode.HERR_INVALID_PARAM)
        with self._lock:
            self._require_open()
            with InspireFaceSession._managed_image_stream(image) as stream:
                progress = _native.HFFaceCaptureProgress()
                if snapshot is None:
                    function = _require_native("HFUpdateFaceCaptureSession")
                    status = function(
                        self._handle,
                        stream.handle,
                        frame_id,
                        timestamp_ms,
                        ctypes.byref(progress),
                    )
                else:
                    with snapshot._lock:
                        snapshot._require_open()
                        function = _require_native("HFUpdateFaceCaptureSessionWithSnapshot")
                        status = function(
                            self._handle,
                            stream.handle,
                            snapshot._handle,
                            frame_id,
                            timestamp_ms,
                            ctypes.byref(progress),
                        )
                check_error(status, "Update face capture session", frame_id=frame_id, timestamp_ms=timestamp_ms)
                return _progress_from_native(progress)

    def results(self) -> List[FaceCaptureResult]:
        with self._lock:
            self._require_open()
            count = _native.HFUInt32()
            function = _require_native("HFGetFaceCaptureResults")
            check_error(function(self._handle, None, 0, ctypes.byref(count)), "Query face capture results")
            if count.value == 0:
                return []
            if count.value > _native.HF_FACE_CAPTURE_MAX_RESULTS:
                raise ResourceError(
                    "Native face capture result count exceeds the public limit",
                    errcode.HERR_INVALID_BUFFER_SIZE,
                    result_count=count.value,
                )
            native_results = (_native.HFFaceCaptureResult * count.value)()
            copied = _native.HFUInt32()
            check_error(
                function(self._handle, native_results, count.value, ctypes.byref(copied)),
                "Get face capture results",
            )
            if copied.value != count.value:
                raise ResourceError(
                    "Native face capture result count changed during a serialized read",
                    errcode.HERR_INVALID_BUFFER_SIZE,
                    expected=count.value,
                    actual=copied.value,
                )
            output = []
            for value in native_results:
                face = _face_from_native(
                    value.trackId,
                    value.trackCount,
                    -1.0,
                    value.rect,
                    value.roll,
                    value.yaw,
                    value.pitch,
                    value.token,
                )
                output.append(
                    FaceCaptureResult(
                        frame_id=int(value.frameId),
                        timestamp_ms=int(value.timestampMs),
                        score=float(value.score),
                        face=face,
                        metrics=_metrics_from_native(value.metrics),
                    )
                )
            return output

    def finish(self) -> FaceCaptureProgress:
        with self._lock:
            self._require_open()
            progress = _native.HFFaceCaptureProgress()
            function = _require_native("HFFinishFaceCaptureSession")
            check_error(function(self._handle, ctypes.byref(progress)), "Finish face capture session")
            return _progress_from_native(progress)

    def reset(self) -> None:
        with self._lock:
            self._require_open()
            function = _require_native("HFResetFaceCaptureSession")
            check_error(function(self._handle), "Reset face capture session")

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            self._handle = None
            function = _require_native("HFReleaseFaceCaptureSession")
            check_error(function(handle), "Release face capture session")

    release = close

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = (
    "FaceCaptureConfig",
    "FaceCaptureFilter",
    "FaceCaptureMetrics",
    "FaceCaptureProgress",
    "FaceCaptureRejectReason",
    "FaceCaptureResult",
    "FaceCaptureSession",
    "FaceCaptureState",
    "FaceDetectionSnapshot",
)
