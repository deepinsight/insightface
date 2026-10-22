from __future__ import annotations

import copy
import logging
import math
import sqlite3
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np

from ..errors import ApiError
from ..request_context import without_request_deadline
from ..storage import CursorCodec, Repository, SecretCodec
from .core import FaceService
from .images import ImageData

LOGGER = logging.getLogger("insightface_server.monitors")


class MonitorNotFoundError(KeyError):
    pass


class MonitorAlreadyExistsError(ValueError):
    pass


class MonitorLimitError(RuntimeError):
    pass


class MonitorPreviewDisabledError(RuntimeError):
    pass


class MonitorUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MonitorOptions:
    id: str
    name: str
    description: str
    enabled: bool
    url: str
    collection_id: str
    inference_fps: float
    match_threshold: float | None
    event_buffer_size: int
    confirm_frames: int
    absence_timeout_seconds: float
    cooldown_seconds: float
    emit_unknown: bool
    preview_enabled: bool
    created_at: str
    updated_at: str


@dataclass(slots=True)
class _Track:
    id: str
    kind: str
    person: dict[str, Any] | None
    matched_face_id: str | None
    similarity: float | None
    face: dict[str, Any]
    first_seen: float
    last_seen: float
    consecutive: int = 1
    confirmed: bool = False


def next_inference_time(*, started: float, finished: float, inference_fps: float) -> float:
    """Return the next eligible start time without ever scheduling catch-up work."""

    interval = 1.0 / inference_fps
    return max(started + interval, finished)


def redacted_rtsp_source(url: str) -> str:
    """Return a display-safe source that never includes credentials or query values."""

    parsed = urlsplit(url)
    host = parsed.hostname or "unknown"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    path = parsed.path or "/"
    return f"{parsed.scheme.lower()}://{host}{port}{path}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _open_capture(url: str, open_timeout_ms: int, read_timeout_ms: int):
    parameters: list[int] = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        parameters.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_ms])
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        parameters.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_ms])
    try:
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG, parameters)
    except (TypeError, cv2.error):
        capture = cv2.VideoCapture()
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_ms)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_ms)
        capture.open(url, cv2.CAP_FFMPEG)
        return capture


def _monitor_result(item: dict[str, Any]) -> dict[str, Any]:
    match = item.get("match")
    face = copy.deepcopy(item.get("face") or {})
    if not isinstance(match, dict):
        return {
            "status": "liveness_blocked" if item.get("status") == "liveness_blocked" else "unknown",
            "face": face,
            "person": None,
            "similarity": None,
            "matched_face_id": None,
        }
    source_person = match.get("person") or {}
    person = {
        "id": str(source_person.get("id") or match.get("person_id") or ""),
        "name": str(source_person.get("name") or ""),
        "external_id": str(source_person.get("external_id") or ""),
    }
    return {
        "status": "matched",
        "face": face,
        "person": person,
        "similarity": round(float(match.get("similarity", 0.0)), 6),
        "matched_face_id": match.get("matched_face_id"),
    }


def _normalized_box(result: dict[str, Any]) -> dict[str, float] | None:
    value = result.get("face", {}).get("bbox", {}).get("normalized", {})
    try:
        return {
            "left": float(value["left"]),
            "top": float(value["top"]),
            "width": float(value["width"]),
            "height": float(value["height"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _intersection_over_union(
    left: dict[str, Any] | _Track,
    right: dict[str, Any] | _Track,
) -> float:
    left_result = (
        {"face": left.face} if isinstance(left, _Track) else left
    )
    right_result = (
        {"face": right.face} if isinstance(right, _Track) else right
    )
    a = _normalized_box(left_result)
    b = _normalized_box(right_result)
    if a is None or b is None:
        return 0.0
    ax2 = a["left"] + a["width"]
    ay2 = a["top"] + a["height"]
    bx2 = b["left"] + b["width"]
    by2 = b["top"] + b["height"]
    intersection_width = max(0.0, min(ax2, bx2) - max(a["left"], b["left"]))
    intersection_height = max(0.0, min(ay2, by2) - max(a["top"], b["top"]))
    intersection = intersection_width * intersection_height
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return intersection / union if union > 0.0 else 0.0


class MonitorSession:
    """One long-running RTSP reader, inference loop, event buffer, and preview."""

    def __init__(
        self,
        service: FaceService,
        options: MonitorOptions,
        *,
        max_faces: int,
        preview_fps: float,
        jpeg_quality: int,
        open_timeout_seconds: float,
        read_timeout_seconds: float,
        reconnect_delay_seconds: float,
        capture_factory: Callable[[str, int, int], Any] = _open_capture,
    ) -> None:
        self.service = service
        self.options = options
        self.max_faces = max_faces
        self.preview_fps = preview_fps
        self.jpeg_quality = jpeg_quality
        self.open_timeout_ms = int(open_timeout_seconds * 1000)
        self.read_timeout_ms = int(read_timeout_seconds * 1000)
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.capture_factory = capture_factory
        self.stream_epoch = str(uuid.uuid4())
        self.started_at = _utc_now()

        self._condition = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._reader: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._state = "starting"
        self._connected = False
        self._connection_error: str | None = None
        self._inference_error: str | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_sequence = 0
        self._processed_frame_sequence = 0
        self._latest_jpeg: bytes | None = None
        self._preview_sequence = 0
        self._results: list[dict[str, Any]] = []
        self._threshold = options.match_threshold
        self._decoded_frames = 0
        self._processed_frames = 0
        self._skipped_frames = 0
        self._preview_frames = 0
        self._reconnects = 0
        self._inference_errors = 0
        self._last_processing_ms: float | None = None
        self._last_frame_at: str | None = None
        self._last_inference_at: str | None = None
        self._source_fps: float | None = None
        self._width: int | None = None
        self._height: int | None = None
        self._viewer_count = 0
        self._inference_times: deque[float] = deque(maxlen=120)
        self._events: deque[dict[str, Any]] = deque(maxlen=options.event_buffer_size)
        self._event_sequence = 0
        self._tracks: dict[str, _Track] = {}
        self._last_emitted: dict[tuple[str, str], float] = {}
        self._active_errors: set[str] = set()

    def start(self) -> None:
        with self._condition:
            if self._reader is not None:
                return
            self._reader = threading.Thread(
                target=self._reader_loop,
                name=f"monitor-reader-{self.options.id}",
                daemon=True,
            )
            self._worker = threading.Thread(
                target=self._inference_loop,
                name=f"monitor-inference-{self.options.id}",
                daemon=True,
            )
            self._reader.start()
            self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        current = threading.current_thread()
        for thread in (self._reader, self._worker):
            if thread is not None and thread is not current:
                thread.join(timeout=max(2.0, self.read_timeout_ms / 1000.0 + 1.0))
        with self._condition:
            self._connected = False
            self._state = "stopped"
            self._latest_jpeg = None
            self._condition.notify_all()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    @property
    def preview_enabled(self) -> bool:
        with self._condition:
            return self.options.preview_enabled

    @property
    def connected(self) -> bool:
        with self._condition:
            return self._connected and self._latest_frame is not None

    def apply_live_options(
        self,
        *,
        preview_enabled: bool | None = None,
        event_buffer_size: int | None = None,
        name: str | None = None,
        description: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        with self._condition:
            changes: dict[str, Any] = {}
            if preview_enabled is not None:
                changes["preview_enabled"] = preview_enabled
            if name is not None:
                changes["name"] = name
            if description is not None:
                changes["description"] = description
            if updated_at is not None:
                changes["updated_at"] = updated_at
            if event_buffer_size is not None:
                changes["event_buffer_size"] = event_buffer_size
                self._events = deque(
                    list(self._events)[-event_buffer_size:],
                    maxlen=event_buffer_size,
                )
            if changes:
                self.options = replace(self.options, **changes)
            if not self.options.preview_enabled:
                self._latest_jpeg = None
            self._condition.notify_all()

    def summary(self) -> dict[str, Any]:
        state = self.state()
        return {
            "status": state["status"],
            "connected": state["connected"],
            "stream_epoch": state["stream_epoch"],
            "last_frame_at": state["inference"]["last_frame_at"],
            "last_error": state["last_error"],
            "preview_active": state["preview"]["active"],
            "preview_viewers": state["preview"]["viewers"],
        }

    def state(self) -> dict[str, Any]:
        with self._condition:
            times = tuple(self._inference_times)
            effective_fps = 0.0
            if len(times) > 1 and times[-1] > times[0]:
                effective_fps = (len(times) - 1) / (times[-1] - times[0])
            status = self._state
            if self._connected and self._inference_error:
                status = "degraded"
            faces = copy.deepcopy(self._results)
            matched = sum(item.get("status") == "matched" for item in faces)
            return {
                "monitor_id": self.options.id,
                "stream_epoch": self.stream_epoch,
                "status": status,
                "connected": self._connected,
                "source": {
                    "width": self._width,
                    "height": self._height,
                    "source_fps": self._source_fps,
                },
                "inference": {
                    "configured_fps": self.options.inference_fps,
                    "actual_fps": round(effective_fps, 3),
                    "processing_ms": (
                        round(self._last_processing_ms, 3)
                        if self._last_processing_ms is not None
                        else None
                    ),
                    "last_frame_at": self._last_frame_at,
                    "last_inference_at": self._last_inference_at,
                    "frame_sequence": self._latest_frame_sequence,
                    "decoded_frames": self._decoded_frames,
                    "processed_frames": self._processed_frames,
                    "dropped_frames": self._skipped_frames,
                    "capacity_limited": bool(
                        self._last_processing_ms is not None
                        and self._last_processing_ms
                        > 1000.0 / self.options.inference_fps
                    ),
                },
                "threshold": self._threshold,
                "faces": faces,
                "matched_faces": matched,
                "unknown_faces": sum(item.get("status") == "unknown" for item in faces),
                "liveness_blocked_faces": sum(item.get("status") == "liveness_blocked" for item in faces),
                "preview": {
                    "enabled": self.options.preview_enabled,
                    "active": bool(
                        self.options.preview_enabled and self._viewer_count > 0
                    ),
                    "viewers": self._viewer_count,
                    "frames": self._preview_frames,
                },
                "reconnects": self._reconnects,
                "inference_errors": self._inference_errors,
                "last_error": self._inference_error or self._connection_error,
                "started_at": self.started_at,
            }

    def event_page(
        self,
        *,
        cursor_epoch: str | None,
        after_sequence: int | None,
        limit: int,
    ) -> dict[str, Any]:
        with self._condition:
            events = list(self._events)
            oldest = int(events[0]["sequence"]) if events else self._event_sequence + 1
            latest = self._event_sequence
            stream_reset = (
                cursor_epoch is not None and cursor_epoch != self.stream_epoch
            )
            truncated = False
            if cursor_epoch is None or stream_reset:
                selected = events[-limit:]
            else:
                after = int(after_sequence or 0)
                truncated = bool(events and after < oldest - 1)
                selected = [
                    item
                    for item in events
                    if int(item["sequence"]) > after
                ][:limit]
            if selected:
                next_sequence = int(selected[-1]["sequence"])
            elif cursor_epoch == self.stream_epoch and after_sequence is not None:
                next_sequence = min(int(after_sequence), latest)
            else:
                next_sequence = latest
            has_more = any(
                int(item["sequence"]) > next_sequence for item in events
            )
            return {
                "monitor_id": self.options.id,
                "stream_epoch": self.stream_epoch,
                "events": copy.deepcopy(selected),
                "oldest_sequence": oldest if events else None,
                "latest_sequence": latest,
                "next_sequence": next_sequence,
                "has_more": has_more,
                "truncated": truncated,
                "stream_reset": stream_reset,
            }

    def iter_mjpeg(self) -> Iterator[bytes]:
        last_sequence = -1
        with self._condition:
            if not self.options.preview_enabled:
                raise MonitorPreviewDisabledError
            if not self._connected or self._latest_frame is None:
                raise MonitorUnavailableError
            self._viewer_count += 1
            self._condition.notify_all()
        try:
            while not self._stop_event.is_set():
                with self._condition:
                    if not self.options.preview_enabled:
                        break
                    if (
                        self._latest_jpeg is None
                        or self._preview_sequence == last_sequence
                    ):
                        self._condition.wait(timeout=5.0)
                    if self._stop_event.is_set() or not self.options.preview_enabled:
                        break
                    jpeg = self._latest_jpeg
                    sequence = self._preview_sequence
                if jpeg is None or sequence == last_sequence:
                    continue
                last_sequence = sequence
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )
        finally:
            with self._condition:
                self._viewer_count = max(0, self._viewer_count - 1)
                if self._viewer_count == 0:
                    self._latest_jpeg = None

    def _append_event(
        self,
        event_type: str,
        *,
        now: float,
        track: _Track | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        identity = (
            str(track.person.get("id"))
            if track is not None and track.person
            else (track.id if track is not None else event_type)
        )
        cooldown_key = (event_type, identity)
        last = self._last_emitted.get(cooldown_key)
        if (
            last is not None
            and self.options.cooldown_seconds > 0
            and now - last < self.options.cooldown_seconds
        ):
            return
        self._last_emitted[cooldown_key] = now
        self._event_sequence += 1
        event: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "sequence": self._event_sequence,
            "type": event_type,
            "monitor_id": self.options.id,
            "stream_epoch": self.stream_epoch,
            "occurred_at": _utc_now(),
        }
        if track is not None:
            event.update(
                {
                    "track_id": track.id,
                    "person": copy.deepcopy(track.person),
                    "similarity": track.similarity,
                    "threshold": self._threshold,
                    "matched_face_id": track.matched_face_id,
                    "face": copy.deepcopy(track.face),
                }
            )
        if error is not None:
            event["error"] = copy.deepcopy(error)
        self._events.append(event)

    def _mark_error(
        self,
        *,
        source: str,
        code: str,
        message: str,
        now: float,
    ) -> None:
        with self._condition:
            if source not in self._active_errors:
                self._append_event(
                    "monitor_error",
                    now=now,
                    error={"code": code, "message": message},
                )
            self._active_errors.add(source)

    def _mark_recovered(self, *, source: str, now: float) -> None:
        with self._condition:
            was_active = source in self._active_errors
            self._active_errors.discard(source)
            if was_active and not self._active_errors:
                self._append_event("monitor_recovered", now=now)

    def _reader_loop(self) -> None:
        retry_delay = self.reconnect_delay_seconds
        next_preview = time.monotonic()
        while not self._stop_event.is_set():
            with self._condition:
                self._state = (
                    "starting" if self._decoded_frames == 0 else "reconnecting"
                )
                self._connected = False
            capture = None
            try:
                capture = self.capture_factory(
                    self.options.url,
                    self.open_timeout_ms,
                    self.read_timeout_ms,
                )
                if capture is None or not capture.isOpened():
                    raise RuntimeError("capture_open_failed")
                source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                with self._condition:
                    self._source_fps = (
                        round(source_fps, 3)
                        if math.isfinite(source_fps) and source_fps > 0
                        else None
                    )
                retry_delay = self.reconnect_delay_seconds
                while not self._stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None or frame.ndim != 3:
                        raise RuntimeError("capture_read_failed")
                    now = time.monotonic()
                    with self._condition:
                        self._decoded_frames += 1
                        self._latest_frame_sequence += 1
                        self._latest_frame = frame
                        self._width = int(frame.shape[1])
                        self._height = int(frame.shape[0])
                        self._last_frame_at = _utc_now()
                        self._connected = True
                        self._state = "running"
                        self._connection_error = None
                        viewers = self._viewer_count
                        preview_enabled = self.options.preview_enabled
                        self._condition.notify_all()
                    self._mark_recovered(source="source", now=now)
                    if preview_enabled and viewers > 0 and now >= next_preview:
                        self._publish_preview(frame)
                        next_preview = max(
                            next_preview + 1.0 / self.preview_fps,
                            now,
                        )
            except Exception as exc:
                if not self._stop_event.is_set():
                    LOGGER.warning(
                        "monitor_source_unavailable monitor_id=%s error_type=%s",
                        self.options.id,
                        type(exc).__name__,
                    )
                    message = "The RTSP source is unavailable; reconnecting."
                    with self._condition:
                        self._connected = False
                        self._connection_error = message
                        self._state = "reconnecting"
                        self._reconnects += 1
                    self._mark_error(
                        source="source",
                        code="rtsp_source_unavailable",
                        message=message,
                        now=time.monotonic(),
                    )
            finally:
                if capture is not None:
                    try:
                        capture.release()
                    except Exception:
                        pass
            if not self._stop_event.wait(retry_delay):
                retry_delay = min(
                    30.0,
                    max(self.reconnect_delay_seconds, retry_delay * 2.0),
                )

    def _inference_loop(self) -> None:
        next_allowed = time.monotonic()
        while not self._stop_event.is_set():
            frame: np.ndarray | None = None
            sequence = 0
            with self._condition:
                now = time.monotonic()
                fresh_frame = (
                    self._latest_frame is not None
                    and self._latest_frame_sequence != self._processed_frame_sequence
                )
                if fresh_frame and now >= next_allowed:
                    assert self._latest_frame is not None
                    frame = self._latest_frame.copy()
                    sequence = self._latest_frame_sequence
                else:
                    wait_time = 0.5
                    if fresh_frame:
                        wait_time = max(0.001, min(wait_time, next_allowed - now))
                    self._condition.wait(timeout=wait_time)
                    self._expire_tracks(time.monotonic())
                    continue
            started = time.monotonic()
            results: list[dict[str, Any]] | None = None
            selected_threshold: float | None = None
            error_message: str | None = None
            try:
                assert frame is not None
                image = ImageData(
                    content=b"",
                    sha256="",
                    pixels=frame,
                    filename="rtsp-frame",
                )
                raw_results, selected_threshold = self.service.search_all_faces(
                    self.options.collection_id,
                    image,
                    max_faces=self.max_faces,
                    threshold=self.options.match_threshold,
                )
                results = [_monitor_result(item) for item in raw_results]
            except ApiError as exc:
                error_message = f"{exc.code}: {exc.message}"
            except Exception as exc:
                LOGGER.exception(
                    "monitor_inference_failed monitor_id=%s error_type=%s",
                    self.options.id,
                    type(exc).__name__,
                )
                error_message = "internal_error: The frame could not be processed."
            finished = time.monotonic()
            with self._condition:
                previous_sequence = self._processed_frame_sequence
                self._processed_frame_sequence = sequence
                self._skipped_frames += max(0, sequence - previous_sequence - 1)
                self._processed_frames += 1
                self._last_processing_ms = (finished - started) * 1000.0
                self._last_inference_at = _utc_now()
                self._inference_times.append(started)
                if results is not None:
                    self._threshold = selected_threshold
                    self._results = self._update_tracks(results, finished)
                    self._inference_error = None
                else:
                    self._inference_errors += 1
                    self._inference_error = error_message
                    if error_message and error_message.startswith("liveness_unavailable:"):
                        self._results = []
                        self._tracks.clear()
            if results is None:
                self._mark_error(
                    source="inference",
                    code="monitor_inference_failed",
                    message=error_message or "The frame could not be processed.",
                    now=finished,
                )
            elif self._connected:
                self._mark_recovered(source="inference", now=finished)
            next_allowed = next_inference_time(
                started=started,
                finished=finished,
                inference_fps=self.options.inference_fps,
            )

    def _update_tracks(
        self,
        results: list[dict[str, Any]],
        now: float,
    ) -> list[dict[str, Any]]:
        unmatched_tracks = set(self._tracks)
        rendered: list[dict[str, Any]] = []
        for result in results:
            best_id: str | None = None
            best_iou = 0.15
            incoming_person = (result.get("person") or {}).get("id")
            for track_id in unmatched_tracks:
                track = self._tracks[track_id]
                track_person = (track.person or {}).get("id")
                if incoming_person and track_person and incoming_person != track_person:
                    continue
                overlap = _intersection_over_union(track, result)
                if overlap > best_iou:
                    best_iou = overlap
                    best_id = track_id
            if best_id is None:
                track = _Track(
                    id=str(uuid.uuid4()),
                    kind=str(result["status"]),
                    person=copy.deepcopy(result.get("person")),
                    matched_face_id=result.get("matched_face_id"),
                    similarity=result.get("similarity"),
                    face=copy.deepcopy(result.get("face") or {}),
                    first_seen=now,
                    last_seen=now,
                )
                self._tracks[track.id] = track
            else:
                unmatched_tracks.remove(best_id)
                track = self._tracks[best_id]
                incoming_kind = str(result["status"])
                if (
                    track.kind != incoming_kind
                    or (track.person or {}).get("id") != incoming_person
                ):
                    if track.confirmed:
                        self._emit_track_event(track, entering=False, now=now)
                    track.kind = incoming_kind
                    track.first_seen = now
                    track.consecutive = 1
                    track.confirmed = False
                else:
                    track.consecutive += 1
                track.last_seen = now
                track.person = copy.deepcopy(result.get("person"))
                track.matched_face_id = result.get("matched_face_id")
                track.similarity = result.get("similarity")
                track.face = copy.deepcopy(result.get("face") or {})
            if (
                track.kind in {"matched", "unknown"}
                and not track.confirmed
                and track.consecutive >= self.options.confirm_frames
            ):
                track.confirmed = True
                self._emit_track_event(track, entering=True, now=now)
            rendered.append(
                {
                    **copy.deepcopy(result),
                    "track_id": track.id,
                }
            )
        self._expire_tracks(now)
        return rendered

    def _expire_tracks(self, now: float) -> None:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if now - track.last_seen >= self.options.absence_timeout_seconds
        ]
        for track_id in expired:
            track = self._tracks.pop(track_id)
            if track.confirmed:
                self._emit_track_event(track, entering=False, now=now)

    def _emit_track_event(self, track: _Track, *, entering: bool, now: float) -> None:
        if track.kind not in {"matched", "unknown"}:
            return
        if track.kind == "unknown" and not self.options.emit_unknown:
            return
        prefix = "person" if track.kind == "matched" else "unknown"
        suffix = "enter" if entering else "exit"
        self._append_event(f"{prefix}_{suffix}", now=now, track=track)

    def _publish_preview(self, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            return
        with self._condition:
            if not self.options.preview_enabled or self._viewer_count == 0:
                return
            self._latest_jpeg = encoded.tobytes()
            self._preview_frames += 1
            self._preview_sequence += 1
            self._condition.notify_all()


class MonitorManager:
    def __init__(
        self,
        service: FaceService,
        repository: Repository,
        cursors: CursorCodec,
        credentials: SecretCodec,
        *,
        max_monitors: int = 4,
        max_faces: int = 100,
        preview_fps: float = 5.0,
        jpeg_quality: int = 82,
        open_timeout_seconds: float = 5.0,
        read_timeout_seconds: float = 5.0,
        reconnect_delay_seconds: float = 1.0,
        capture_factory: Callable[[str, int, int], Any] = _open_capture,
        session_factory: Callable[..., MonitorSession] = MonitorSession,
    ) -> None:
        self.service = service
        self.repository = repository
        self.cursors = cursors
        self.credentials = credentials
        self.max_monitors = max_monitors
        self.max_faces = max_faces
        self.preview_fps = preview_fps
        self.jpeg_quality = jpeg_quality
        self.open_timeout_seconds = open_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.capture_factory = capture_factory
        self.session_factory = session_factory
        # Lifecycle mutations are deliberately serialized.  Capacity checks,
        # durable configuration changes, and session publication must be one
        # operation; protecting only individual _sessions reads allows two
        # concurrent callers to both reserve the final slot.
        self._mutation_lock = threading.RLock()
        self._lock = threading.RLock()
        self._sessions: dict[str, MonitorSession] = {}
        self._startup_errors: dict[str, str] = {}
        self._closed = False

    def startup(self) -> None:
        with self._mutation_lock:
            self._require_open()
            for row in self.repository.list_enabled_monitors():
                monitor_id = str(row["id"])
                with self._lock:
                    current = self._sessions.get(monitor_id)
                if current is not None and not current.stopped:
                    continue
                try:
                    self._start_row(row)
                except Exception as exc:
                    LOGGER.error(
                        "monitor_restore_failed monitor_id=%s error_type=%s",
                        monitor_id,
                        type(exc).__name__,
                    )
                    with self._lock:
                        self._startup_errors[monitor_id] = (
                            "The Monitor could not be restored at startup."
                        )

    def close(self) -> None:
        with self._mutation_lock:
            self._closed = True
            with self._lock:
                sessions = list(self._sessions.items())
                self._sessions.clear()
            for monitor_id, session in sessions:
                self._stop_session(session, monitor_id=monitor_id)

    def create(self, item: dict[str, Any]) -> dict[str, Any]:
        with self._mutation_lock:
            self._require_open()
            collection_id = str(item["collection_id"])
            self.service.collection(collection_id)
            monitor_id = str(item["id"])
            if self.repository.get_monitor(monitor_id) is not None:
                raise MonitorAlreadyExistsError(monitor_id)
            if item.get("enabled", True):
                self._ensure_capacity(excluding=None)
            values = dict(item)
            values["source_url_ciphertext"] = self.credentials.encrypt(
                str(values.pop("url")),
                scope=f"monitor:{monitor_id}",
            )
            try:
                row = self.repository.create_monitor(values)
            except sqlite3.IntegrityError as exc:
                if self.repository.get_monitor(monitor_id) is not None:
                    raise MonitorAlreadyExistsError(monitor_id) from exc
                # A Collection delete can win after the validation read but
                # before this insert. Re-run the public lookup so that race is
                # reported as the existing collection_not_found API error.
                self.service.collection(collection_id)
                raise
            if row["enabled"]:
                try:
                    candidate = self._start_candidate(row)
                except Exception:
                    try:
                        # The row was already committed. Its cleanup must not be
                        # rejected solely because session startup consumed the
                        # originating request's remaining deadline.
                        with without_request_deadline():
                            self.repository.delete_monitor(monitor_id)
                    except Exception:
                        LOGGER.exception(
                            "monitor_create_compensation_failed monitor_id=%s",
                            monitor_id,
                        )
                    raise
                previous = self._publish_session(monitor_id, candidate)
                if previous is not None:
                    self._stop_session(previous, monitor_id=monitor_id)
            return self._public_monitor(row)

    def list(self, after: str, limit: int) -> list[dict[str, Any]]:
        return [
            self._public_monitor(row)
            for row in self.repository.list_monitors(after, limit)
        ]

    def get(self, monitor_id: str) -> dict[str, Any]:
        row = self.repository.get_monitor(monitor_id)
        if row is None:
            raise MonitorNotFoundError(monitor_id)
        return self._public_monitor(row)

    def update(self, monitor_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        with self._mutation_lock:
            self._require_open()
            existing = self.repository.get_monitor(monitor_id)
            if existing is None:
                raise MonitorNotFoundError(monitor_id)
            collection_changed = "collection_id" in changes
            if collection_changed:
                self.service.collection(str(changes["collection_id"]))
            desired_enabled = bool(changes.get("enabled", existing["enabled"]))
            persisted = dict(changes)
            if "url" in persisted:
                persisted["source_url_ciphertext"] = self.credentials.encrypt(
                    str(persisted.pop("url")),
                    scope=f"monitor:{monitor_id}",
                )

            restart_fields = {
                "url",
                "collection_id",
                "inference_fps",
                "match_threshold",
                "confirm_frames",
                "absence_timeout_seconds",
                "cooldown_seconds",
                "emit_unknown",
            }
            with self._lock:
                session = self._sessions.get(monitor_id)
            should_restart = bool(set(changes) & restart_fields)
            needs_candidate = desired_enabled and (
                session is None or session.stopped or should_restart
            )
            candidate: MonitorSession | None = None
            if needs_candidate:
                self._ensure_capacity(excluding=monitor_id)
                proposed = dict(existing)
                proposed.update(persisted)
                proposed["enabled"] = True
                candidate = self._start_candidate(proposed)

            try:
                updated = self.repository.update_monitor(monitor_id, persisted)
            except Exception:
                if candidate is not None:
                    self._stop_session(candidate, monitor_id=monitor_id)
                if collection_changed:
                    # As in create(), turn a foreign-key race into the stable
                    # collection_not_found API error when appropriate.
                    self.service.collection(str(changes["collection_id"]))
                raise
            if updated is None:
                if candidate is not None:
                    self._stop_session(candidate, monitor_id=monitor_id)
                raise MonitorNotFoundError(monitor_id)

            if not updated["enabled"]:
                with self._lock:
                    previous = self._sessions.pop(monitor_id, None)
                if previous is not None:
                    self._stop_session(previous, monitor_id=monitor_id)
            elif candidate is not None:
                previous = self._publish_session(monitor_id, candidate)
                if previous is not None and previous is not candidate:
                    self._stop_session(previous, monitor_id=monitor_id)
            elif session is not None:
                session.apply_live_options(
                    preview_enabled=(
                        bool(updated["preview_enabled"])
                        if "preview_enabled" in changes
                        else None
                    ),
                    event_buffer_size=(
                        int(updated["event_buffer_size"])
                        if "event_buffer_size" in changes
                        else None
                    ),
                    name=str(updated["name"]) if "name" in changes else None,
                    description=(
                        str(updated["description"])
                        if "description" in changes
                        else None
                    ),
                    updated_at=str(updated["updated_at"]),
                )
            with self._lock:
                self._startup_errors.pop(monitor_id, None)
            return self._public_monitor(updated)

    def delete(self, monitor_id: str) -> None:
        with self._mutation_lock:
            self._require_open()
            if self.repository.get_monitor(monitor_id) is None:
                raise MonitorNotFoundError(monitor_id)
            # Keep the live session intact if the durable delete fails. Once
            # the row is gone, detach before stopping so readers cannot obtain
            # a session for a deleted Monitor.
            if not self.repository.delete_monitor(monitor_id):
                raise MonitorNotFoundError(monitor_id)
            with self._lock:
                session = self._sessions.pop(monitor_id, None)
                self._startup_errors.pop(monitor_id, None)
            if session is not None:
                self._stop_session(session, monitor_id=monitor_id)

    def state(self, monitor_id: str) -> dict[str, Any]:
        row = self.repository.get_monitor(monitor_id)
        if row is None:
            raise MonitorNotFoundError(monitor_id)
        with self._lock:
            session = self._sessions.get(monitor_id)
            error = self._startup_errors.get(monitor_id)
        if session is not None:
            return session.state()
        return {
            "monitor_id": monitor_id,
            "stream_epoch": None,
            "status": "error" if error else "stopped",
            "connected": False,
            "source": {"width": None, "height": None, "source_fps": None},
            "inference": {
                "configured_fps": float(row["inference_fps"]),
                "actual_fps": 0.0,
                "processing_ms": None,
                "last_frame_at": None,
                "last_inference_at": None,
                "frame_sequence": 0,
                "decoded_frames": 0,
                "processed_frames": 0,
                "dropped_frames": 0,
                "capacity_limited": False,
            },
            "threshold": row["match_threshold"],
            "faces": [],
            "matched_faces": 0,
            "unknown_faces": 0,
            "liveness_blocked_faces": 0,
            "preview": {
                "enabled": bool(row["preview_enabled"]),
                "active": False,
                "viewers": 0,
                "frames": 0,
            },
            "reconnects": 0,
            "inference_errors": 0,
            "last_error": error,
            "started_at": None,
        }

    def events(
        self,
        monitor_id: str,
        *,
        cursor: str | None,
        limit: int,
    ) -> dict[str, Any]:
        if self.repository.get_monitor(monitor_id) is None:
            raise MonitorNotFoundError(monitor_id)
        scope = f"monitor-events:{monitor_id}"
        cursor_epoch: str | None = None
        after_sequence: int | None = None
        if cursor:
            raw = self.cursors.decode(cursor, scope)
            try:
                cursor_epoch, sequence_text = raw.rsplit(":", 1)
                after_sequence = int(sequence_text)
                if after_sequence < 0 or not cursor_epoch:
                    raise ValueError
            except ValueError as exc:
                from ..errors import bad_request

                raise bad_request(
                    "The cursor is invalid.",
                    code="invalid_cursor",
                ) from exc
        with self._lock:
            session = self._sessions.get(monitor_id)
        if session is None:
            page: dict[str, Any] = {
                "monitor_id": monitor_id,
                "stream_epoch": None,
                "events": [],
                "oldest_sequence": None,
                "latest_sequence": 0,
                "next_sequence": 0,
                "has_more": False,
                "truncated": False,
                "stream_reset": cursor_epoch is not None,
            }
            next_epoch = "stopped"
        else:
            page = session.event_page(
                cursor_epoch=cursor_epoch,
                after_sequence=after_sequence,
                limit=limit,
            )
            next_epoch = session.stream_epoch
        page["next_cursor"] = self.cursors.encode(
            scope,
            f"{next_epoch}:{page.pop('next_sequence')}",
        )
        return page

    def preview(self, monitor_id: str) -> Iterator[bytes]:
        row = self.repository.get_monitor(monitor_id)
        if row is None:
            raise MonitorNotFoundError(monitor_id)
        if not bool(row["preview_enabled"]):
            raise MonitorPreviewDisabledError
        with self._lock:
            session = self._sessions.get(monitor_id)
        if session is None or session.stopped:
            raise MonitorUnavailableError
        if not session.connected:
            raise MonitorUnavailableError
        return session.iter_mjpeg()

    def _ensure_capacity(self, *, excluding: str | None) -> None:
        with self._lock:
            active = sum(
                not session.stopped and monitor_id != excluding
                for monitor_id, session in self._sessions.items()
            )
        if active >= self.max_monitors:
            raise MonitorLimitError

    def _start_row(self, row: dict[str, Any]) -> MonitorSession:
        with self._mutation_lock:
            self._require_open()
            monitor_id = str(row["id"])
            self._ensure_capacity(excluding=monitor_id)
            session = self._start_candidate(row)
            previous = self._publish_session(monitor_id, session)
            if previous is not None and previous is not session:
                self._stop_session(previous, monitor_id=monitor_id)
            return session

    def _start_candidate(self, row: dict[str, Any]) -> MonitorSession:
        options = self._options(row)
        session = self.session_factory(
            self.service,
            options,
            max_faces=self.max_faces,
            preview_fps=self.preview_fps,
            jpeg_quality=self.jpeg_quality,
            open_timeout_seconds=self.open_timeout_seconds,
            read_timeout_seconds=self.read_timeout_seconds,
            reconnect_delay_seconds=self.reconnect_delay_seconds,
            capture_factory=self.capture_factory,
        )
        try:
            session.start()
        except Exception:
            self._stop_session(session, monitor_id=options.id)
            raise
        return session

    def _publish_session(
        self,
        monitor_id: str,
        session: MonitorSession,
    ) -> MonitorSession | None:
        with self._lock:
            previous = self._sessions.get(monitor_id)
            self._sessions[monitor_id] = session
            self._startup_errors.pop(monitor_id, None)
        return previous

    @staticmethod
    def _stop_session(session: MonitorSession, *, monitor_id: str) -> None:
        try:
            session.stop()
        except Exception:
            LOGGER.exception(
                "monitor_stop_failed monitor_id=%s",
                monitor_id,
            )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Monitor manager is closed.")

    def _options(self, row: dict[str, Any]) -> MonitorOptions:
        monitor_id = str(row["id"])
        url = self.credentials.decrypt(
            str(row["source_url_ciphertext"]),
            scope=f"monitor:{monitor_id}",
        )
        return MonitorOptions(
            id=monitor_id,
            name=str(row["name"]),
            description=str(row["description"]),
            enabled=bool(row["enabled"]),
            url=url,
            collection_id=str(row["collection_id"]),
            inference_fps=float(row["inference_fps"]),
            match_threshold=(
                float(row["match_threshold"])
                if row["match_threshold"] is not None
                else None
            ),
            event_buffer_size=int(row["event_buffer_size"]),
            confirm_frames=int(row["confirm_frames"]),
            absence_timeout_seconds=float(row["absence_timeout_seconds"]),
            cooldown_seconds=float(row["cooldown_seconds"]),
            emit_unknown=bool(row["emit_unknown"]),
            preview_enabled=bool(row["preview_enabled"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def _public_monitor(self, row: dict[str, Any]) -> dict[str, Any]:
        options = self._options(row)
        with self._lock:
            session = self._sessions.get(options.id)
            startup_error = self._startup_errors.get(options.id)
        runtime = (
            session.summary()
            if session is not None
            else {
                "status": (
                    "error" if startup_error else "stopped"
                ),
                "connected": False,
                "stream_epoch": None,
                "last_frame_at": None,
                "last_error": startup_error,
                "preview_active": False,
                "preview_viewers": 0,
            }
        )
        return {
            "id": options.id,
            "name": options.name,
            "description": options.description,
            "enabled": options.enabled,
            "source": {
                "type": "rtsp",
                "url": redacted_rtsp_source(options.url),
            },
            "collection_id": options.collection_id,
            "inference_fps": options.inference_fps,
            "match_threshold": options.match_threshold,
            "event_buffer_size": options.event_buffer_size,
            "event_policy": {
                "confirm_frames": options.confirm_frames,
                "absence_timeout_seconds": options.absence_timeout_seconds,
                "cooldown_seconds": options.cooldown_seconds,
                "emit_unknown": options.emit_unknown,
            },
            "preview_enabled": options.preview_enabled,
            "runtime": runtime,
            "created_at": options.created_at,
            "updated_at": options.updated_at,
        }
