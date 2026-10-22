from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest
from insightface_server.services.rtsp import (
    MonitorOptions,
    MonitorPreviewDisabledError,
    MonitorSession,
    _monitor_result,
    next_inference_time,
    redacted_rtsp_source,
)


def _options(**changes) -> MonitorOptions:
    values = {
        "id": "front-gate",
        "name": "Front gate",
        "description": "",
        "enabled": True,
        "url": "rtsp://camera.example/live",
        "collection_id": "employees",
        "inference_fps": 30.0,
        "match_threshold": None,
        "event_buffer_size": 10,
        "confirm_frames": 2,
        "absence_timeout_seconds": 0.1,
        "cooldown_seconds": 0.0,
        "emit_unknown": True,
        "preview_enabled": False,
        "created_at": "2026-07-23T00:00:00Z",
        "updated_at": "2026-07-23T00:00:00Z",
    }
    values.update(changes)
    return MonitorOptions(**values)


def test_next_inference_time_never_schedules_catch_up_work() -> None:
    assert next_inference_time(started=10.0, finished=10.2, inference_fps=2.0) == 10.5
    assert next_inference_time(started=10.0, finished=10.8, inference_fps=2.0) == 10.8


def test_rtsp_source_redaction_removes_credentials_and_query() -> None:
    source = redacted_rtsp_source(
        "rtsp://camera-user:secret@192.168.1.20:8554/live?token=private"
    )
    assert source == "rtsp://192.168.1.20:8554/live"
    assert "secret" not in source
    assert "token" not in source


class _FastCapture:
    def __init__(self) -> None:
        self.released = False
        self.value = 40

    def isOpened(self) -> bool:
        return not self.released

    def get(self, property_id: int) -> float:
        return 25.0 if property_id == cv2.CAP_PROP_FPS else 0.0

    def read(self):
        if self.released:
            return False, None
        time.sleep(0.002)
        self.value = 40 + ((self.value + 1) % 4)
        return True, np.full((80, 120, 3), self.value, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class _SlowService:
    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0
        self.lock = threading.Lock()

    def search_all_faces(self, *_args, **_kwargs):
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            time.sleep(0.05)
            return (
                [
                    {
                        "face": {
                            "bbox": {
                                "pixels": {
                                    "x": 10,
                                    "y": 10,
                                    "width": 30,
                                    "height": 30,
                                },
                                "normalized": {
                                    "left": 1 / 12,
                                    "top": 1 / 8,
                                    "width": 0.25,
                                    "height": 0.375,
                                },
                            },
                            "landmarks": [],
                            "detection_score": 0.99,
                            "quality": {"score": 0.9},
                        },
                        "match": {
                            "person": {
                                "id": "alice",
                                "name": "Alice",
                                "external_id": "A-1",
                                "metadata": {"must_not_leak": True},
                            },
                            "similarity": 0.88,
                            "matched_face_id": "face-1",
                        },
                    }
                ],
                0.4,
            )
        finally:
            with self.lock:
                self.active -= 1


def _session(options: MonitorOptions) -> tuple[MonitorSession, _SlowService]:
    service = _SlowService()
    session = MonitorSession(
        service,  # type: ignore[arg-type]
        options,
        max_faces=100,
        preview_fps=30.0,
        jpeg_quality=90,
        open_timeout_seconds=1.0,
        read_timeout_seconds=1.0,
        reconnect_delay_seconds=0.1,
        capture_factory=lambda *_args: _FastCapture(),
    )
    return session, service


def test_session_keeps_latest_frame_and_emits_confirmed_memory_events() -> None:
    session, service = _session(_options())
    session.start()
    deadline = time.monotonic() + 2.0
    while session.state()["inference"]["processed_frames"] < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    state = session.state()
    events = session.event_page(cursor_epoch=None, after_sequence=None, limit=10)
    session.stop()

    assert state["inference"]["processed_frames"] >= 3
    assert state["inference"]["decoded_frames"] > state["inference"]["processed_frames"]
    assert state["inference"]["dropped_frames"] > 0
    assert state["inference"]["capacity_limited"] is True
    assert service.maximum_active == 1
    assert state["matched_faces"] == 1
    assert state["faces"][0]["person"] == {
        "id": "alice",
        "name": "Alice",
        "external_id": "A-1",
    }
    assert any(event["type"] == "person_enter" for event in events["events"])


def test_preview_is_disabled_by_default_and_raw_when_explicitly_enabled() -> None:
    disabled, _ = _session(_options(preview_enabled=False))
    with pytest.raises(MonitorPreviewDisabledError):
        next(disabled.iter_mjpeg())

    session, _ = _session(_options(preview_enabled=True, confirm_frames=1))
    session.start()
    deadline = time.monotonic() + 2.0
    while not session.connected and time.monotonic() < deadline:
        time.sleep(0.01)
    frame = next(session.iter_mjpeg())
    session.stop()

    header, jpeg = frame.split(b"\r\n\r\n", 1)
    decoded = cv2.imdecode(
        np.frombuffer(jpeg.rstrip(b"\r\n"), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert b"Content-Type: image/jpeg" in header
    assert decoded is not None
    # The server relays the unannotated source. A drawn green/amber rectangle
    # would create a large channel spread on this otherwise uniform frame.
    assert int(decoded.max()) - int(decoded.min()) < 10


def test_event_ring_reports_truncation_reset_and_resizes_to_latest_items() -> None:
    session, _ = _session(_options(event_buffer_size=3))
    with session._condition:  # noqa: SLF001 - deterministic unit-level state exercise
        for index in range(5):
            session._append_event(  # noqa: SLF001
                f"test_{index}",
                now=float(index),
            )
    page = session.event_page(
        cursor_epoch=session.stream_epoch,
        after_sequence=1,
        limit=10,
    )
    reset = session.event_page(
        cursor_epoch="old-epoch",
        after_sequence=5,
        limit=2,
    )
    session.apply_live_options(event_buffer_size=2)
    resized = session.event_page(cursor_epoch=None, after_sequence=None, limit=10)

    assert [item["sequence"] for item in page["events"]] == [3, 4, 5]
    assert page["truncated"] is True
    assert reset["stream_reset"] is True
    assert [item["sequence"] for item in reset["events"]] == [4, 5]
    assert [item["sequence"] for item in resized["events"]] == [4, 5]


@pytest.mark.parametrize(
    ("fps", "expected"),
    [(0.5, 2.0), (1.0, 1.0), (10.0, 0.1)],
)
def test_requested_rate_maps_to_start_interval(fps: float, expected: float) -> None:
    assert next_inference_time(
        started=4.0,
        finished=4.01,
        inference_fps=fps,
    ) == pytest.approx(4.0 + expected)


@pytest.mark.parametrize("liveness", [
    {"status": "ok", "is_live": False, "live_score": 0.1},
    {"status": "input_rejected", "is_live": None, "live_score": None},
    {
        "status": "input_rejected", "is_live": None, "live_score": None,
        "reason": (
            "Insufficient image area around the face for liveness detection. Move the face "
            "toward the center, step back from the camera, or use a less tightly cropped image."
        ),
    },
])
def test_blocked_face_is_not_unknown_and_restarts_identity_confirmation(liveness) -> None:
    session, service = _session(_options(confirm_frames=2))
    items, _ = service.search_all_faces()
    matched = _monitor_result(items[0])
    face = {**items[0]["face"], "liveness": liveness}
    blocked = _monitor_result({"status": "liveness_blocked", "face": face, "match": None})
    session._update_tracks([matched], 0.0)
    session._results = session._update_tracks([blocked], 0.01)
    session._results = session._update_tracks([blocked], 0.02)
    state = session.state()
    assert state["matched_faces"] == state["unknown_faces"] == 0
    assert state["liveness_blocked_faces"] == 1
    assert state["faces"][0]["person"] is None
    assert state["faces"][0]["face"]["liveness"] == face["liveness"]
    assert not session.event_page(cursor_epoch=None, after_sequence=None, limit=10)["events"]
    session._update_tracks([matched], 0.03)
    assert not session.event_page(cursor_epoch=None, after_sequence=None, limit=10)["events"]
    session._update_tracks([matched], 0.04)
    session._update_tracks([blocked], 0.05)
    events = session.event_page(cursor_epoch=None, after_sequence=None, limit=10)["events"]
    assert [event["type"] for event in events] == ["person_enter", "person_exit"]


def test_liveness_failure_clears_previous_video_identity(monkeypatch) -> None:
    from insightface_server.errors import ApiError

    session, service = _session(_options(confirm_frames=1))
    session.start()
    try:
        deadline = time.monotonic() + 2.0
        while session.state()["matched_faces"] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert session.state()["matched_faces"] == 1

        def fail(*args, **kwargs):
            raise ApiError("liveness_unavailable", "Liveness inference is unavailable.", 503)

        monkeypatch.setattr(service, "search_all_faces", fail)
        deadline = time.monotonic() + 2.0
        while session.state()["inference_errors"] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        state = session.state()
        assert state["inference_errors"] > 0
        assert state["faces"] == []
        assert state["matched_faces"] == 0
        assert not session._tracks
    finally:
        session.stop()
