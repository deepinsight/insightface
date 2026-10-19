from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, cast

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from insightface_server.config import Settings
from insightface_server.services import rtsp
from insightface_server.services.rtsp import (
    FFMPEG_CAPTURE_OPTIONS_ENV,
    MonitorOptions,
    MonitorSession,
    configure_rtsp_transport,
    effective_rtsp_transport,
    opencv_inverts_capture_options,
    rtsp_capture_options,
)

SECRET_URL = "rtsp://camera-user:s3cret-pass@10.0.0.20:554/cam/realmonitor?channel=20&token=qtok"


def _options(**changes: Any) -> MonitorOptions:
    values: dict[str, Any] = {
        "id": "gate",
        "name": "Gate",
        "description": "",
        "enabled": True,
        "url": "rtsp://camera.example/live",
        "collection_id": "employees",
        "inference_fps": 5.0,
        "match_threshold": None,
        "event_buffer_size": 20,
        "confirm_frames": 2,
        "absence_timeout_seconds": 1.0,
        "cooldown_seconds": 0.0,
        "emit_unknown": True,
        "preview_enabled": False,
        "created_at": "2026-10-15T00:00:00Z",
        "updated_at": "2026-10-15T00:00:00Z",
    }
    values.update(changes)
    return MonitorOptions(**values)


class _NoFaces:
    def search_all_faces(self, *_args: Any, **_kwargs: Any):
        return [], 0.4


class _Capture:
    """Fake capture: delivers `frames` frames, then fails reads."""

    def __init__(
        self,
        *,
        opened: bool = True,
        frames: int | None = None,
        read_delay: float = 0.002,
    ) -> None:
        self.opened = opened
        self.remaining = frames
        self.read_delay = read_delay
        self.released = False

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def get(self, property_id: int) -> float:
        return 15.0 if property_id == cv2.CAP_PROP_FPS else 0.0

    def read(self):
        time.sleep(self.read_delay)
        if self.released or self.remaining == 0:
            return False, None
        if self.remaining is not None:
            self.remaining -= 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def _session(factory, **changes: Any) -> MonitorSession:
    options = _options(**changes.pop("options", {}))
    values: dict[str, Any] = {
        "max_faces": 10,
        "preview_fps": 5.0,
        "jpeg_quality": 80,
        "open_timeout_seconds": 1.0,
        "read_timeout_seconds": 1.0,
        "reconnect_delay_seconds": 0.01,
        "capture_factory": factory,
        "transport": "tcp",
    }
    values.update(changes)
    return MonitorSession(cast(Any, _NoFaces()), options, **values)


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@pytest.fixture(autouse=True)
def _fresh_install_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtsp, "_installed_capture_options", None)


@pytest.mark.parametrize(
    ("version", "inverted"),
    [("4.10.0", False), ("4.11.0", True), ("4.11.1", True), ("4.12.0", False), ("5.0.0", False)],
)
def test_only_opencv_4_11_inverts_capture_options(version: str, inverted: bool) -> None:
    assert opencv_inverts_capture_options(version) is inverted


@pytest.mark.parametrize(
    ("transport", "version", "value", "effective"),
    [
        # Any non-empty value makes 4.11 prefer TCP; the absent value is UDP-first.
        ("tcp", "4.11.0", "rtsp_transport;tcp", "tcp_preferred"),
        ("udp", "4.11.0", None, "udp_preferred"),
        # Fixed releases parse the value literally and do not fall back.
        ("tcp", "4.12.0", "rtsp_transport;tcp", "tcp"),
        ("udp", "4.12.0", "rtsp_transport;udp", "udp"),
    ],
)
def test_transport_maps_to_capture_options(
    transport: str, version: str, value: str | None, effective: str
) -> None:
    assert rtsp_capture_options(transport, opencv_version=version) == value
    assert effective_rtsp_transport(transport, opencv_version=version) == effective


@pytest.mark.parametrize("transport", ["http", "TCP", "", "udp_multicast"])
def test_invalid_transport_is_rejected(transport: str) -> None:
    with pytest.raises(ValueError, match="tcp, udp"):
        rtsp_capture_options(transport)
    with pytest.raises(ValueError, match="tcp, udp"):
        configure_rtsp_transport(transport, environ={})


def test_configure_installs_and_switches_its_own_value() -> None:
    environ: dict[str, str] = {}
    assert configure_rtsp_transport("tcp", environ=environ, opencv_version="4.11.0") == (
        "tcp_preferred"
    )
    assert environ[FFMPEG_CAPTURE_OPTIONS_ENV] == "rtsp_transport;tcp"

    # UDP on 4.11 requires the variable to be absent; our own value is removable.
    assert configure_rtsp_transport("udp", environ=environ, opencv_version="4.11.0") == (
        "udp_preferred"
    )
    assert FFMPEG_CAPTURE_OPTIONS_ENV not in environ

    assert configure_rtsp_transport("udp", environ=environ, opencv_version="4.12.0") == "udp"
    assert environ[FFMPEG_CAPTURE_OPTIONS_ENV] == "rtsp_transport;udp"


def test_configure_treats_empty_variable_as_unset() -> None:
    environ = {FFMPEG_CAPTURE_OPTIONS_ENV: ""}
    configure_rtsp_transport("tcp", environ=environ, opencv_version="4.12.0")
    assert environ[FFMPEG_CAPTURE_OPTIONS_ENV] == "rtsp_transport;tcp"


def test_configure_respects_operator_override(caplog: pytest.LogCaptureFixture) -> None:
    environ = {FFMPEG_CAPTURE_OPTIONS_ENV: "rtsp_transport;udp|stimeout;1"}
    with caplog.at_level(logging.WARNING, logger="insightface_server.monitors"):
        effective = configure_rtsp_transport("tcp", environ=environ, opencv_version="4.12.0")
    assert effective == "custom"
    assert environ[FFMPEG_CAPTURE_OPTIONS_ENV] == "rtsp_transport;udp|stimeout;1"
    assert "rtsp_transport_override" in caplog.text

    # On 4.11 any non-empty override is reduced to prefer_tcp by OpenCV.
    assert (
        configure_rtsp_transport("udp", environ=environ, opencv_version="4.11.0")
        == "tcp_preferred"
    )


def test_settings_default_to_tcp_and_validate_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INSIGHTFACE_RTSP_TRANSPORT", raising=False)
    assert Settings.from_env().rtsp_transport == "tcp"
    monkeypatch.setenv("INSIGHTFACE_RTSP_TRANSPORT", " UDP ")
    assert Settings.from_env().rtsp_transport == "udp"
    monkeypatch.setenv("INSIGHTFACE_RTSP_TRANSPORT", "rtp")
    with pytest.raises(ValueError, match="INSIGHTFACE_RTSP_TRANSPORT must be one of: tcp, udp"):
        Settings.from_env()


def test_app_startup_installs_transport_before_monitors(
    make_settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from insightface_server.app import create_app

    monkeypatch.delenv(FFMPEG_CAPTURE_OPTIONS_ENV, raising=False)
    with TestClient(create_app(make_settings(rtsp_transport="tcp"))) as client:
        manager = cast(Any, client.app).state.monitors
        assert os.environ[FFMPEG_CAPTURE_OPTIONS_ENV] == "rtsp_transport;tcp"
        assert manager.transport == effective_rtsp_transport("tcp")


def test_open_capture_passes_timeouts_to_ffmpeg_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []

    def fake_video_capture(*args: Any) -> str:
        calls.append(args)
        return "capture"

    monkeypatch.setattr(rtsp.cv2, "VideoCapture", fake_video_capture)
    assert rtsp._open_capture("rtsp://cam/live", 4000, 3000) == "capture"
    assert calls == [
        (
            "rtsp://cam/live",
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000],
        )
    ]


def test_open_capture_falls_back_to_property_setters(monkeypatch: pytest.MonkeyPatch) -> None:
    class LegacyCapture:
        instance: LegacyCapture | None = None

        def __init__(self, *args: Any) -> None:
            if args:
                raise TypeError("params unsupported")
            self.properties: dict[int, float] = {}
            self.opened: tuple[Any, ...] | None = None
            LegacyCapture.instance = self

        def set(self, prop: int, value: float) -> bool:
            self.properties[prop] = value
            return True

        def open(self, *args: Any) -> bool:
            self.opened = args
            return True

    monkeypatch.setattr(rtsp.cv2, "VideoCapture", LegacyCapture)
    capture = rtsp._open_capture("rtsp://cam/live", 4000, 3000)
    assert capture is LegacyCapture.instance
    assert capture.properties == {
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC: 4000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC: 3000,
    }
    assert capture.opened == ("rtsp://cam/live", cv2.CAP_FFMPEG)


def test_concurrent_sessions_open_with_one_stable_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(FFMPEG_CAPTURE_OPTIONS_ENV, raising=False)  # restored at teardown
    configure_rtsp_transport("tcp")
    observed: list[str | None] = []
    lock = threading.Lock()
    barrier = threading.Barrier(4, timeout=3.0)

    def fake_video_capture(*_args: Any) -> _Capture:
        barrier.wait()  # all four opens are in flight at once
        with lock:
            observed.append(os.environ.get(FFMPEG_CAPTURE_OPTIONS_ENV))
        return _Capture()

    monkeypatch.setattr(rtsp.cv2, "VideoCapture", fake_video_capture)
    sessions = [
        _session(rtsp._open_capture, options={"id": f"cam-{index}"}) for index in range(4)
    ]
    try:
        for session in sessions:
            session.start()
        assert _wait_for(lambda: all(session.connected for session in sessions))
    finally:
        for session in sessions:
            session.stop()
    assert observed == ["rtsp_transport;tcp"] * 4


def test_failures_reconnect_release_captures_and_resume_frames() -> None:
    first = _Capture(frames=3)  # decoder gives up after packet loss
    rejected = _Capture(opened=False)
    healthy = _Capture()
    sequence = iter([first, None, rejected, healthy])
    created: list[_Capture] = []

    def factory(*_args: Any):
        capture = next(sequence)
        if capture is not None:
            created.append(capture)
        return capture

    session = _session(factory)
    session.start()
    try:
        assert _wait_for(lambda: session.connected and session.state()["reconnects"] == 3)
        assert _wait_for(lambda: session.state()["inference"]["decoded_frames"] > 5)
        state = session.state()
        events = session.event_page(cursor_epoch=None, after_sequence=None, limit=20)
    finally:
        session.stop()

    assert state["status"] == "running"
    assert state["last_error"] is None
    assert [event["type"] for event in events["events"]] == [
        "monitor_error",
        "monitor_recovered",
    ]
    assert all(capture.released for capture in created)


def test_backoff_grows_when_source_opens_but_never_delivers_frames() -> None:
    opened_at: list[float] = []

    def factory(*_args: Any) -> _Capture:
        opened_at.append(time.monotonic())
        return _Capture(frames=0, read_delay=0.0)

    session = _session(factory, reconnect_delay_seconds=0.02)
    session.start()
    try:
        assert _wait_for(lambda: len(opened_at) >= 5)
        reader_threads = [
            thread for thread in threading.enumerate() if thread.name == "monitor-reader-gate"
        ]
    finally:
        session.stop()

    gaps = [later - earlier for earlier, later in zip(opened_at, opened_at[1:], strict=False)]
    # 0.02, 0.04, 0.08, 0.16 ... instead of a fixed 0.02 retry loop.
    assert gaps[3] >= 0.12
    assert len(reader_threads) == 1


def test_stop_is_responsive_during_reconnect_backoff() -> None:
    attempts: list[int] = []

    def factory(*_args: Any) -> None:
        attempts.append(1)
        return None

    session = _session(factory, reconnect_delay_seconds=10.0)
    session.start()
    assert _wait_for(lambda: attempts and session.state()["reconnects"] == 1)
    started = time.monotonic()
    session.stop()
    assert time.monotonic() - started < 1.0
    assert session._reader is not None and not session._reader.is_alive()
    assert session.state()["status"] == "stopped"


def test_logs_label_read_timeout_and_never_include_credentials(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stalled = _Capture(frames=2)
    original_read = stalled.read

    def read():
        if stalled.remaining == 0:
            time.sleep(0.06)  # longer than the 50 ms read timeout
            return False, None
        return original_read()

    stalled.read = read  # type: ignore[method-assign]
    sequence = iter([None, stalled, _Capture()])
    session = _session(
        lambda *_args: next(sequence),
        options={"url": SECRET_URL},
        read_timeout_seconds=0.05,
    )
    with caplog.at_level(logging.INFO, logger="insightface_server.monitors"):
        session.start()
        try:
            assert _wait_for(lambda: session.state()["reconnects"] == 2 and session.connected)
            assert _wait_for(lambda: caplog.text.count("monitor_source_connected") == 2)
        finally:
            session.stop()

    text = caplog.text
    assert "stage=open" in text
    assert "stage=read_timeout" in text
    assert "transport=tcp" in text
    assert "width=64 height=48 source_fps=15.0" in text
    assert "failed_attempts=1" in text
    assert "rtsp://10.0.0.20:554/cam/realmonitor" in text
    for secret in ("camera-user", "s3cret-pass", "qtok", "channel=20"):
        assert secret not in text
