from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from insightface_server.request_context import REQUEST_DEADLINE
from insightface_server.services.rtsp import (
    MonitorLimitError,
    MonitorManager,
    MonitorOptions,
)


def _manager_payload(monitor_id: str, *, enabled: bool) -> dict[str, Any]:
    return {
        "id": monitor_id,
        "name": monitor_id.title(),
        "description": "Synthetic lifecycle test Monitor",
        "enabled": enabled,
        "source_type": "rtsp",
        "url": f"rtsp://camera.example/{monitor_id}",
        "collection_id": "employees",
        "inference_fps": 2.0,
        "match_threshold": None,
        "event_buffer_size": 100,
        "confirm_frames": 3,
        "absence_timeout_seconds": 3.0,
        "cooldown_seconds": 10.0,
        "emit_unknown": True,
        "preview_enabled": False,
    }


class _ControlledSession:
    def __init__(
        self,
        options: MonitorOptions,
        on_start: Callable[[_ControlledSession], None] | None,
    ) -> None:
        self.options = options
        self.on_start = on_start
        self.start_calls = 0
        self.stop_calls = 0
        self._stopped = False

    def start(self) -> None:
        self.start_calls += 1
        if self.on_start is not None:
            self.on_start(self)

    def stop(self) -> None:
        self.stop_calls += 1
        self._stopped = True

    @property
    def stopped(self) -> bool:
        return self._stopped

    def apply_live_options(
        self,
        *,
        preview_enabled: bool | None = None,
        event_buffer_size: int | None = None,
        name: str | None = None,
        description: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        self.options = replace(
            self.options,
            preview_enabled=(
                self.options.preview_enabled
                if preview_enabled is None
                else preview_enabled
            ),
            event_buffer_size=(
                self.options.event_buffer_size
                if event_buffer_size is None
                else event_buffer_size
            ),
            name=self.options.name if name is None else name,
            description=(
                self.options.description
                if description is None
                else description
            ),
            updated_at=(
                self.options.updated_at if updated_at is None else updated_at
            ),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "status": "stopped" if self.stopped else "starting",
            "connected": False,
            "stream_epoch": self.options.id,
            "last_frame_at": None,
            "last_error": None,
            "preview_active": False,
            "preview_viewers": 0,
        }


class _SessionFactory:
    def __init__(
        self,
        on_start: Callable[[_ControlledSession], None] | None = None,
    ) -> None:
        self.on_start = on_start
        self.sessions: list[_ControlledSession] = []
        self._lock = threading.Lock()

    def __call__(
        self,
        _service,
        options: MonitorOptions,
        **_kwargs: Any,
    ) -> _ControlledSession:
        session = _ControlledSession(options, self.on_start)
        with self._lock:
            self.sessions.append(session)
        return session


def _manager(
    client: TestClient,
    *,
    max_monitors: int,
    factory: _SessionFactory,
) -> MonitorManager:
    manager: MonitorManager = cast(Any, client.app).state.monitors
    manager.max_monitors = max_monitors
    manager.session_factory = cast(Any, factory)
    return manager


def _concurrent(
    operations: list[Callable[[], Any]],
) -> list[Any | BaseException]:
    ready = threading.Barrier(len(operations) + 1)
    results: list[Any | BaseException | None] = [None] * len(operations)

    def run(index: int, operation: Callable[[], Any]) -> None:
        ready.wait()
        try:
            results[index] = operation()
        except BaseException as exc:  # retain worker failures for parent assertions
            results[index] = exc

    threads = [
        threading.Thread(target=run, args=(index, operation), daemon=True)
        for index, operation in enumerate(operations)
    ]
    for thread in threads:
        thread.start()
    ready.wait()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()
    return [result for result in results if result is not None]


def test_concurrent_enabled_creates_cannot_oversubscribe_capacity(
    client: TestClient,
    create_collection,
) -> None:
    create_collection(client)
    start_entered = threading.Event()
    release_start = threading.Event()
    first_start_lock = threading.Lock()
    first_start = True

    def block_first_start(_session: _ControlledSession) -> None:
        nonlocal first_start
        with first_start_lock:
            should_block = first_start
            first_start = False
        if should_block:
            start_entered.set()
            assert release_start.wait(timeout=5.0)

    factory = _SessionFactory(block_first_start)
    manager = _manager(client, max_monitors=1, factory=factory)
    ready = threading.Barrier(3)
    results: list[Any | BaseException | None] = [None, None]

    def create(index: int, monitor_id: str) -> None:
        ready.wait()
        try:
            results[index] = manager.create(
                _manager_payload(monitor_id, enabled=True)
            )
        except BaseException as exc:
            results[index] = exc

    threads = [
        threading.Thread(target=create, args=(0, "north"), daemon=True),
        threading.Thread(target=create, args=(1, "south"), daemon=True),
    ]
    for thread in threads:
        thread.start()
    ready.wait()
    assert start_entered.wait(timeout=5.0)
    release_start.set()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(isinstance(result, MonitorLimitError) for result in results) == 1
    rows = manager.repository.list_monitors("", 10)
    assert len(rows) == 1
    assert rows[0]["enabled"] is True
    assert len(manager._sessions) == 1  # noqa: SLF001 - lifecycle invariant
    assert len(factory.sessions) == 1


def test_concurrent_enables_leave_exactly_one_durable_and_running(
    client: TestClient,
    create_collection,
) -> None:
    create_collection(client)
    factory = _SessionFactory()
    manager = _manager(client, max_monitors=1, factory=factory)
    manager.create(_manager_payload("north", enabled=False))
    manager.create(_manager_payload("south", enabled=False))

    results = _concurrent(
        [
            lambda: manager.update("north", {"enabled": True}),
            lambda: manager.update("south", {"enabled": True}),
        ]
    )

    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(isinstance(result, MonitorLimitError) for result in results) == 1
    rows = manager.repository.list_monitors("", 10)
    assert sum(bool(row["enabled"]) for row in rows) == 1
    assert len(manager._sessions) == 1  # noqa: SLF001 - lifecycle invariant
    running_id = next(iter(manager._sessions))  # noqa: SLF001
    running_row = manager.repository.get_monitor(running_id)
    assert running_row is not None
    assert running_row["enabled"] is True


def test_startup_serializes_capacity_against_a_concurrent_create(
    client: TestClient,
    create_collection,
) -> None:
    create_collection(client)
    app_manager = _manager(
        client,
        max_monitors=1,
        factory=_SessionFactory(),
    )
    app_manager.create(_manager_payload("restored", enabled=False))
    restored = app_manager.repository.update_monitor(
        "restored",
        {"enabled": True},
    )
    assert restored is not None

    start_entered = threading.Event()
    release_start = threading.Event()

    def block_restore(_session: _ControlledSession) -> None:
        start_entered.set()
        assert release_start.wait(timeout=5.0)

    factory = _SessionFactory(block_restore)
    manager = MonitorManager(
        app_manager.service,
        app_manager.repository,
        app_manager.cursors,
        app_manager.credentials,
        max_monitors=1,
        session_factory=cast(Any, factory),
    )
    results: list[Any | BaseException | None] = [None, None]

    def startup() -> None:
        try:
            manager.startup()
            results[0] = "started"
        except BaseException as exc:
            results[0] = exc

    def create() -> None:
        try:
            results[1] = manager.create(
                _manager_payload("late", enabled=True)
            )
        except BaseException as exc:
            results[1] = exc

    startup_thread = threading.Thread(target=startup, daemon=True)
    startup_thread.start()
    assert start_entered.wait(timeout=5.0)
    create_thread = threading.Thread(target=create, daemon=True)
    create_thread.start()
    release_start.set()
    for thread in (startup_thread, create_thread):
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    assert results[0] == "started"
    assert isinstance(results[1], MonitorLimitError)
    assert app_manager.repository.get_monitor("late") is None
    assert set(manager._sessions) == {"restored"}  # noqa: SLF001
    manager.close()


def test_restart_and_delete_are_serialized_without_leaking_either_session(
    client: TestClient,
    create_collection,
) -> None:
    create_collection(client)
    replacement_entered = threading.Event()
    release_replacement = threading.Event()

    def block_replacement(session: _ControlledSession) -> None:
        if session.options.url.endswith("/replacement"):
            replacement_entered.set()
            assert release_replacement.wait(timeout=5.0)

    factory = _SessionFactory(block_replacement)
    manager = _manager(client, max_monitors=1, factory=factory)
    manager.create(_manager_payload("front-gate", enabled=True))
    results: list[Any | BaseException | None] = [None, None]

    def update() -> None:
        try:
            results[0] = manager.update(
                "front-gate",
                {"url": "rtsp://camera.example/replacement"},
            )
        except BaseException as exc:
            results[0] = exc

    delete_started = threading.Event()

    def delete() -> None:
        delete_started.set()
        try:
            manager.delete("front-gate")
            results[1] = "deleted"
        except BaseException as exc:
            results[1] = exc

    update_thread = threading.Thread(target=update, daemon=True)
    update_thread.start()
    assert replacement_entered.wait(timeout=5.0)
    delete_thread = threading.Thread(target=delete, daemon=True)
    delete_thread.start()
    assert delete_started.wait(timeout=5.0)
    release_replacement.set()
    for thread in (update_thread, delete_thread):
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    assert isinstance(results[0], dict)
    assert results[1] == "deleted"
    assert manager.repository.get_monitor("front-gate") is None
    assert "front-gate" not in manager._sessions  # noqa: SLF001
    assert len(factory.sessions) == 2
    assert all(session.stopped for session in factory.sessions)


def test_create_start_failure_removes_durable_row_and_candidate(
    client: TestClient,
    create_collection,
) -> None:
    create_collection(client)

    def fail_start(_session: _ControlledSession) -> None:
        raise RuntimeError("synthetic start failure")

    factory = _SessionFactory(fail_start)
    manager = _manager(client, max_monitors=1, factory=factory)

    with pytest.raises(RuntimeError, match="synthetic start failure"):
        manager.create(_manager_payload("front-gate", enabled=True))

    assert manager.repository.get_monitor("front-gate") is None
    assert "front-gate" not in manager._sessions  # noqa: SLF001
    assert len(factory.sessions) == 1
    assert factory.sessions[0].stopped


def test_create_start_failure_compensates_after_request_deadline_expires(
    client: TestClient,
    create_collection,
) -> None:
    create_collection(client)

    def expire_request_then_fail(_session: _ControlledSession) -> None:
        REQUEST_DEADLINE.set(time.monotonic() - 1.0)
        raise RuntimeError("synthetic delayed start failure")

    factory = _SessionFactory(expire_request_then_fail)
    manager = _manager(client, max_monitors=1, factory=factory)
    deadline_token = REQUEST_DEADLINE.set(time.monotonic() + 5.0)
    try:
        with pytest.raises(RuntimeError, match="synthetic delayed start failure"):
            manager.create(_manager_payload("front-gate", enabled=True))
    finally:
        REQUEST_DEADLINE.reset(deadline_token)

    assert manager.repository.get_monitor("front-gate") is None
    assert "front-gate" not in manager._sessions  # noqa: SLF001
    assert len(factory.sessions) == 1
    assert factory.sessions[0].stopped


def test_restart_start_failure_preserves_durable_config_and_old_session(
    client: TestClient,
    create_collection,
) -> None:
    create_collection(client)

    def fail_replacement(session: _ControlledSession) -> None:
        if session.options.url.endswith("/replacement"):
            raise RuntimeError("synthetic replacement failure")

    factory = _SessionFactory(fail_replacement)
    manager = _manager(client, max_monitors=1, factory=factory)
    manager.create(_manager_payload("front-gate", enabled=True))
    old_session = manager._sessions["front-gate"]  # noqa: SLF001

    with pytest.raises(RuntimeError, match="synthetic replacement failure"):
        manager.update(
            "front-gate",
            {"url": "rtsp://camera.example/replacement"},
        )

    row = manager.repository.get_monitor("front-gate")
    assert row is not None
    assert manager.credentials.decrypt(
        row["source_url_ciphertext"],
        scope="monitor:front-gate",
    ) == "rtsp://camera.example/front-gate"
    assert manager._sessions["front-gate"] is old_session  # noqa: SLF001
    assert not old_session.stopped
    assert len(factory.sessions) == 2
    assert factory.sessions[1].stopped


def test_restart_database_failure_stops_candidate_and_preserves_old_session(
    client: TestClient,
    create_collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_collection(client)
    factory = _SessionFactory()
    manager = _manager(client, max_monitors=1, factory=factory)
    manager.create(_manager_payload("front-gate", enabled=True))
    old_session = manager._sessions["front-gate"]  # noqa: SLF001

    def fail_update(_monitor_id: str, _changes: dict[str, Any]):
        raise RuntimeError("synthetic database failure")

    monkeypatch.setattr(manager.repository, "update_monitor", fail_update)

    with pytest.raises(RuntimeError, match="synthetic database failure"):
        manager.update(
            "front-gate",
            {"url": "rtsp://camera.example/replacement"},
        )

    row = manager.repository.get_monitor("front-gate")
    assert row is not None
    assert manager.credentials.decrypt(
        row["source_url_ciphertext"],
        scope="monitor:front-gate",
    ) == "rtsp://camera.example/front-gate"
    assert manager._sessions["front-gate"] is old_session  # noqa: SLF001
    assert not old_session.stopped
    assert len(factory.sessions) == 2
    assert factory.sessions[1].stopped


def test_close_waits_for_inflight_create_and_prevents_late_session_publish(
    client: TestClient,
    create_collection,
) -> None:
    create_collection(client)
    start_entered = threading.Event()
    release_start = threading.Event()

    def block_start(_session: _ControlledSession) -> None:
        start_entered.set()
        assert release_start.wait(timeout=5.0)

    factory = _SessionFactory(block_start)
    manager = _manager(client, max_monitors=1, factory=factory)
    results: list[Any | BaseException | None] = [None, None]

    def create() -> None:
        try:
            results[0] = manager.create(
                _manager_payload("front-gate", enabled=True)
            )
        except BaseException as exc:
            results[0] = exc

    def close() -> None:
        try:
            manager.close()
            results[1] = "closed"
        except BaseException as exc:
            results[1] = exc

    create_thread = threading.Thread(target=create, daemon=True)
    create_thread.start()
    assert start_entered.wait(timeout=5.0)
    close_thread = threading.Thread(target=close, daemon=True)
    close_thread.start()
    release_start.set()
    for thread in (create_thread, close_thread):
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    assert isinstance(results[0], dict)
    assert results[1] == "closed"
    assert not manager._sessions  # noqa: SLF001
    assert factory.sessions[0].stopped
    with pytest.raises(RuntimeError, match="closed"):
        manager.create(_manager_payload("too-late", enabled=True))
