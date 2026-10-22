from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient
from insightface_server.services.rtsp import MonitorSession


def _monitor_payload(**overrides):
    payload = {
        "id": "front-gate",
        "name": "Front gate",
        "description": "Synthetic API test Monitor",
        "enabled": False,
        "source": {
            "type": "rtsp",
            "url": "rtsp://viewer:secret@192.168.1.20:8554/live?token=private",
        },
        "collection_id": "employees",
        "inference_fps": 2.0,
        "match_threshold": None,
        "event_buffer_size": 1000,
        "event_policy": {
            "confirm_frames": 3,
            "absence_timeout_seconds": 3,
            "cooldown_seconds": 10,
            "emit_unknown": True,
        },
        "preview_enabled": False,
    }
    payload.update(overrides)
    return payload


def test_monitor_crud_persists_config_without_exposing_credentials(
    client: TestClient, create_collection
) -> None:
    create_collection(client)

    created = client.post("/v1/monitors", json=_monitor_payload())

    assert created.status_code == 201, created.text
    monitor = created.json()["monitor"]
    assert monitor["id"] == "front-gate"
    assert monitor["source"] == {
        "type": "rtsp",
        "url": "rtsp://192.168.1.20:8554/live",
    }
    assert monitor["preview_enabled"] is False
    assert monitor["runtime"]["status"] == "stopped"
    assert "secret" not in created.text
    assert "private" not in created.text

    database_path = client.app.state.database.path
    assert b"viewer" not in database_path.read_bytes()
    assert b"secret" not in database_path.read_bytes()
    with sqlite3.connect(database_path) as connection:
        stored = connection.execute(
            "SELECT source_url_ciphertext FROM monitors WHERE id='front-gate'"
        ).fetchone()[0]
    assert stored.startswith("v1.")

    listed = client.get("/v1/monitors")
    fetched = client.get("/v1/monitors/front-gate")
    patched = client.patch(
        "/v1/monitors/front-gate",
        json={
            "name": "Main entrance",
            "event_buffer_size": 500,
            "event_policy": {"confirm_frames": 5},
            "preview_enabled": True,
        },
    )
    state = client.get("/v1/monitors/front-gate/state")
    events = client.get("/v1/monitors/front-gate/events", params={"limit": 25})

    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["monitors"]] == ["front-gate"]
    assert fetched.status_code == 200
    assert patched.status_code == 200
    assert patched.json()["monitor"]["name"] == "Main entrance"
    assert patched.json()["monitor"]["event_buffer_size"] == 500
    assert patched.json()["monitor"]["event_policy"] == {
        "confirm_frames": 5,
        "absence_timeout_seconds": 3.0,
        "cooldown_seconds": 10.0,
        "emit_unknown": True,
    }
    assert patched.json()["monitor"]["preview_enabled"] is True
    assert state.status_code == 200
    assert state.json()["state"]["status"] == "stopped"
    assert state.json()["state"]["preview"]["enabled"] is True
    assert events.status_code == 200
    assert events.json()["events"] == []
    assert isinstance(events.json()["next_cursor"], str)

    deleted = client.delete("/v1/monitors/front-gate")
    assert deleted.status_code == 204
    assert client.get("/v1/monitors/front-gate").status_code == 404


def test_monitor_validation_conflicts_and_collection_delete_guard(
    client: TestClient, create_collection
) -> None:
    create_collection(client)

    invalid_scheme = client.post(
        "/v1/monitors",
        json=_monitor_payload(source={"type": "rtsp", "url": "http://camera/live"}),
    )
    missing_collection = client.post(
        "/v1/monitors",
        json=_monitor_payload(collection_id="missing"),
    )
    created = client.post("/v1/monitors", json=_monitor_payload())
    duplicate = client.post("/v1/monitors", json=_monitor_payload())
    collection_delete = client.delete(
        "/v1/collections/employees",
        params={"force": "true"},
    )
    empty_patch = client.patch("/v1/monitors/front-gate", json={})
    null_collection = client.patch(
        "/v1/monitors/front-gate",
        json={"collection_id": None},
    )
    empty_policy = client.patch(
        "/v1/monitors/front-gate",
        json={"event_policy": {}},
    )

    assert invalid_scheme.status_code == 400
    assert invalid_scheme.json()["error"]["code"] == "invalid_request"
    assert missing_collection.status_code == 404
    assert missing_collection.json()["error"]["code"] == "collection_not_found"
    assert created.status_code == 201
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "monitor_exists"
    assert collection_delete.status_code == 409
    assert collection_delete.json()["error"]["code"] == "collection_in_use"
    assert empty_patch.status_code == 400
    assert null_collection.status_code == 400
    assert empty_policy.status_code == 400


def test_monitor_preview_has_explicit_disabled_and_unavailable_errors(
    client: TestClient, create_collection
) -> None:
    create_collection(client)
    client.post("/v1/monitors", json=_monitor_payload())

    disabled = client.get("/v1/monitors/front-gate/preview.mjpeg")
    client.patch(
        "/v1/monitors/front-gate",
        json={"preview_enabled": True},
    )
    unavailable = client.get("/v1/monitors/front-gate/preview.mjpeg")

    assert disabled.status_code == 409
    assert disabled.json()["error"]["code"] == "preview_disabled"
    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "stream_unavailable"


def test_monitor_event_cursor_is_incremental_signed_and_epoch_aware(
    client: TestClient, create_collection
) -> None:
    create_collection(client)
    client.post("/v1/monitors", json=_monitor_payload())
    manager = client.app.state.monitors
    row = client.app.state.repository.get_monitor("front-gate")
    assert row is not None
    session = MonitorSession(
        manager.service,
        manager._options(row),
        max_faces=manager.max_faces,
        preview_fps=manager.preview_fps,
        jpeg_quality=manager.jpeg_quality,
        open_timeout_seconds=1.0,
        read_timeout_seconds=1.0,
        reconnect_delay_seconds=0.1,
        capture_factory=lambda *_args: None,
    )
    manager._sessions["front-gate"] = session
    with session._condition:
        for index in range(3):
            session._append_event(f"test_{index}", now=float(index))

    first = client.get("/v1/monitors/front-gate/events", params={"limit": 2})
    cursor = first.json()["next_cursor"]
    with session._condition:
        session._append_event("test_3", now=3.0)
    incremental = client.get(
        "/v1/monitors/front-gate/events",
        params={"limit": 10, "cursor": cursor},
    )
    invalid = client.get(
        "/v1/monitors/front-gate/events",
        params={"cursor": f"{cursor}altered"},
    )
    with session._condition:
        session.stream_epoch = "replacement-epoch"
        session._events.clear()
        session._event_sequence = 0
        session._append_event("test_new", now=4.0)
    reset = client.get(
        "/v1/monitors/front-gate/events",
        params={"limit": 1, "cursor": incremental.json()["next_cursor"]},
    )

    assert [item["sequence"] for item in first.json()["events"]] == [2, 3]
    assert [item["sequence"] for item in incremental.json()["events"]] == [4]
    assert incremental.json()["stream_reset"] is False
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "invalid_cursor"
    assert reset.status_code == 200
    assert reset.json()["stream_reset"] is True
    assert [item["sequence"] for item in reset.json()["events"]] == [1]


def test_monitor_routes_replace_ephemeral_rtsp_routes_in_openapi(
    client: TestClient,
) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for path in (
        "/v1/monitors",
        "/v1/monitors/{monitor_id}",
        "/v1/monitors/{monitor_id}/state",
        "/v1/monitors/{monitor_id}/events",
        "/v1/monitors/{monitor_id}/preview.mjpeg",
    ):
        assert path in paths
    assert "/v1/streams/rtsp" not in paths
