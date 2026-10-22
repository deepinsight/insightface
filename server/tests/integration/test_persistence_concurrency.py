from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from insightface_server.app import create_app
from insightface_server.config import Settings
from insightface_server.storage import Database, Repository


def test_database_migration_is_idempotent_and_data_survives_restart(
    make_settings: Callable[..., Settings], image_bytes
) -> None:
    settings = make_settings()
    query = image_bytes(1)
    app = create_app(settings)
    with TestClient(app) as first:
        collection = first.post("/v1/collections", json={"id": "employees", "name": "Employees"})
        person = first.post(
            "/v1/collections/employees/persons",
            data={"id": "alice"},
            files={"images": ("alice.png", query, "image/png")},
        )
        assert collection.status_code == 201
        assert person.status_code == 201
        face_id = person.json()["faces"][0]["id"]
        monitor = first.post(
            "/v1/monitors",
            json={
                "id": "front-gate",
                "name": "Front gate",
                "enabled": False,
                "source": {
                    "type": "rtsp",
                    "url": "rtsp://viewer:secret@camera.test/live",
                },
                "collection_id": "employees",
            },
        )
        assert monitor.status_code == 201

    with TestClient(create_app(settings)) as restarted:
        person = restarted.get("/v1/collections/employees/persons/alice")
        faces = restarted.get("/v1/collections/employees/persons/alice/faces")
        searched = restarted.post(
            "/v1/collections/employees/search",
            data={"threshold": "0.99"},
            files={"image": ("query.png", query, "image/png")},
        )
        status = restarted.get("/v1/system")
        restored_monitor = restarted.get("/v1/monitors/front-gate")

    assert status.status_code == 200
    assert status.json()["database"]["migration_count"] == 9
    assert restored_monitor.status_code == 200
    assert restored_monitor.json()["monitor"]["source"]["url"] == (
        "rtsp://camera.test/live"
    )
    assert "secret" not in restored_monitor.text
    assert status.json()["database"]["quick_check"] == "ok"
    assert person.status_code == 200
    assert person.json()["person"]["face_count"] == 1
    assert len(faces.json()["faces"]) == 1
    assert searched.status_code == 200
    assert searched.json()["matches"] == [
        {
            "person": {
                "id": "alice",
                "name": None,
                "external_id": None,
                "metadata": {},
                "face_count": 1,
                "created_at": person.json()["person"]["created_at"],
                "updated_at": person.json()["person"]["updated_at"],
            },
            "similarity": pytest.approx(1.0),
            "matched_face_id": face_id,
        }
    ]
    runtime_collection = status.json()["search"]["collections"][0]
    assert runtime_collection["collection_id"] == "employees"
    assert runtime_collection["state"] == "ready"
    assert runtime_collection["live_rows"] == 1
    assert runtime_collection["applied_revision"] == 1


def test_sqlite_serializes_concurrent_writers(tmp_path: Path) -> None:
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(tmp_path / "data" / "server.db", migrations)
    database.initialize()
    repository = Repository(database)

    def create(index: int) -> None:
        repository.create_collection(
            {
                "id": f"collection-{index:02d}",
                "name": f"Collection {index}",
                "description": "",
                "default_threshold": 0.68,
                "metadata": {},
                "model_id": "mock",
                "model_digest": "a" * 64,
                "embedding_dimension": 8,
                "preprocessing_version": "1",
            }
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(create, range(30)))

    assert repository.stats() == {
        "collection_count": 30,
        "person_count": 0,
        "face_count": 0,
    }
    assert [item["id"] for item in repository.list_collections("", 100)] == [
        f"collection-{index:02d}" for index in range(30)
    ]


def test_database_process_lock_rejects_second_server_for_same_data_store(
    tmp_path: Path,
) -> None:
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    path = tmp_path / "data" / "server.db"
    first = Database(path, migrations)
    second = Database(path, migrations)

    first.acquire_process_lock()
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            second.acquire_process_lock()
    finally:
        first.release_process_lock()

    # A clean shutdown releases the lease so the next Server can start normally.
    second.acquire_process_lock()
    second.release_process_lock()


def test_concurrent_read_requests_are_stable(client: TestClient, image_bytes) -> None:
    content = image_bytes(17)

    def detect(_: int) -> tuple[int, int]:
        response = client.post(
            "/v1/detect",
            files={"image": ("synthetic.png", content, "image/png")},
        )
        return response.status_code, len(response.json()["faces"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(detect, range(24)))

    assert results == [(200, 1)] * 24
