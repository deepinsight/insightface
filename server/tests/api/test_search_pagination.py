from __future__ import annotations

from fastapi.testclient import TestClient


def test_search_uses_best_face_per_person_and_sorts_by_similarity(
    client: TestClient, create_collection, create_person, image_bytes
) -> None:
    create_collection(client, threshold=0.0)
    query = image_bytes(10)
    alice = create_person(client, "employees", "alice", seed=10)
    exact_face_id = alice["faces"][0]["id"]
    create_person(client, "employees", "bob", seed=11)
    # Add an inferior second sample to Alice; the person score must still use her exact face.
    client.post(
        "/v1/collections/employees/persons/alice/faces",
        files={"images": ("alternate.png", image_bytes(12), "image/png")},
    )

    response = client.post(
        "/v1/collections/employees/search",
        data={"limit": "5", "threshold": "0"},
        files={"image": ("query.png", query, "image/png")},
    )

    assert response.status_code == 200
    matches = response.json()["matches"]
    assert [item["person"]["id"] for item in matches] == ["alice", "bob"]
    assert matches[0]["similarity"] == 1.0
    assert matches[0]["matched_face_id"] == exact_face_id
    assert matches[0]["similarity"] >= matches[1]["similarity"]


def test_search_threshold_limit_and_no_face_behavior(
    client: TestClient, create_collection, create_person, image_bytes
) -> None:
    create_collection(client, threshold=0.0)
    create_person(client, "employees", "alice", seed=1)
    create_person(client, "employees", "bob", seed=2)

    limited = client.post(
        "/v1/collections/employees/search",
        data={"limit": "1", "threshold": "0"},
        files={"image": ("query.png", image_bytes(1), "image/png")},
    )
    none = client.post(
        "/v1/collections/employees/search",
        data={"threshold": "0.999999"},
        files={"image": ("other.png", image_bytes(99), "image/png")},
    )
    no_face = client.post(
        "/v1/collections/employees/search",
        files={"image": ("blank.png", image_bytes(blank=True), "image/png")},
    )

    assert limited.status_code == 200
    assert len(limited.json()["matches"]) == 1
    assert limited.json()["matches"][0]["person"]["id"] == "alice"
    assert none.status_code == 200
    assert none.json()["matches"] == []
    assert no_face.status_code == 422
    assert no_face.json()["error"]["code"] == "face_not_found"


def test_deleted_face_is_immediately_absent_from_search(
    client: TestClient, create_collection, create_person, image_bytes
) -> None:
    create_collection(client)
    query = image_bytes(23)
    created = create_person(client, "employees", "alice", seed=23)
    face_id = created["faces"][0]["id"]

    before = client.post(
        "/v1/collections/employees/search",
        data={"threshold": "0.99"},
        files={"image": ("query.png", query, "image/png")},
    )
    deleted = client.delete(
        f"/v1/collections/employees/persons/alice/faces/{face_id}"
    )
    after = client.post(
        "/v1/collections/employees/search",
        data={"threshold": "0.99"},
        files={"image": ("query.png", query, "image/png")},
    )

    assert before.status_code == 200
    assert before.json()["matches"][0]["matched_face_id"] == face_id
    assert deleted.status_code == 204
    assert after.status_code == 200
    assert after.json()["matches"] == []


def test_collection_cursor_is_opaque_paginated_and_tamper_evident(
    client: TestClient, create_collection
) -> None:
    for collection_id in ("alpha", "bravo", "charlie"):
        create_collection(client, collection_id)

    first = client.get("/v1/collections?limit=1")
    token = first.json()["next_cursor"]
    second = client.get("/v1/collections", params={"limit": 1, "cursor": token})
    payload, signature = token.split(".", 1)
    tampered_signature = ("A" if signature[0] != "A" else "B") + signature[1:]
    tampered = f"{payload}.{tampered_signature}"
    rejected = client.get("/v1/collections", params={"cursor": tampered})

    assert [item["id"] for item in first.json()["collections"]] == ["alpha"]
    assert [item["id"] for item in second.json()["collections"]] == ["bravo"]
    assert "alpha" not in token
    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "invalid_cursor"


def test_cursor_cannot_be_reused_across_scopes(
    client: TestClient, create_collection, create_person
) -> None:
    create_collection(client)
    create_person(client, "employees", "alpha", seed=1)
    create_person(client, "employees", "bravo", seed=2)
    first = client.get("/v1/collections/employees/persons?limit=1")
    token = first.json()["next_cursor"]

    wrong_search = client.get(
        "/v1/collections/employees/persons",
        params={"limit": 1, "search": "a", "cursor": token},
    )
    wrong_collection = client.get("/v1/collections", params={"limit": 1, "cursor": token})

    assert token
    assert wrong_search.status_code == 400
    assert wrong_search.json()["error"]["code"] == "invalid_cursor"
    assert wrong_collection.status_code == 400
    assert wrong_collection.json()["error"]["code"] == "invalid_cursor"
