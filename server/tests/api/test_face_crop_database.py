from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient
from insightface_server.app import create_app
from insightface_server.config import Settings


def _create_collection(
    client: TestClient, collection_id: str, *, save_face_crops: bool | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {"id": collection_id, "name": collection_id}
    if save_face_crops is not None:
        payload["save_face_crops"] = save_face_crops
    response = client.post("/v1/collections", json=payload)
    assert response.status_code == 201, response.text
    return response.json()["collection"]


def _create_person(
    client: TestClient,
    image_bytes: Callable[..., bytes],
    collection_id: str,
    person_id: str,
    *,
    seed: int,
    headers: dict[str, str] | None = None,
) -> dict[str, object]:
    response = client.post(
        f"/v1/collections/{collection_id}/persons",
        data={"id": person_id},
        files={"images": (f"{person_id}.png", image_bytes(seed), "image/png")},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_collection_crop_setting_defaults_and_can_be_changed(
    client: TestClient,
) -> None:
    inherited = _create_collection(client, "inherited")
    explicit = _create_collection(client, "explicit", save_face_crops=True)

    enabled = client.patch(
        "/v1/collections/inherited", json={"save_face_crops": True}
    )
    disabled = client.patch(
        "/v1/collections/explicit", json={"save_face_crops": False}
    )
    invalid = client.patch(
        "/v1/collections/explicit", json={"save_face_crops": None}
    )

    assert inherited["save_face_crops"] is False
    assert explicit["save_face_crops"] is True
    assert enabled.status_code == 200
    assert enabled.json()["collection"]["save_face_crops"] is True
    assert disabled.status_code == 200
    assert disabled.json()["collection"]["save_face_crops"] is False
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "invalid_request"


def test_collection_crop_default_can_be_enabled_by_deployment_setting(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(save_face_crops=True)
    with TestClient(create_app(settings)) as client:
        collection = _create_collection(client, "inherited")

    assert collection["save_face_crops"] is True


def test_saved_crop_is_private_jpeg_and_disabling_only_affects_new_faces(
    client: TestClient, image_bytes: Callable[..., bytes]
) -> None:
    _create_collection(client, "employees", save_face_crops=True)
    created = _create_person(
        client, image_bytes, "employees", "alice", seed=101
    )
    first = created["faces"][0]
    first_id = first["id"]
    first_url = f"/v1/collections/employees/persons/alice/faces/{first_id}/image"

    downloaded = client.get(first_url)
    listed = client.get("/v1/collections/employees/persons/alice/faces")
    disabled = client.patch(
        "/v1/collections/employees", json={"save_face_crops": False}
    )
    added = client.post(
        "/v1/collections/employees/persons/alice/faces",
        files={"images": ("alice-2.png", image_bytes(102), "image/png")},
    )
    second = added.json()["faces"][0]
    missing = client.get(
        f"/v1/collections/employees/persons/alice/faces/{second['id']}/image"
    )
    old_after_disable = client.get(first_url)

    assert first["has_crop"] is True
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "image/jpeg"
    assert downloaded.headers["cache-control"] == "no-store"
    assert downloaded.headers["x-request-id"]
    assert downloaded.content.startswith(b"\xff\xd8")
    assert downloaded.content.endswith(b"\xff\xd9")
    assert listed.status_code == 200
    assert listed.json()["faces"][0]["has_crop"] is True
    assert all(
        "crop_image" not in face and "crop_path" not in face
        for face in listed.json()["faces"]
    )
    assert disabled.status_code == 200
    assert second["has_crop"] is False
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "face_image_not_found"
    assert old_after_disable.status_code == 200
    assert old_after_disable.content == downloaded.content


def test_disabled_collection_does_not_store_a_crop(
    client: TestClient, image_bytes: Callable[..., bytes]
) -> None:
    _create_collection(client, "employees")
    created = _create_person(client, image_bytes, "employees", "alice", seed=103)
    face = created["faces"][0]

    response = client.get(
        f"/v1/collections/employees/persons/alice/faces/{face['id']}/image"
    )
    unknown = client.get(
        "/v1/collections/employees/persons/alice/faces/unknown-face/image"
    )

    assert face["has_crop"] is False
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "face_image_not_found"
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "face_not_found"


def test_face_image_requires_bearer_authentication(
    make_settings: Callable[..., Settings], image_bytes: Callable[..., bytes]
) -> None:
    settings = make_settings(auth_enabled=True, startup_api_key="test-secret")
    headers = {"Authorization": "Bearer test-secret"}
    with TestClient(create_app(settings)) as client:
        collection = client.post(
            "/v1/collections",
            json={"id": "employees", "name": "Employees", "save_face_crops": True},
            headers=headers,
        )
        assert collection.status_code == 201, collection.text
        created = _create_person(
            client,
            image_bytes,
            "employees",
            "alice",
            seed=104,
            headers=headers,
        )
        face_id = created["faces"][0]["id"]
        url = f"/v1/collections/employees/persons/alice/faces/{face_id}/image"

        unauthenticated = client.get(url)
        authenticated = client.get(url, headers=headers)

    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 200
    assert authenticated.headers["cache-control"] == "no-store"


def test_strict_rejection_does_not_persist_encoded_crop(
    client: TestClient, image_bytes: Callable[..., bytes]
) -> None:
    _create_collection(client, "employees", save_face_crops=True)
    _create_person(client, image_bytes, "employees", "alice", seed=105)
    _create_person(client, image_bytes, "employees", "bob", seed=106)

    rejected = client.post(
        "/v1/collections/employees/persons/alice/faces",
        data={"review_mode": "strict"},
        files={"images": ("bob.png", image_bytes(106), "image/png")},
    )

    assert rejected.status_code == 201, rejected.text
    assert rejected.json()["faces"] == []
    assert rejected.json()["rejected_images"][0]["reason"] == (
        "identity_similarity_conflict"
    )
    with client.app.state.database.read() as connection:
        persisted = connection.execute(
            "SELECT count(*) FROM face_samples WHERE crop_image IS NOT NULL"
        ).fetchone()[0]
    assert persisted == 2


def test_openapi_documents_collection_crop_flag_and_jpeg_download(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()
    collection_create = schema["components"]["schemas"]["CollectionCreate"]
    collection_patch = schema["components"]["schemas"]["CollectionPatch"]
    operation = schema["paths"][
        "/v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image"
    ]["get"]

    assert "save_face_crops" in collection_create["properties"]
    assert "save_face_crops" in collection_patch["properties"]
    assert operation["security"] == [{"bearerAuth": []}]
    jpeg = operation["responses"]["200"]["content"]["image/jpeg"]["schema"]
    assert jpeg == {"type": "string", "format": "binary"}
