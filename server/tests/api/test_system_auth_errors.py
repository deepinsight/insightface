from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from insightface_server.app import create_app
from insightface_server.config import Settings


def assert_request_id(response) -> str:
    request_id = response.headers["x-request-id"]
    uuid.UUID(request_id)
    if response.content:
        assert response.json()["request_id"] == request_id
    return request_id


def test_health_is_public_and_has_correlated_request_id(client: TestClient) -> None:
    response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["auth_enabled"] is False
    assert_request_id(response)
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["permissions-policy"] == (
        "camera=(), microphone=(), geolocation=()"
    )


def test_versioned_multilingual_markdown_is_served_read_only(client: TestClient) -> None:
    assert client.get("/").status_code == 200
    assert client.get("/docs").status_code == 200

    english = client.get("/guide-content/en/user-guide.md")
    assert english.status_code == 200
    assert english.text.startswith("# InsightFace Server user guide")
    assert english.headers["x-content-type-options"] == "nosniff"
    assert "text/markdown" in english.headers["content-type"]

    readme = client.get("/guide-content/zh/readme.md")
    assert readme.status_code == 404

    api_guide = client.get("/guide-content/zh/api.md")
    assert api_guide.status_code == 200
    assert "REST API使用手册" in api_guide.text
    assert "POST /v1/collections/{collection_id}/search" in api_guide.text

    maintainer = client.get("/guide-content/zh/maintainer.md")
    assert maintainer.status_code == 200
    assert maintainer.text.startswith("# InsightFace Server maintainer guide")
    assert "English only" in maintainer.text

    screenshot = client.get("/guide-images/customer/dashboard-en.jpg")
    assert screenshot.status_code == 200
    assert screenshot.headers["content-type"] == "image/jpeg"
    assert screenshot.content.startswith(b"\xff\xd8")

    assert client.get("/guide-content/it/user-guide.md").status_code == 404
    assert client.get("/guide-content/en/not-a-document.md").status_code == 404


def test_api_only_mode_does_not_register_web_ui_routes(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(web_ui_disabled=True)
    with TestClient(create_app(settings)) as api_client:
        assert api_client.get("/v1/health").status_code == 200
        schema = api_client.get("/openapi.json")
        assert schema.status_code == 200
        assert "/v1/health" in schema.json()["paths"]
        for path in (
            "/",
            "/docs",
            "/assets/app.mjs",
            "/guide-content/en/user-guide.md",
            "/guide-content/en/api.md",
            "/guide-content/en/maintainer.md",
            "/guide-content/en/readme.md",
            "/guide-images/customer/dashboard-en.jpg",
        ):
            response = api_client.get(path)
            assert response.status_code == 404
            assert response.json()["error"]["code"] == "route_not_found"


def test_detector_configuration_is_visible_but_not_mutable_at_runtime(
    client: TestClient,
) -> None:
    system = client.get("/v1/system")
    mutation = client.patch(
        "/v1/system/config",
        json={"detector_input_sizes": [[128, 128]]},
    )

    assert system.status_code == 200
    assert system.json()["safe_config"]["detector_input_sizes"] == [
        [96, 96],
        [512, 512],
    ]
    assert system.json()["safe_config"]["web_ui_disabled"] is False
    assert system.json()["safe_config"]["config_file"] is None
    assert mutation.status_code == 404
    assert mutation.json()["error"]["code"] == "route_not_found"


def test_authentication_rejects_missing_and_invalid_keys(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(auth_enabled=True, startup_api_key="correct-secret")
    with TestClient(create_app(settings)) as client:
        health = client.get("/v1/health")
        assert health.status_code == 200
        assert health.json()["auth_enabled"] is True

        missing = client.get("/v1/system")
        invalid = client.get("/v1/system", headers={"Authorization": "Bearer wrong-secret"})
        valid = client.get("/v1/system", headers={"Authorization": "Bearer correct-secret"})

    for response in (missing, invalid):
        assert response.status_code == 401
        assert response.json()["error"] == {
            "code": "unauthorized",
            "message": "A valid Bearer API key is required.",
            "details": {},
        }
        assert_request_id(response)
    assert valid.status_code == 200
    assert valid.json()["api_key"] == {
        "authentication_enabled": True,
        "configured": True,
    }

    database = sqlite3.connect(settings.data_dir / "insightface-server.db")
    row = database.execute("SELECT salt,digest FROM api_keys").fetchone()
    database.close()
    assert row is not None
    assert b"correct-secret" not in row[0]
    assert b"correct-secret" not in row[1]


def test_startup_api_key_rotates_and_is_optional_after_initialization(
    make_settings: Callable[..., Settings],
) -> None:
    initial = make_settings(auth_enabled=True, startup_api_key="first-secret")
    with TestClient(create_app(initial)) as client:
        assert client.get(
            "/v1/system", headers={"Authorization": "Bearer first-secret"}
        ).status_code == 200

    rotated = replace(initial, startup_api_key="second-secret")
    with TestClient(create_app(rotated)) as client:
        assert client.get(
            "/v1/system", headers={"Authorization": "Bearer first-secret"}
        ).status_code == 401
        assert client.get(
            "/v1/system", headers={"Authorization": "Bearer second-secret"}
        ).status_code == 200

    retained = replace(initial, startup_api_key=None)
    with TestClient(create_app(retained)) as client:
        assert client.get(
            "/v1/system", headers={"Authorization": "Bearer second-secret"}
        ).status_code == 200

    database = sqlite3.connect(initial.data_dir / "insightface-server.db")
    rows = database.execute(
        "SELECT label,active FROM api_keys ORDER BY active,id"
    ).fetchall()
    database.close()
    assert rows == [("startup", 0), ("startup", 1)]


def test_new_authenticated_database_requires_a_startup_key(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(auth_enabled=True, startup_api_key=None)
    with pytest.raises(RuntimeError, match="set INSIGHTFACE_API_KEY"):
        with TestClient(create_app(settings)):
            pass


def test_validation_and_domain_errors_use_one_error_shape(client: TestClient, image_bytes) -> None:
    invalid_id = client.post("/v1/collections", json={"id": "bad id", "name": "Invalid"})
    invalid_image = client.post(
        "/v1/detect", files={"image": ("bad.jpg", b"not-an-image", "image/jpeg")}
    )

    assert invalid_id.status_code == 400
    assert invalid_id.json()["error"]["code"] == "invalid_request"
    assert invalid_image.status_code == 422
    assert invalid_image.json()["error"] == {
        "code": "invalid_image",
        "message": "The image could not be decoded.",
        "details": {},
    }
    assert_request_id(invalid_id)
    assert_request_id(invalid_image)


def test_request_body_limit_returns_413(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(max_request_bytes=32)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/collections",
            content=b"{" + b"x" * 100 + b"}",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert_request_id(response)


def test_streamed_request_body_cannot_bypass_limit(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(max_request_bytes=48)

    def chunks():
        yield b'{"id":"streamed","name":"'
        yield b"x" * 128
        yield b'"}'

    with TestClient(create_app(settings)) as client:
        request = client.build_request(
            "POST",
            "/v1/collections",
            content=chunks(),
            headers={"content-type": "application/json"},
        )
        response = client.send(request)

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert_request_id(response)


def test_unknown_collection_is_a_structured_404(client: TestClient) -> None:
    response = client.get("/v1/collections/missing")

    assert response.status_code == 404
    assert response.json()["error"] == {
        "code": "collection_not_found",
        "message": "Collection 'missing' was not found.",
        "details": {},
    }
    assert_request_id(response)


def test_framework_404_and_405_use_the_standard_error_shape(client: TestClient) -> None:
    missing = client.get("/v1/not-a-route")
    wrong_method = client.put("/v1/health")

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "route_not_found"
    assert wrong_method.status_code == 405
    assert wrong_method.json()["error"]["code"] == "method_not_allowed"
    assert_request_id(missing)
    assert_request_id(wrong_method)


def test_openapi_declares_bearer_auth_and_custom_errors(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert schema["components"]["securitySchemes"]["bearerAuth"] == {
        "type": "http",
        "scheme": "bearer",
    }
    assert schema["paths"]["/v1/system"]["get"]["security"] == [{"bearerAuth": []}]
    assert "security" not in schema["paths"]["/v1/health"]["get"]
    assert "400" in schema["paths"]["/v1/collections"]["post"]["responses"]


def test_openapi_documents_resource_creation_and_deletion_statuses(
    client: TestClient,
) -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert "201" in paths["/v1/collections"]["post"]["responses"]
    assert "204" in paths["/v1/collections/{collection_id}"]["delete"]["responses"]
    assert "201" in paths["/v1/collections/{collection_id}/persons"]["post"]["responses"]
    assert (
        "204"
        in paths["/v1/collections/{collection_id}/persons/{person_id}"]["delete"][
            "responses"
        ]
    )
    assert (
        "201"
        in paths["/v1/collections/{collection_id}/persons/{person_id}/faces"]["post"][
            "responses"
        ]
    )
    assert (
        "204"
        in paths[
            "/v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}"
        ]["delete"]["responses"]
    )


def test_openapi_documents_enrollment_review_modes(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    for path in (
        "/v1/collections/{collection_id}/persons",
        "/v1/collections/{collection_id}/persons/{person_id}/faces",
    ):
        request_schema = schema["paths"][path]["post"]["requestBody"]["content"][
            "multipart/form-data"
        ]["schema"]
        reference = request_schema["$ref"].rsplit("/", 1)[-1]
        review = schema["components"]["schemas"][reference]["properties"][
            "review_mode"
        ]
        assert review["enum"] == ["off", "standard", "strict"]
        assert review["default"] == "off"
