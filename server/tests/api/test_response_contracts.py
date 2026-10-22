from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel


def _assert_response_model(
    client: TestClient,
    response,
    *,
    method: str,
    path: str,
) -> None:
    assert response.status_code in {200, 201}, response.text
    route = next(
        route
        for route in client.app.routes
        if isinstance(route, APIRoute)
        and route.path == path
        and method in route.methods
    )
    model = route.response_model
    assert isinstance(model, type) and issubclass(model, BaseModel)
    model.model_validate(response.json())


def test_system_and_models_endpoints_share_one_license_summary(
    client: TestClient,
) -> None:
    license_summary = {
        "license_id": None,
        "issuer": None,
        "model_id": "synthetic-recognition",
        "grant": "non-commercial",
        "customer": None,
        "reference": None,
        "valid_from": None,
        "valid_until": None,
        "signature_valid": False,
        "commercial_use_permitted": False,
        "status": "default_non_commercial",
        "defaulted": True,
        "message": "MODEL.LICENSE is absent; defaulting to non-commercial use",
    }
    engine = client.app.state.engine
    engine.summary = replace(engine.summary, license=license_summary)

    models = client.get("/v1/models")
    system = client.get("/v1/system")

    assert models.status_code == 200
    assert system.status_code == 200
    assert models.json()["license"] == license_summary
    assert system.json()["model"]["license"] == license_summary


def test_every_json_success_response_conforms_to_its_public_model(
    client: TestClient,
    image_bytes: Callable[..., bytes],
) -> None:
    responses = [
        ("GET", "/v1/health", client.get("/v1/health")),
        ("GET", "/v1/system", client.get("/v1/system")),
        ("GET", "/v1/models", client.get("/v1/models")),
        (
            "POST",
            "/v1/detect",
            client.post(
                "/v1/detect",
                files={"image": ("detect.png", image_bytes(1), "image/png")},
            ),
        ),
        (
            "POST",
            "/v1/compare",
            client.post(
                "/v1/compare",
                files={
                    "source": ("source.png", image_bytes(2), "image/png"),
                    "target": ("target.png", image_bytes(3), "image/png"),
                },
            ),
        ),
        (
            "POST",
            "/v1/embeddings",
            client.post(
                "/v1/embeddings",
                files={"image": ("embedding.png", image_bytes(4), "image/png")},
            ),
        ),
    ]

    responses.append(
        (
            "POST",
            "/v1/collections",
            client.post(
                "/v1/collections",
                json={"id": "contracts", "name": "Contract fixtures"},
            ),
        )
    )
    responses.extend(
        [
            ("GET", "/v1/collections", client.get("/v1/collections")),
            (
                "GET",
                "/v1/collections/{collection_id}",
                client.get("/v1/collections/contracts"),
            ),
            (
                "PATCH",
                "/v1/collections/{collection_id}",
                client.patch(
                    "/v1/collections/contracts",
                    json={"description": "Updated"},
                ),
            ),
        ]
    )

    responses.append(
        (
            "POST",
            "/v1/collections/{collection_id}/persons",
            client.post(
                "/v1/collections/contracts/persons",
                data={"id": "alice", "name": "Alice"},
                files={"images": ("alice.png", image_bytes(5), "image/png")},
            ),
        )
    )
    responses.extend(
        [
            (
                "GET",
                "/v1/collections/{collection_id}/persons",
                client.get("/v1/collections/contracts/persons"),
            ),
            (
                "GET",
                "/v1/collections/{collection_id}/persons/{person_id}",
                client.get("/v1/collections/contracts/persons/alice"),
            ),
            (
                "PATCH",
                "/v1/collections/{collection_id}/persons/{person_id}",
                client.patch(
                    "/v1/collections/contracts/persons/alice",
                    json={"metadata": {"team": "vision"}},
                ),
            ),
            (
                "POST",
                "/v1/collections/{collection_id}/persons/{person_id}/faces",
                client.post(
                    "/v1/collections/contracts/persons/alice/faces",
                    files={"images": ("alice-2.png", image_bytes(6), "image/png")},
                ),
            ),
            (
                "GET",
                "/v1/collections/{collection_id}/persons/{person_id}/faces",
                client.get("/v1/collections/contracts/persons/alice/faces"),
            ),
            (
                "POST",
                "/v1/collections/{collection_id}/search",
                client.post(
                    "/v1/collections/contracts/search",
                    files={"image": ("query.png", image_bytes(5), "image/png")},
                ),
            ),
        ]
    )

    monitor = {
        "id": "contract-monitor",
        "name": "Contract monitor",
        "enabled": False,
        "source": {"type": "rtsp", "url": "rtsp://camera.test/live"},
        "collection_id": "contracts",
    }
    responses.append(
        (
            "POST",
            "/v1/monitors",
            client.post("/v1/monitors", json=monitor),
        )
    )
    responses.extend(
        [
            ("GET", "/v1/monitors", client.get("/v1/monitors")),
            (
                "GET",
                "/v1/monitors/{monitor_id}",
                client.get("/v1/monitors/contract-monitor"),
            ),
            (
                "PATCH",
                "/v1/monitors/{monitor_id}",
                client.patch(
                    "/v1/monitors/contract-monitor",
                    json={"event_policy": {"emit_unknown": False}},
                ),
            ),
            (
                "GET",
                "/v1/monitors/{monitor_id}/state",
                client.get("/v1/monitors/contract-monitor/state"),
            ),
            (
                "GET",
                "/v1/monitors/{monitor_id}/events",
                client.get("/v1/monitors/contract-monitor/events"),
            ),
        ]
    )

    assert len(responses) == 23
    for method, path, response in responses:
        _assert_response_model(
            client,
            response,
            method=method,
            path=path,
        )


def test_openapi_has_typed_success_and_special_response_contracts(
    client: TestClient,
) -> None:
    schema = client.app.openapi()
    json_operations = 0
    for route in client.app.routes:
        if (
            not isinstance(route, APIRoute)
            or not route.path.startswith("/v1/")
            or route.status_code == 204
            or route.path.endswith("/image")
            or route.path.endswith("/preview.mjpeg")
        ):
            continue
        for method in route.methods & {"GET", "POST", "PATCH", "DELETE"}:
            success = schema["paths"][route.path][method.lower()]["responses"][
                str(route.status_code or 200)
            ]
            response_schema = success["content"]["application/json"]["schema"]
            assert response_schema, f"{method} {route.path}"
            assert response_schema != {"type": "object"}, f"{method} {route.path}"
            json_operations += 1
    assert json_operations == 25

    health_503 = schema["paths"]["/v1/health"]["get"]["responses"]["503"]
    assert health_503["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/HealthResponse"
    }
    for method in ("post", "patch"):
        path = "/v1/monitors" if method == "post" else "/v1/monitors/{monitor_id}"
        rate_limited = schema["paths"][path][method]["responses"]["429"]
        assert rate_limited["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/ErrorEnvelope"
        }

    jpeg = schema["paths"][
        "/v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image"
    ]["get"]["responses"]["200"]["content"]["image/jpeg"]["schema"]
    assert jpeg == {"type": "string", "format": "binary"}
    preview = schema["paths"]["/v1/monitors/{monitor_id}/preview.mjpeg"]["get"][
        "responses"
    ]["200"]["content"]["multipart/x-mixed-replace"]["schema"]
    assert preview == {"type": "string", "format": "binary"}

    for route in client.app.routes:
        if not isinstance(route, APIRoute) or route.status_code != 204:
            continue
        for method in route.methods & {"DELETE"}:
            response = schema["paths"][route.path][method.lower()]["responses"]["204"]
            assert "content" not in response
