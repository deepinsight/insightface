from __future__ import annotations

import math
import time
from collections.abc import Callable
from io import BytesIO

from fastapi.testclient import TestClient
from insightface_server.app import create_app
from insightface_server.config import Settings
from PIL import Image


def test_detect_sorts_faces_and_empty_image_is_success(client: TestClient, image_bytes) -> None:
    two_faces = client.post(
        "/v1/detect",
        data={"max_faces": "2"},
        files={"image": ("wide.png", image_bytes(4, width=256, height=100), "image/png")},
    )
    blank = client.post(
        "/v1/detect",
        files={"image": ("blank.png", image_bytes(blank=True), "image/png")},
    )

    assert two_faces.status_code == 200
    faces = two_faces.json()["faces"]
    assert len(faces) == 2
    areas = [face["bbox"]["pixels"]["width"] * face["bbox"]["pixels"]["height"] for face in faces]
    assert areas == sorted(areas, reverse=True)
    assert all(0 <= face["detection_score"] <= 1 for face in faces)
    assert all(len(face["landmarks"]) == 5 for face in faces)
    assert blank.status_code == 200
    assert blank.json()["faces"] == []


def test_detect_respects_max_faces_and_rejects_request_threshold_override(
    client: TestClient, image_bytes
) -> None:
    maximum = client.post(
        "/v1/detect",
        data={"max_faces": "1"},
        files={"image": ("wide.png", image_bytes(9, width=256, height=100), "image/png")},
    )
    strict = client.post(
        "/v1/detect",
        data={"min_score": "1.0"},
        files={"image": ("face.png", image_bytes(9), "image/png")},
    )

    assert maximum.status_code == 200
    assert len(maximum.json()["faces"]) == 1
    assert strict.status_code == 400
    assert strict.json()["error"]["code"] == "request_detection_override_not_supported"


def test_detect_accepts_webp(client: TestClient, image_bytes) -> None:
    encoded = BytesIO()
    with Image.open(BytesIO(image_bytes(17))) as image:
        image.save(encoded, format="WEBP", quality=90)

    response = client.post(
        "/v1/detect",
        files={"image": ("face.webp", encoded.getvalue(), "image/webp")},
    )

    assert response.status_code == 200
    assert len(response.json()["faces"]) == 1


def test_compare_same_synthetic_image_and_face_not_found(client: TestClient, image_bytes) -> None:
    sample = image_bytes(21)
    compared = client.post(
        "/v1/compare",
        data={"threshold": "0.99"},
        files={
            "source": ("source.png", sample, "image/png"),
            "target": ("target.png", sample, "image/png"),
        },
    )
    missing = client.post(
        "/v1/compare",
        files={
            "source": ("source.png", sample, "image/png"),
            "target": ("blank.png", image_bytes(blank=True), "image/png"),
        },
    )

    assert compared.status_code == 200
    assert compared.json()["matched"] is True
    assert math.isclose(compared.json()["similarity"], 1.0)
    assert compared.json()["threshold"] == 0.99
    assert compared.json()["source_face"]["bbox"] == compared.json()["target_face"]["bbox"]
    assert missing.status_code == 422
    assert missing.json()["error"]["code"] == "face_not_found"


def test_compare_returns_raw_negative_cosine_without_affine_mapping(
    client: TestClient, image_bytes
) -> None:
    compared = client.post(
        "/v1/compare",
        data={"threshold": "0"},
        files={
            "source": ("source.png", image_bytes(3), "image/png"),
            "target": ("target.png", image_bytes(20), "image/png"),
        },
    )

    assert compared.status_code == 200
    assert compared.json()["similarity"] < 0.0
    assert compared.json()["matched"] is False
    assert compared.json()["threshold"] == 0.0


def test_embeddings_uses_profile_selection_and_rejects_request_override(
    client: TestClient, image_bytes
) -> None:
    selected = client.post(
        "/v1/embeddings",
        files={"image": ("wide.png", image_bytes(8, width=256, height=100), "image/png")},
    )
    invalid = client.post(
        "/v1/embeddings",
        data={"face_selection": "nearest"},
        files={"image": ("face.png", image_bytes(8), "image/png")},
    )

    assert selected.status_code == 200
    assert len(selected.json()["faces"]) == 1
    for face in selected.json()["faces"]:
        embedding = face["embedding"]
        assert len(embedding) == 512
        assert math.isclose(sum(value * value for value in embedding), 1.0, abs_tol=1e-5)
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "request_detection_override_not_supported"


def test_inference_timeout_returns_without_waiting_for_worker(
    make_settings: Callable[..., Settings], image_bytes
) -> None:
    settings = make_settings(request_timeout_seconds=1)
    with TestClient(create_app(settings)) as client:
        def slow_detect(*_args, **_kwargs):
            time.sleep(2.0)
            return []

        client.app.state.service.detect = slow_detect
        started = time.monotonic()
        response = client.post(
            "/v1/detect",
            files={"image": ("face.png", image_bytes(7), "image/png")},
        )
        elapsed = time.monotonic() - started

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "request_timeout"
    assert elapsed < 1.6
