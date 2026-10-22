from __future__ import annotations

import json
from collections.abc import Callable
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from insightface_server.app import create_app
from insightface_server.config import Settings
from PIL import Image


def as_bmp(png: bytes) -> bytes:
    stream = BytesIO()
    with Image.open(BytesIO(png)) as image:
        image.save(stream, format="BMP")
    return stream.getvalue()


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [("face.bmp", "image/bmp"), ("face.png", "image/png"), ("upload", "application/octet-stream")],
)
def test_bmp_detect_embeddings_and_compare_use_decoded_pixels(
    client: TestClient, image_bytes, filename: str, content_type: str
) -> None:
    png = image_bytes(121)
    bmp_upload = (filename, as_bmp(png), content_type)
    png_upload = ("face.png", png, "image/png")

    # Match the complete face and embedding payloads against a lossless PNG of
    # the same pixels, regardless of the BMP upload's filename or declared MIME.
    for endpoint in ("/v1/detect", "/v1/embeddings"):
        bmp_response = client.post(endpoint, files={"image": bmp_upload})
        png_response = client.post(endpoint, files={"image": png_upload})
        assert bmp_response.status_code == 200, bmp_response.text
        assert png_response.status_code == 200, png_response.text
        assert len(bmp_response.json()["faces"]) == 1
        assert bmp_response.json()["faces"] == png_response.json()["faces"]

    compared = client.post(
        "/v1/compare",
        data={"threshold": "0.99"},
        files={"source": bmp_upload, "target": png_upload},
    )
    assert compared.status_code == 200, compared.text
    assert compared.json()["matched"] is True
    assert compared.json()["similarity"] == pytest.approx(1.0)


@pytest.mark.parametrize("embedding_mode", ["server", "external_trusted"])
def test_bmp_registration_add_samples_and_search(
    client: TestClient, create_collection, image_bytes, embedding_mode: str
) -> None:
    collection = create_collection(client)
    first = as_bmp(image_bytes(122))
    second = as_bmp(image_bytes(123))

    def registration_data(content: bytes) -> dict[str, str]:
        if embedding_mode == "server":
            return {"embedding_mode": "server"}
        response = client.post(
            "/v1/embeddings", files={"image": ("sample.bmp", content, "image/bmp")}
        )
        assert response.status_code == 200, response.text
        return {
            "embedding_mode": "external_trusted",
            "embedding_contract_id": str(collection["embedding_contract_id"]),
            "external_embeddings": json.dumps([response.json()["faces"][0]["embedding"]]),
        }

    created = client.post(
        "/v1/collections/employees/persons",
        data={"id": "bmp-person", **registration_data(first)},
        files={"images": ("first.bmp", first, "image/bmp")},
    )
    added = client.post(
        "/v1/collections/employees/persons/bmp-person/faces",
        data=registration_data(second),
        files={"images": ("second.bmp", second, "application/octet-stream")},
    )
    for response in (created, added):
        assert response.status_code == 201, response.text
        assert len(response.json()["faces"]) == 1
        assert response.json()["faces"][0]["embedding_source"] == embedding_mode
        assert response.json()["rejected_images"] == []

    searched = client.post(
        "/v1/collections/employees/search",
        data={"threshold": "0.99"},
        files={"image": ("query.bmp", second, "image/bmp")},
    )
    assert searched.status_code == 200, searched.text
    assert len(searched.json()["matches"]) == 1
    match = searched.json()["matches"][0]
    assert match["person"]["id"] == "bmp-person"
    assert match["matched_face_id"] == added.json()["faces"][0]["id"]
    assert match["similarity"] == pytest.approx(1.0)


def test_invalid_bmp_does_not_prevent_valid_bmp_registration(
    client: TestClient, create_collection, image_bytes
) -> None:
    create_collection(client)
    response = client.post(
        "/v1/collections/employees/persons",
        data={"id": "bmp-person"},
        files=[
            ("images", ("broken.bmp", b"BM\x00\x00", "image/bmp")),
            ("images", ("valid.bmp", as_bmp(image_bytes(124)), "image/bmp")),
        ],
    )

    assert response.status_code == 201, response.text
    assert len(response.json()["faces"]) == 1
    assert response.json()["rejected_images"] == [
        {"index": 0, "filename": "broken.bmp", "reason": "invalid_image"}
    ]


@pytest.mark.parametrize("limit", ["max_image_bytes", "max_image_pixels"])
def test_bmp_upload_limits_return_structured_errors(
    make_settings: Callable[..., Settings], image_bytes, limit: str
) -> None:
    content = as_bmp(image_bytes(125))
    boundary = len(content) if limit == "max_image_bytes" else 128 * 128
    settings = make_settings(**{limit: boundary - 1})
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/detect", files={"image": ("large.bmp", content, "image/bmp")}
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "image_too_large"
