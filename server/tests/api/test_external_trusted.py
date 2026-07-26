from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _unit_vector(dimension: int, index: int = 0) -> list[float]:
    values = [0.0] * dimension
    values[index] = 1.0
    return values


def _trusted_data(
    contract_id: str,
    embeddings: list[Any],
    **values: str,
) -> dict[str, str]:
    return {
        "embedding_mode": "external_trusted",
        "embedding_contract_id": contract_id,
        "external_embeddings": json.dumps(embeddings),
        **values,
    }


def test_external_trusted_skips_recognizer_persists_and_indexes_supplied_vector(
    client: TestClient,
    create_collection,
    image_bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = create_collection(client)
    contract_id = str(collection["embedding_contract_id"])
    dimension = int(collection["embedding_dimension"])
    supplied = _unit_vector(dimension, 7)
    calls: list[bool] = []
    original_analyze = client.app.state.engine.analyze

    def tracked_analyze(image, *, require_embeddings=True, **kwargs):
        calls.append(bool(require_embeddings))
        observations = original_analyze(image, require_embeddings=require_embeddings, **kwargs)
        if not require_embeddings:
            assert all(face.embedding is None for face in observations)
        return observations

    monkeypatch.setattr(client.app.state.engine, "analyze", tracked_analyze)
    response = client.post(
        "/v1/collections/employees/persons",
        data=_trusted_data(contract_id, [supplied], id="alice"),
        files={"images": ("alice.png", image_bytes(501), "image/png")},
    )

    assert response.status_code == 201, response.text
    face = response.json()["faces"][0]
    assert face["embedding_source"] == "external_trusted"
    assert face["embedding_contract_id"] == contract_id
    assert calls == [False]
    stored = client.app.state.repository.list_face_embeddings("employees", "alice")
    assert len(stored) == 1
    np.testing.assert_allclose(stored[0], np.asarray(supplied, dtype=np.float32))
    matches = client.app.state.search_indexes.search(
        "employees",
        np.asarray(supplied, dtype=np.float32),
        limit=1,
        threshold=-1.0,
    )
    assert matches[0]["person"]["id"] == "alice"
    assert matches[0]["matched_face_id"] == face["id"]


def test_external_trusted_preserves_original_image_mapping_and_partial_success(
    client: TestClient,
    create_collection,
    image_bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = create_collection(client)
    contract_id = str(collection["embedding_contract_id"])
    dimension = int(collection["embedding_dimension"])
    valid = _unit_vector(dimension)
    calls: list[bool] = []
    original_analyze = client.app.state.engine.analyze

    def tracked_analyze(image, *, require_embeddings=True, **kwargs):
        calls.append(bool(require_embeddings))
        observations = original_analyze(image, require_embeddings=require_embeddings, **kwargs)
        # Even if an engine violates require_embeddings=False, the service must
        # never replace a rejected trusted vector with this recognizer output.
        for face in observations:
            face.embedding = np.asarray(_unit_vector(dimension), dtype=np.float32)
        return observations

    monkeypatch.setattr(client.app.state.engine, "analyze", tracked_analyze)
    response = client.post(
        "/v1/collections/employees/persons",
        data=_trusted_data(
            contract_id,
            [valid, valid, [0.0] * dimension],
            id="alice",
        ),
        files=[
            ("images", ("accepted.png", image_bytes(502), "image/png")),
            ("images", ("invalid.png", b"not-an-image", "image/png")),
            ("images", ("bad-vector.png", image_bytes(503), "image/png")),
        ],
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert len(body["faces"]) == 1
    assert {(item["index"], item["reason"]) for item in body["rejected_images"]} == {
        (1, "invalid_image"),
        (2, "invalid_external_embedding"),
    }
    # Invalid image bytes never reach inference, while both decoded images run
    # detector/quality without recognition. The invalid vector never falls back.
    assert calls == [False, False]


def test_external_trusted_review_modes_match_image_registration_semantics(
    client: TestClient,
    create_collection,
    image_bytes,
) -> None:
    collection = create_collection(client)
    contract_id = str(collection["embedding_contract_id"])
    vector = _unit_vector(int(collection["embedding_dimension"]))
    small = image_bytes(504, width=32, height=32)

    off = client.post(
        "/v1/collections/employees/persons",
        data=_trusted_data(contract_id, [vector], id="off", review_mode="off"),
        files={"images": ("small-off.png", small, "image/png")},
    )
    standard = client.post(
        "/v1/collections/employees/persons",
        data=_trusted_data(contract_id, [vector], id="standard", review_mode="standard"),
        files={"images": ("small-standard.png", small, "image/png")},
    )

    assert off.status_code == 201, off.text
    assert standard.status_code == 422
    assert standard.json()["error"]["details"]["rejected_images"] == [
        {"index": 0, "filename": "small-standard.png", "reason": "face_too_small"}
    ]


def test_external_trusted_off_binds_vector_to_largest_detected_face(
    client: TestClient,
    create_collection,
    image_bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = create_collection(client)
    contract_id = str(collection["embedding_contract_id"])
    vector = _unit_vector(int(collection["embedding_dimension"]))
    multiple = image_bytes(506, width=256, height=100)
    calls: list[bool] = []
    original_analyze = client.app.state.engine.analyze

    def tracked_analyze(image, *, require_embeddings=True, **kwargs):
        calls.append(bool(require_embeddings))
        return original_analyze(image, require_embeddings=require_embeddings, **kwargs)

    monkeypatch.setattr(client.app.state.engine, "analyze", tracked_analyze)

    off = client.post(
        "/v1/collections/employees/persons",
        data=_trusted_data(contract_id, [vector], id="off-largest", review_mode="off"),
        files={"images": ("multiple-off.png", multiple, "image/png")},
    )
    standard = client.post(
        "/v1/collections/employees/persons",
        data=_trusted_data(
            contract_id,
            [vector],
            id="standard-multiple",
            review_mode="standard",
        ),
        files={"images": ("multiple-standard.png", multiple, "image/png")},
    )

    assert off.status_code == 201, off.text
    assert len(off.json()["faces"]) == 1
    assert off.json()["faces"][0]["bounding_box"]["pixels"]["width"] == 62
    assert off.json()["rejected_images"] == []
    np.testing.assert_allclose(
        client.app.state.repository.list_face_embeddings("employees", "off-largest")[0],
        np.asarray(vector, dtype=np.float32),
    )
    assert standard.status_code == 422
    assert standard.json()["error"]["details"]["rejected_images"][0]["reason"] == (
        "multiple_faces"
    )
    assert calls == [False, False]


def test_external_trusted_strict_review_uses_supplied_embedding(
    client: TestClient,
    create_collection,
    image_bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = create_collection(client)
    contract_id = str(collection["embedding_contract_id"])
    dimension = int(collection["embedding_dimension"])
    first = _unit_vector(dimension, 0)
    second = _unit_vector(dimension, 1)
    created = client.post(
        "/v1/collections/employees/persons",
        data=_trusted_data(contract_id, [first], id="alice"),
        files={"images": ("first.png", image_bytes(505), "image/png")},
    )
    assert created.status_code == 201, created.text

    observed_queries: list[np.ndarray] = []

    def no_other(_collection_id, query, *, exclude_person_id):
        assert exclude_person_id == "alice"
        observed_queries.append(np.asarray(query, dtype=np.float32).copy())
        return None

    monkeypatch.setattr(
        client.app.state.search_indexes,
        "best_other_person",
        no_other,
        raising=False,
    )
    response = client.post(
        "/v1/collections/employees/persons/alice/faces",
        data=_trusted_data(contract_id, [second], review_mode="strict"),
        files={"images": ("second.png", image_bytes(506), "image/png")},
    )

    assert response.status_code == 201, response.text
    assert len(response.json()["faces"]) == 1
    assert len(observed_queries) == 1
    np.testing.assert_array_equal(observed_queries[0], np.asarray(second, dtype=np.float32))
    stored = client.app.state.repository.list_face_embeddings("employees", "alice")
    assert any(np.array_equal(value, np.asarray(second, dtype=np.float32)) for value in stored)


@pytest.mark.parametrize(
    ("data", "expected_code"),
    [
        (
            {"external_embeddings": "[]"},
            "unexpected_external_embedding",
        ),
        (
            {"embedding_mode": "external_trusted", "embedding_contract_id": "value"},
            "missing_external_embeddings",
        ),
        (
            {
                "embedding_mode": "external_trusted",
                "embedding_contract_id": "value",
                "external_embeddings": "not-json",
            },
            "invalid_external_embeddings",
        ),
        (
            {
                "embedding_mode": "external_trusted",
                "embedding_contract_id": "value",
                "external_embeddings": "[]",
            },
            "external_embedding_count_mismatch",
        ),
    ],
)
def test_external_trusted_rejects_ambiguous_or_malformed_multipart_contract(
    client: TestClient,
    create_collection,
    image_bytes,
    data: dict[str, str],
    expected_code: str,
) -> None:
    create_collection(client)
    response = client.post(
        "/v1/collections/employees/persons",
        data={"id": "alice", **data},
        files={"images": ("alice.png", image_bytes(507), "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == expected_code


def test_external_trusted_rejects_contract_mismatch_before_inference(
    client: TestClient,
    create_collection,
    image_bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = create_collection(client)
    vector = _unit_vector(int(collection["embedding_dimension"]))
    calls = 0

    def unexpected_analyze(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("contract mismatch must not run inference")

    monkeypatch.setattr(client.app.state.engine, "analyze", unexpected_analyze)
    response = client.post(
        "/v1/collections/employees/persons",
        data=_trusted_data("ifsemb-v1-sha256:" + "0" * 64, [vector], id="alice"),
        files={"images": ("alice.png", image_bytes(508), "image/png")},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "embedding_contract_mismatch"
    assert calls == 0


def test_external_trusted_invalid_vector_is_per_image_rejection_without_fallback(
    client: TestClient,
    create_collection,
    image_bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = create_collection(client)
    contract_id = str(collection["embedding_contract_id"])
    dimension = int(collection["embedding_dimension"])
    calls: list[bool] = []
    original_analyze = client.app.state.engine.analyze

    def tracked_analyze(image, *, require_embeddings=True, **kwargs):
        calls.append(bool(require_embeddings))
        return original_analyze(image, require_embeddings=require_embeddings, **kwargs)

    monkeypatch.setattr(client.app.state.engine, "analyze", tracked_analyze)
    response = client.post(
        "/v1/collections/employees/persons",
        data=_trusted_data(contract_id, [[0.0] * dimension], id="alice"),
        files={"images": ("alice.png", image_bytes(509), "image/png")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["details"]["rejected_images"] == [
        {
            "index": 0,
            "filename": "alice.png",
            "reason": "invalid_external_embedding",
        }
    ]
    assert calls == [False]
