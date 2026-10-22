from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from insightface_server import app as app_module
from insightface_server.addons import LivenessUnavailable
from insightface_server.api.responses import LivenessResult
from insightface_server.errors import ApiError
from insightface_server.inference.mock import MockInferenceEngine

INPUT_REJECTED = {
    "status": "input_rejected",
    "is_live": None,
    "live_score": None,
    "reason": (
        "Insufficient image area around the face for liveness detection. Move the face "
        "toward the center, step back from the camera, or use a less tightly cropped image."
    ),
}


@pytest.fixture
def liveness_client(make_settings, monkeypatch):
    clients = []

    def make(mode="normal", scope="both", result=None, on_registration=False):
        state = {
            "result": result or {"status": "ok", "is_live": True, "live_score": 0.99},
            "fail": False,
            "missing_embedding": False,
            "calls": [],
        }

        class Engine(MockInferenceEngine):
            def analyze(self, image, *, apply_liveness=True, **kwargs):
                state["calls"].append((apply_liveness, kwargs.get("require_embeddings", True)))
                if apply_liveness and state["fail"]:
                    raise LivenessUnavailable("simulated runtime failure")
                faces = super().analyze(image, **kwargs)
                if apply_liveness:
                    for face in faces:
                        face.liveness = dict(state["result"])
                        if mode == "normal" and face.liveness["is_live"] is not True:
                            face.embedding = None
                        if state["missing_embedding"]:
                            face.embedding = None
                return faces

        monkeypatch.setattr(app_module, "create_engine", lambda settings: Engine())
        settings = make_settings(
            addons=("liveness",),
            liveness_mode=mode,
            liveness_compare_scope=scope,
            liveness_on_registration=on_registration,
        )
        client = TestClient(app_module.create_app(settings))
        client.__enter__()
        clients.append(client)
        return client, state

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.mark.parametrize(
    "result,code",
    [
        ({"status": "ok", "is_live": False, "live_score": 0.1}, "liveness_fake"),
        (
            {"status": "input_rejected", "is_live": None, "live_score": None},
            "liveness_input_rejected",
        ),
        (INPUT_REJECTED, "liveness_input_rejected"),
    ],
)
def test_detect_preserves_liveness_and_normal_operations_report_distinct_errors(
    liveness_client, image_bytes, create_collection, result, code
):
    client, state = liveness_client(result=result, on_registration=True)
    collection = create_collection(client)
    image = ("face.png", image_bytes(), "image/png")
    response = client.post("/v1/detect", files={"image": image})
    assert response.status_code == 200
    assert response.json()["faces"][0]["liveness"] == result
    for path in ("/v1/embeddings", f"/v1/collections/{collection['id']}/search"):
        response = client.post(path, files={"image": image})
        assert response.status_code == 422, response.text
        assert response.json()["error"]["code"] == code
        assert response.json()["error"]["details"]["liveness"] == result
        if code == "liveness_input_rejected":
            assert response.json()["error"]["message"] == result.get(
                "reason", "Input does not meet liveness requirements. Adjust face position and retry."
            )
    response = client.post("/v1/compare", files={"source": image, "target": image})
    assert response.status_code == 422
    assert response.json()["error"]["details"]["side"] == "source"
    assert response.json()["error"]["details"]["liveness"] == result
    if "reason" in result:
        assert response.json()["error"]["message"] == result["reason"]
    response = client.post(
        f"/v1/collections/{collection['id']}/persons",
        data={"id": "alice", "review_mode": "off", "liveness_on_registration": "false"},
        files={"images": image},
    )
    assert response.status_code == 422, response.text
    rejected = response.json()["error"]["details"]["rejected_images"][0]
    assert response.json()["error"]["code"] == "registration_failed"
    assert rejected["reason"] == code and rejected["liveness"] == result
    state["result"] = {"status": "ok", "is_live": True, "live_score": 0.99}
    accepted = client.post(
        f"/v1/collections/{collection['id']}/persons",
        data={"id": "alice", "review_mode": "off"},
        files={"images": image},
    )
    assert accepted.status_code == 201, accepted.text
    state["result"] = result
    added = client.post(
        f"/v1/collections/{collection['id']}/persons/alice/faces",
        data={"review_mode": "off"},
        files={"images": image},
    )
    assert added.status_code == 201 and added.json()["faces"] == [], added.text
    assert added.json()["rejected_images"][0]["reason"] == code
    assert added.json()["rejected_images"][0]["liveness"] == result


@pytest.mark.parametrize("result", [
    {"status": "ok", "is_live": False, "live_score": 0.1},
    {"status": "input_rejected", "is_live": None, "live_score": None},
    INPUT_REJECTED,
])
def test_observe_keeps_matching_and_persists_registration_snapshot(
    liveness_client, image_bytes, create_collection, result
):
    client, state = liveness_client(mode="observe", result=result, on_registration=True)
    collection = create_collection(client)
    image = ("face.png", image_bytes(), "image/png")
    response = client.post("/v1/compare", files={"source": image, "target": image})
    assert response.status_code == 200
    assert response.json()["matched"] is True
    assert response.json()["source_face"]["liveness"] == result
    response = client.post(
        f"/v1/collections/{collection['id']}/persons",
        data={"id": "alice", "review_mode": "off"},
        files={"images": image},
    )
    assert response.status_code == 201, response.text
    face_id = response.json()["faces"][0]["id"]
    added = client.post(
        f"/v1/collections/{collection['id']}/persons/alice/faces",
        data={"review_mode": "off"},
        files={"images": image},
    )
    assert added.status_code == 201 and added.json()["faces"][0]["liveness"] == result
    searched = client.post(
        f"/v1/collections/{collection['id']}/search", files={"image": image},
    )
    assert searched.status_code == 200, searched.text
    assert searched.json()["searched_face"]["liveness"] == result
    embedded = client.post("/v1/embeddings", files={"image": image})
    assert embedded.status_code == 200, embedded.text
    assert embedded.json()["faces"][0]["liveness"] == result
    state["result"] = {"status": "ok", "is_live": True, "live_score": 0.99}
    fetched = client.get(f"/v1/collections/{collection['id']}/persons/alice/faces")
    assert fetched.status_code == 200, fetched.text
    stored = next(face for face in fetched.json()["faces"] if face["id"] == face_id)
    assert stored["liveness"] == result
    state["fail"] = True
    failed = client.post("/v1/detect", files={"image": image})
    assert failed.status_code == 503
    assert failed.json()["error"]["code"] == "liveness_unavailable"


def test_compare_scope_is_server_controlled(liveness_client, image_bytes):
    client, state = liveness_client(
        scope="target", result={"status": "ok", "is_live": False, "live_score": 0.1}
    )
    image = ("face.png", image_bytes(), "image/png")
    result = client.post("/v1/compare", files={"source": image, "target": image})
    assert result.status_code == 422
    assert result.json()["error"]["details"]["side"] == "target"
    assert state["calls"] == [(False, True), (True, True)]


def test_external_trusted_enrollment_does_not_bypass_liveness(
    liveness_client, image_bytes, create_collection
):
    client, state = liveness_client(
        result={"status": "ok", "is_live": False, "live_score": 0.1},
        on_registration=True,
    )
    collection = create_collection(client)
    response = client.post(
        f"/v1/collections/{collection['id']}/persons",
        data={
            "id": "alice",
            "review_mode": "off",
            "embedding_mode": "external_trusted",
            "embedding_contract_id": collection["embedding_contract_id"],
            "external_embeddings": json.dumps([[1.0] + [0.0] * 511]),
        },
        files={"images": ("face.png", image_bytes(), "image/png")},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["details"]["rejected_images"][0]["reason"] == "liveness_fake"
    assert state["calls"] == [(True, False)]


@pytest.mark.parametrize("review_mode", ["off", "standard", "strict"])
@pytest.mark.parametrize("embedding_mode", ["server", "external_trusted"])
@pytest.mark.parametrize(
    "result",
    [
        {"status": "ok", "is_live": False, "live_score": 0.1},
        {"status": "input_rejected", "is_live": None, "live_score": None},
        INPUT_REJECTED,
    ],
)
def test_registration_defaults_to_skipping_liveness_for_new_and_added_faces(
    liveness_client,
    image_bytes,
    create_collection,
    review_mode,
    embedding_mode,
    result,
):
    client, state = liveness_client(result=result)
    collection = create_collection(client)
    image = ("face.png", image_bytes(), "image/png")
    data = {"review_mode": review_mode, "embedding_mode": embedding_mode}
    if embedding_mode == "external_trusted":
        data.update(
            embedding_contract_id=collection["embedding_contract_id"],
            external_embeddings=json.dumps([[1.0] + [0.0] * 511]),
        )
    base = f"/v1/collections/{collection['id']}/persons"
    created = client.post(base, data={"id": "alice", **data}, files={"images": image})
    added = client.post(f"{base}/alice/faces", data=data, files={"images": image})
    for response in [created, added]:
        assert response.status_code == 201, response.text
        assert len(response.json()["faces"]) == 1
        assert "liveness" not in response.json()["faces"][0]
        assert response.json()["rejected_images"] == []
    assert state["calls"] == [(False, embedding_mode == "server")] * 2
    stored = client.get(f"{base}/alice/faces").json()["faces"]
    assert len(stored) == 2 and all("liveness" not in face for face in stored)
    assert client.get("/v1/system").json()["safe_config"]["liveness_on_registration"] is False
    detected = client.post("/v1/detect", files={"image": image})
    assert detected.status_code == 200 and detected.json()["faces"][0]["liveness"] == result
    blocked = client.post("/v1/embeddings", files={"image": image})
    assert blocked.status_code == 422
    assert blocked.json()["error"]["code"] in {"liveness_fake", "liveness_input_rejected"}


@pytest.mark.parametrize("mode", ["normal", "observe"])
def test_skipped_registration_does_not_call_a_failing_liveness_model(
    liveness_client,
    image_bytes,
    create_collection,
    mode,
):
    client, state = liveness_client(mode=mode)
    collection = create_collection(client)
    state["fail"] = True
    image = ("face.png", image_bytes(), "image/png")
    created = client.post(
        f"/v1/collections/{collection['id']}/persons",
        data={"id": "alice"},
        files={"images": image},
    )
    assert created.status_code == 201, created.text
    assert "liveness" not in created.json()["faces"][0]
    detected = client.post("/v1/detect", files={"image": image})
    assert detected.status_code == 503
    assert detected.json()["error"]["code"] == "liveness_unavailable"


@pytest.mark.parametrize("result", [
    {"status": "ok", "is_live": False, "live_score": 0.1}, INPUT_REJECTED,
])
def test_video_search_never_queries_index_for_blocked_faces(
    liveness_client, image_bytes, create_collection, monkeypatch, result
):
    from insightface_server.services.images import ImageLoader

    client, state = liveness_client(result=result)
    collection = create_collection(client)
    service = client.app.state.service

    def unexpected(*args, **kwargs):
        pytest.fail("A liveness-blocked face must not query the recognition index")

    monkeypatch.setattr(service.search_indexes, "search", unexpected)
    image = ImageLoader(service.settings).from_bytes(image_bytes(), filename="frame.png")
    faces, _ = service.search_all_faces(collection["id"], image, max_faces=10, threshold=None)
    assert len(faces) == 1 and faces[0]["status"] == "liveness_blocked"
    assert faces[0]["match"] is None
    assert faces[0]["face"]["liveness"] == result


@pytest.mark.parametrize("result", [
    {"status": "ok", "is_live": True, "live_score": 0.99},
    {"status": "ok", "is_live": False, "live_score": 0.1},
    {"status": "input_rejected", "is_live": None, "live_score": None},
    INPUT_REJECTED,
])
def test_liveness_schema_omits_only_missing_reason_and_keeps_rejection_nulls(result):
    model = LivenessResult.model_validate(result)
    assert model.model_dump() == result
    assert json.loads(model.model_dump_json()) == result
    if "reason" not in result:
        assert LivenessResult.model_validate({**result, "reason": None}).model_dump() == result


def test_strict_registration_identity_rejection_preserves_liveness(
    liveness_client, image_bytes, create_collection, monkeypatch,
):
    client, _ = liveness_client(mode="observe", result=INPUT_REJECTED, on_registration=True)
    collection = create_collection(client)
    monkeypatch.setattr(
        client.app.state.search_indexes,
        "best_other_person",
        lambda *args, **kwargs: {"person_id": "bob", "face_id": "bob-face", "similarity": 1.0},
    )
    sample = image_bytes(102)
    response = client.post(
        f"/v1/collections/{collection['id']}/persons",
        data={"id": "alice", "review_mode": "strict"},
        files=[
            ("images", ("bootstrap.png", sample, "image/png")),
            ("images", ("tie.png", sample, "image/png")),
        ],
    )
    assert response.status_code == 201, response.text
    assert len(response.json()["faces"]) == 1
    assert response.json()["faces"][0]["liveness"] == INPUT_REJECTED
    rejected = response.json()["rejected_images"][0]
    assert rejected["reason"] == "identity_similarity_conflict"
    assert rejected["liveness"] == INPUT_REJECTED


def test_embedding_unavailable_preserves_completed_liveness(liveness_client, image_bytes):
    client, state = liveness_client(mode="observe", result=INPUT_REJECTED)
    state["missing_embedding"] = True
    response = client.post(
        "/v1/embeddings", files={"image": ("face.png", image_bytes(), "image/png")},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "embedding_unavailable"
    assert response.json()["error"]["details"]["liveness"] == INPUT_REJECTED


@pytest.mark.parametrize("mode", ["normal", "observe"])
def test_liveness_faults_abort_image_registration_and_video_operations(
    liveness_client, image_bytes, create_collection, monkeypatch, mode,
):
    from insightface_server.services.images import ImageLoader

    client, state = liveness_client(mode=mode, on_registration=True)
    collection = create_collection(client)
    base = f"/v1/collections/{collection['id']}"
    image = ("face.png", image_bytes(), "image/png")
    registered = client.post(
        f"{base}/persons", data={"id": "alice"}, files={"images": image},
    )
    assert registered.status_code == 201, registered.text
    state["fail"] = True
    service = client.app.state.service

    def unexpected(*args, **kwargs):
        pytest.fail("A liveness failure must abort identity search")

    monkeypatch.setattr(service.search_indexes, "search", unexpected)
    requests = [
        ("/v1/detect", {"files": {"image": image}}),
        ("/v1/embeddings", {"files": {"image": image}}),
        ("/v1/compare", {"files": {"source": image, "target": image}}),
        (f"{base}/search", {"files": {"image": image}}),
        (f"{base}/persons", {"data": {"id": "bob"}, "files": {"images": image}}),
        (f"{base}/persons/alice/faces", {"files": {"images": image}}),
    ]
    for path, options in requests:
        response = client.post(path, **options)
        assert response.status_code == 503, response.text
        assert response.json()["error"]["code"] == "liveness_unavailable"
        assert "liveness" not in response.json()["error"]["details"]
    assert client.get(f"{base}/persons/bob").status_code == 404
    assert len(client.get(f"{base}/persons/alice/faces").json()["faces"]) == 1
    loaded = ImageLoader(service.settings).from_bytes(image_bytes(), filename="frame.png")
    with pytest.raises(ApiError) as caught:
        service.search_all_faces(collection["id"], loaded, max_faces=10, threshold=None)
    assert caught.value.status_code == 503 and caught.value.code == "liveness_unavailable"
