from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient
from insightface_server.config import DetectionProfile
from insightface_server.inference import FaceObservation
from insightface_server.services.core import FaceService


def _observation(
    bbox: tuple[float, float, float, float], embedding_index: int
) -> FaceObservation:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[embedding_index] = 1.0
    left, top, right, bottom = bbox
    landmarks = np.asarray(
        [
            [left + 10, top + 10],
            [right - 10, top + 10],
            [(left + right) / 2, (top + bottom) / 2],
            [left + 12, bottom - 10],
            [right - 12, bottom - 10],
        ],
        dtype=np.float32,
    )
    return FaceObservation(
        bbox=bbox,
        detection_score=0.99,
        landmarks=landmarks,
        embedding=embedding,
        quality_score=0.9,
        sharpness=0.9,
        brightness=0.7,
    )


def test_collection_detection_profile_snapshot_patch_and_revision(
    client: TestClient,
) -> None:
    system_profile = client.get("/v1/system").json()["safe_config"]["detection"]
    inherited = client.post(
        "/v1/collections",
        json={"id": "inherited", "name": "Inherited"},
    )
    configured = client.post(
        "/v1/collections",
        json={
            "id": "configured",
            "name": "Configured",
            "detection": {
                "input_sizes": [[128, 128], [640, 640]],
                "threshold": 0.42,
                "nms_threshold": 0.35,
                "single_face_selection": "center_largest",
            },
        },
    )

    assert inherited.status_code == 201, inherited.text
    assert inherited.json()["collection"]["detection"] == system_profile
    assert inherited.json()["collection"]["detection_revision"] == 1
    assert configured.status_code == 201, configured.text
    item = configured.json()["collection"]
    assert item["detection"] == {
        "input_sizes": [[128, 128], [640, 640]],
        "threshold": 0.42,
        "nms_threshold": 0.35,
        "single_face_selection": "center_largest",
    }

    patched = client.patch(
        "/v1/collections/configured",
        json={"detection": {"single_face_selection": "largest"}},
    )
    unchanged = client.patch(
        "/v1/collections/configured",
        json={"detection": {"single_face_selection": "largest"}},
    )

    assert patched.status_code == 200, patched.text
    assert patched.json()["collection"]["detection_revision"] == 2
    assert patched.json()["collection"]["detection"]["threshold"] == 0.42
    assert unchanged.status_code == 200, unchanged.text
    assert unchanged.json()["collection"]["detection_revision"] == 2


def test_center_largest_uses_one_center_weighted_area_score() -> None:
    edge_large = _observation((0.0, 5.0, 100.0, 95.0), 0)
    center_small = _observation((108.0, 30.0, 148.0, 70.0), 1)
    near_center_large = _observation((70.0, 5.0, 180.0, 95.0), 2)
    above_center = _observation((108.0, 0.0, 148.0, 40.0), 3)
    below_center = _observation((108.0, 60.0, 148.0, 100.0), 4)

    def score(face: FaceObservation) -> float:
        return FaceService._selection_score(
            face,
            image_width=256,
            image_height=100,
            strategy="center_largest",
        )

    assert score(edge_large) == -3168.0
    assert score(center_small) == 1600.0
    assert score(near_center_large) == 9882.0
    assert score(above_center) == score(below_center) == -200.0
    assert max([edge_large, center_small], key=score) is center_small
    # Unlike the previous lexicographic rule, exact center proximity does not
    # automatically beat a much larger face. Both terms share one score.
    assert max([center_small, near_center_large], key=score) is near_center_large


def test_collection_profile_reaches_detect_compare_embeddings_search_and_enrollment(
    client: TestClient, image_bytes, monkeypatch
) -> None:
    created = client.post(
        "/v1/collections",
        json={
            "id": "centered",
            "name": "Centered",
            "detection": {
                "threshold": 0.42,
                "single_face_selection": "center_largest",
            },
        },
    )
    assert created.status_code == 201, created.text

    edge_large = _observation((0.0, 5.0, 100.0, 95.0), 0)
    center_small = _observation((108.0, 30.0, 148.0, 70.0), 1)
    profiles: list[DetectionProfile] = []

    def analyze(_image, *, detection_profile=None, **_kwargs):
        assert detection_profile is not None
        profiles.append(detection_profile)
        return [edge_large, center_small]

    monkeypatch.setattr(client.app.state.engine, "analyze", analyze)
    image = ("wide.png", image_bytes(12, width=256, height=100), "image/png")

    detected = client.post(
        "/v1/detect",
        data={"collection_id": "centered"},
        files={"image": image},
    )
    compared = client.post(
        "/v1/compare",
        data={"collection_id": "centered"},
        files={"source": image, "target": image},
    )
    embedded = client.post(
        "/v1/embeddings",
        data={"collection_id": "centered"},
        files={"image": image},
    )
    searched = client.post(
        "/v1/collections/centered/search",
        files={"image": image},
    )
    enrolled = client.post(
        "/v1/collections/centered/persons",
        data={"id": "center-person", "review_mode": "off"},
        files={"images": image},
    )

    for response in (detected, compared, embedded, searched, enrolled):
        assert response.status_code in {200, 201}, response.text
    assert len(detected.json()["faces"]) == 2
    assert compared.json()["source_face"]["bbox"]["pixels"]["x"] == 108
    assert embedded.json()["faces"][0]["bbox"]["pixels"]["x"] == 108
    assert searched.json()["searched_face"]["bbox"]["pixels"]["x"] == 108
    assert enrolled.json()["faces"][0]["bounding_box"]["pixels"]["x"] == 108
    assert profiles
    assert all(profile.threshold == 0.42 for profile in profiles)
    assert all(profile.single_face_selection == "center_largest" for profile in profiles)
