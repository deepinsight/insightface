"""Opt-in CPU checks against the private models and images mounted by a developer.

Enable with ``INSIGHTFACE_RUN_REAL_CPU=1`` and provide a JSON array of at least
two non-public image paths in ``INSIGHTFACE_TEST_IMAGES_JSON``. The model bundle
is read from ``INSIGHTFACE_MODELS_DIR`` (``/models`` by default).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from insightface_server.inference import cosine_similarity, create_engine

pytestmark = pytest.mark.cpu


def _require_enabled() -> None:
    if os.getenv("INSIGHTFACE_RUN_REAL_CPU") != "1":
        pytest.skip(
            "real CPU inference is opt-in; set INSIGHTFACE_RUN_REAL_CPU=1",
            allow_module_level=True,
        )


_require_enabled()


def _private_images() -> list[np.ndarray]:
    raw = os.getenv("INSIGHTFACE_TEST_IMAGES_JSON")
    if raw is None:
        pytest.fail(
            "INSIGHTFACE_TEST_IMAGES_JSON must be a JSON array of at least two "
            "private image paths when real CPU tests are enabled"
        )
    try:
        paths = json.loads(raw)
    except json.JSONDecodeError as exc:
        pytest.fail(f"INSIGHTFACE_TEST_IMAGES_JSON is invalid JSON: {exc}")
    if not isinstance(paths, list) or len(paths) < 2 or not all(
        isinstance(path, str) and path for path in paths
    ):
        pytest.fail("INSIGHTFACE_TEST_IMAGES_JSON must contain at least two paths")

    images: list[np.ndarray] = []
    payloads: list[bytes] = []
    for value in paths:
        path = Path(value)
        if not path.is_file():
            pytest.fail(f"private test image does not exist: {path}")
        payload = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            pytest.fail(f"private test image is not decodable: {path}")
        payloads.append(payload)
        images.append(image)
    if len(set(payloads)) != len(payloads):
        pytest.fail("real CPU Top-K checks require distinct private images")
    return images


@pytest.fixture(scope="module")
def engine():
    models_dir = Path(os.getenv("INSIGHTFACE_MODELS_DIR", "/models"))
    if not (models_dir / "manifest.json").is_file():
        pytest.fail(f"model manifest is missing: {models_dir / 'manifest.json'}")
    instance = create_engine(
        SimpleNamespace(
            inference_mode="onnx",
            execution_provider="CPUExecutionProvider",
            models_dir=models_dir,
            device_id=0,
            detector_threshold=0.5,
        )
    )
    instance.startup()
    yield instance
    instance.close()


def test_real_cpu_provider_warmup_and_detection_order(engine) -> None:
    runtime = engine.runtime_summary()
    assert runtime["execution_provider"] == "CPUExecutionProvider"
    assert runtime["onnx_runtime_version"] == "1.27.0"
    assert runtime["detector_session_providers"][0] == "CPUExecutionProvider"
    assert runtime["recognizer_session_providers"][0] == "CPUExecutionProvider"

    for image in _private_images():
        first = engine.analyze(image, require_embeddings=True)
        second = engine.analyze(image, require_embeddings=True)
        assert first, "each opt-in CPU image must contain a usable face"
        assert len(first) == len(second)
        assert [face.area for face in first] == sorted(
            (face.area for face in first), reverse=True
        )
        np.testing.assert_allclose(
            [face.bbox for face in first],
            [face.bbox for face in second],
            atol=1e-4,
            rtol=0.0,
        )
        for face in first:
            assert face.embedding is not None
            assert face.embedding.shape == (engine.summary.embedding_dimension,)
            assert np.linalg.norm(face.embedding) == pytest.approx(1.0, abs=1e-5)


def test_real_cpu_cosine_top_k_and_threshold_are_deterministic(engine) -> None:
    observations = [
        engine.analyze(image, require_embeddings=True, max_faces=1)[0]
        for image in _private_images()
    ]
    query = observations[0].embedding
    assert query is not None
    similarities = []
    for index, observation in enumerate(observations):
        assert observation.embedding is not None
        similarities.append((index, cosine_similarity(query, observation.embedding)))

    first_ranking = sorted(similarities, key=lambda item: (-item[1], item[0]))
    second_ranking = sorted(similarities, key=lambda item: (-item[1], item[0]))
    assert first_ranking == second_ranking
    assert first_ranking[0][0] == 0
    assert first_ranking[0][1] == pytest.approx(1.0, abs=1e-6)

    threshold = float(os.getenv("INSIGHTFACE_TEST_THRESHOLD", "0.3"))
    assert 0.0 <= threshold <= 1.0
    decisions = [similarity >= threshold for _, similarity in first_ranking]
    assert decisions[0] is True
