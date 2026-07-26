"""Opt-in strict CUDA checks using private models and a private face image."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from insightface_server.inference import create_engine

pytestmark = pytest.mark.cuda


if os.getenv("INSIGHTFACE_RUN_REAL_CUDA") != "1":
    pytest.skip(
        "real CUDA inference is opt-in; set INSIGHTFACE_RUN_REAL_CUDA=1",
        allow_module_level=True,
    )


def _private_image() -> np.ndarray:
    raw = os.getenv("INSIGHTFACE_TEST_IMAGES_JSON")
    if raw is None:
        pytest.fail(
            "INSIGHTFACE_TEST_IMAGES_JSON must contain a private image path when "
            "real CUDA tests are enabled"
        )
    try:
        paths = json.loads(raw)
    except json.JSONDecodeError as exc:
        pytest.fail(f"INSIGHTFACE_TEST_IMAGES_JSON is invalid JSON: {exc}")
    if not isinstance(paths, list) or not paths or not isinstance(paths[0], str):
        pytest.fail("INSIGHTFACE_TEST_IMAGES_JSON must be a non-empty JSON array")
    path = Path(paths[0])
    if not path.is_file():
        pytest.fail(f"private test image does not exist: {path}")
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        pytest.fail(f"private test image is not decodable: {path}")
    return image


@pytest.fixture(scope="module")
def engine():
    models_dir = Path(os.getenv("INSIGHTFACE_MODELS_DIR", "/models"))
    if not (models_dir / "manifest.json").is_file():
        pytest.fail(f"model manifest is missing: {models_dir / 'manifest.json'}")
    instance = create_engine(
        SimpleNamespace(
            inference_mode="onnx",
            execution_provider="CUDAExecutionProvider",
            models_dir=models_dir,
            device_id=int(os.getenv("INSIGHTFACE_DEVICE_ID", "0")),
            detector_threshold=0.5,
        )
    )
    instance.startup()
    yield instance
    instance.close()


def test_real_cuda_is_primary_and_strict_warmup_ran(engine) -> None:
    runtime = engine.runtime_summary()
    assert runtime["execution_provider"] == "CUDAExecutionProvider"
    assert runtime["detector_session_providers"][0] == "CUDAExecutionProvider"
    assert runtime["recognizer_session_providers"][0] == "CUDAExecutionProvider"
    assert "CUDAExecutionProvider" in runtime["available_execution_providers"]
    assert runtime["onnx_runtime_version"] == "1.27.0"
    assert runtime["cuda_runtime_version"] == "12.9"
    assert runtime["cudnn_version"] == "9.24.0"
    assert runtime["gpus"]
    for gpu in runtime["gpus"]:
        assert gpu["name"]
        assert gpu["compute_capability"]
        assert gpu["driver_version"]

    audits = runtime["strict_provider_audit"]
    assert set(audits) == {"detector", "recognizer"}
    for audit in audits.values():
        assert audit["accepted"] is True
        assert audit["cuda_kernel_count"] > 0
        assert "CPU limited to small integer shape metadata" in audit["policy"]


def test_real_cuda_inference_returns_normalized_embedding(engine) -> None:
    observations = engine.analyze(
        _private_image(), require_embeddings=True, max_faces=1
    )
    assert observations, "the opt-in CUDA image must contain a usable face"
    embedding = observations[0].embedding
    assert embedding is not None
    assert embedding.shape == (engine.summary.embedding_dimension,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)
