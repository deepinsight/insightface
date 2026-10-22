"""Opt-in comparison of real CPU and CUDA model execution.

Run this on an NVIDIA host with ``onnxruntime-gpu==1.27.0``, private models,
``INSIGHTFACE_RUN_REAL_CONSISTENCY=1`` and at least two private image paths in
``INSIGHTFACE_TEST_IMAGES_JSON``.
"""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from insightface_server.inference import cosine_similarity, create_engine

pytestmark = pytest.mark.consistency


if os.getenv("INSIGHTFACE_RUN_REAL_CONSISTENCY") != "1":
    pytest.skip(
        "CPU/CUDA consistency is opt-in; set INSIGHTFACE_RUN_REAL_CONSISTENCY=1",
        allow_module_level=True,
    )


def _images() -> list[np.ndarray]:
    raw = os.getenv("INSIGHTFACE_TEST_IMAGES_JSON")
    if raw is None:
        pytest.fail(
            "INSIGHTFACE_TEST_IMAGES_JSON must be a JSON array of at least two "
            "private image paths"
        )
    try:
        paths = json.loads(raw)
    except json.JSONDecodeError as exc:
        pytest.fail(f"INSIGHTFACE_TEST_IMAGES_JSON is invalid JSON: {exc}")
    if not isinstance(paths, list) or len(paths) < 2 or not all(
        isinstance(path, str) and path for path in paths
    ):
        pytest.fail("consistency tests require at least two private image paths")
    images: list[np.ndarray] = []
    for value in paths:
        path = Path(value)
        if not path.is_file():
            pytest.fail(f"private test image does not exist: {path}")
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            pytest.fail(f"private test image is not decodable: {path}")
        images.append(image)
    return images


def _engine(provider: str):
    models_dir = Path(os.getenv("INSIGHTFACE_MODELS_DIR", "/models"))
    if not (models_dir / "manifest.json").is_file():
        pytest.fail(f"model manifest is missing: {models_dir / 'manifest.json'}")
    instance = create_engine(
        SimpleNamespace(
            inference_mode="onnx",
            execution_provider=provider,
            models_dir=models_dir,
            device_id=int(os.getenv("INSIGHTFACE_DEVICE_ID", "0")),
            detector_threshold=0.5,
        )
    )
    instance.startup()
    return instance


def test_cpu_cuda_real_inference_consistency() -> None:
    images = _images()
    box_tolerance = float(os.getenv("INSIGHTFACE_BOX_TOLERANCE_PIXELS", "3.0"))
    similarity_tolerance = float(os.getenv("INSIGHTFACE_SIMILARITY_TOLERANCE", "0.02"))
    threshold = float(os.getenv("INSIGHTFACE_TEST_THRESHOLD", "0.3"))
    assert box_tolerance >= 0.0
    assert 0.0 <= similarity_tolerance <= 1.0
    assert 0.0 <= threshold <= 1.0

    cpu = _engine("CPUExecutionProvider")
    cuda = _engine("CUDAExecutionProvider")
    with ExitStack() as stack:
        stack.callback(cuda.close)
        stack.callback(cpu.close)
        assert cuda.runtime_summary()["strict_provider_audit"]
        assert cpu.summary.model_digest == cuda.summary.model_digest
        assert cpu.summary.embedding_dimension == cuda.summary.embedding_dimension

        cpu_faces = [cpu.analyze(image, require_embeddings=True) for image in images]
        cuda_faces = [cuda.analyze(image, require_embeddings=True) for image in images]
        for cpu_result, cuda_result in zip(cpu_faces, cuda_faces, strict=True):
            assert cpu_result and cuda_result
            assert len(cpu_result) == len(cuda_result)
            assert [face.area for face in cpu_result] == sorted(
                (face.area for face in cpu_result), reverse=True
            )
            assert [face.area for face in cuda_result] == sorted(
                (face.area for face in cuda_result), reverse=True
            )
            for cpu_face, cuda_face in zip(cpu_result, cuda_result, strict=True):
                np.testing.assert_allclose(
                    cpu_face.bbox, cuda_face.bbox, atol=box_tolerance, rtol=0.0
                )
                assert cpu_face.embedding is not None
                assert cuda_face.embedding is not None
                assert cpu_face.embedding.shape == cuda_face.embedding.shape
                assert cpu_face.embedding.shape == (cpu.summary.embedding_dimension,)
                assert np.linalg.norm(cpu_face.embedding) == pytest.approx(1.0, abs=1e-5)
                assert np.linalg.norm(cuda_face.embedding) == pytest.approx(1.0, abs=1e-5)

        cpu_query = cpu_faces[0][0].embedding
        cuda_query = cuda_faces[0][0].embedding
        assert cpu_query is not None and cuda_query is not None
        cpu_scores: list[tuple[int, float]] = []
        cuda_scores: list[tuple[int, float]] = []
        for index, (cpu_result, cuda_result) in enumerate(
            zip(cpu_faces, cuda_faces, strict=True)
        ):
            cpu_embedding = cpu_result[0].embedding
            cuda_embedding = cuda_result[0].embedding
            assert cpu_embedding is not None and cuda_embedding is not None
            cpu_scores.append((index, cosine_similarity(cpu_query, cpu_embedding)))
            cuda_scores.append((index, cosine_similarity(cuda_query, cuda_embedding)))

        for (_, cpu_score), (_, cuda_score) in zip(cpu_scores, cuda_scores, strict=True):
            assert abs(cpu_score - cuda_score) <= similarity_tolerance
            assert (cpu_score >= threshold) == (cuda_score >= threshold)

        cpu_ranking = [
            index for index, _ in sorted(cpu_scores, key=lambda item: (-item[1], item[0]))
        ]
        cuda_ranking = [
            index for index, _ in sorted(cuda_scores, key=lambda item: (-item[1], item[0]))
        ]
        assert cpu_ranking == cuda_ranking
        assert cpu_ranking[0] == 0
