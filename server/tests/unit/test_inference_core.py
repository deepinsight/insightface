from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from insightface_server.inference import (
    EngineSummary,
    FaceObservation,
    InferenceEngine,
    cosine_similarity,
    create_engine,
    l2_normalize,
    raw_cosine_similarity,
)
from insightface_server.inference.concurrency import InferenceConcurrencyLimiter
from insightface_server.inference.mock import MockInferenceEngine
from insightface_server.inference.quality import (
    RegistrationQualityPolicy,
    enrich_quality,
    registration_rejection_reasons,
)
from insightface_server.services.core import FaceService


def test_l2_normalization_and_raw_cosine_range() -> None:
    vector = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert raw_cosine_similarity(vector, vector) == pytest.approx(1.0)
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)
    assert cosine_similarity(vector, -vector) == pytest.approx(-1.0)
    assert cosine_similarity(np.array([1, 0]), np.array([0, 1])) == pytest.approx(0.0)


def test_embedding_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="zero or non-finite"):
        l2_normalize(np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="dimensions"):
        cosine_similarity(np.ones(2), np.ones(3))


def test_external_trusted_embedding_validation_is_strict_and_canonical() -> None:
    accepted = FaceService._trusted_embedding([1.0001, 0.0, 0.0], 3)
    assert accepted is not None
    assert accepted.dtype == np.float32
    np.testing.assert_array_equal(accepted, np.array([1.0, 0.0, 0.0], np.float32))

    assert FaceService._trusted_embedding([1.001, 0.0, 0.0], 3) is None
    assert FaceService._trusted_embedding([0.0, 0.0, 0.0], 3) is None
    assert FaceService._trusted_embedding([1.0, 0.0], 3) is None
    assert FaceService._trusted_embedding([True, 0.0, 0.0], 3) is None
    assert FaceService._trusted_embedding([float("nan"), 0.0, 0.0], 3) is None


def test_face_observation_bbox_helpers() -> None:
    face = FaceObservation(
        bbox=(10, 20, 50, 80),
        detection_score=0.9,
        landmarks=None,
        embedding=None,
    )
    assert face.area == 2400
    assert face.confidence == 0.9
    assert face.bbox_xywh == (10, 20, 40, 60)
    assert face.normalized_bbox(100, 200) == pytest.approx((0.1, 0.1, 0.4, 0.3))
    clipped = FaceObservation((-10, -20, 50, 80), 0.9, None, None)
    assert clipped.normalized_bbox(100, 200) == pytest.approx((0, 0, 0.5, 0.4))


def test_quality_metrics_and_registration_reasons_are_normalized() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:80, 20:80] = np.indices((60, 60)).sum(axis=0)[..., None] % 2 * 255
    landmarks = np.array([[35, 40], [65, 40], [50, 52], [39, 66], [61, 66]], dtype=np.float32)
    face = FaceObservation((20, 20, 80, 80), 0.99, landmarks, None)
    enrich_quality(image, face)
    assert 0.0 <= face.brightness <= 1.0
    assert 0.0 <= face.sharpness <= 1.0
    assert 0.0 <= face.quality_score <= 1.0
    assert registration_rejection_reasons(face) == []

    weak = FaceObservation((0, 0, 12, 12), 0.1, landmarks, None)
    enrich_quality(np.zeros((20, 20, 3), dtype=np.uint8), weak)
    reasons = registration_rejection_reasons(weak, RegistrationQualityPolicy(min_quality_score=0.9))
    assert reasons == ["face_too_small", "low_detection_score", "low_quality"]


def test_mock_engine_is_deterministic_sorted_and_l2_normalized() -> None:
    engine = MockInferenceEngine(embedding_dimension=32)
    assert isinstance(engine, InferenceEngine)
    engine.startup()
    generator = np.random.default_rng(42)
    image = generator.integers(0, 256, (80, 180, 3), dtype=np.uint8)
    first = engine.analyze(image)
    second = engine.analyze(image)
    assert len(first) == 2
    assert first[0].area > first[1].area
    assert first[0].embedding is not None
    assert np.linalg.norm(first[0].embedding) == pytest.approx(1.0)
    np.testing.assert_array_equal(first[0].embedding, second[0].embedding)
    assert all(0.0 <= face.detection_score <= 1.0 for face in first)
    assert len(engine.analyze(image, max_faces=1)) == 1
    assert engine.analyze(image, min_score=1.0) == []


def test_mock_engine_blank_image_and_lifecycle() -> None:
    engine = MockInferenceEngine()
    with pytest.raises(RuntimeError, match="not been started"):
        engine.analyze(np.zeros((64, 64, 3), dtype=np.uint8))
    engine.startup()
    assert engine.analyze(np.zeros((64, 64, 3), dtype=np.uint8)) == []
    assert engine.runtime_summary()["mode"] == "mock"
    assert engine.runtime_summary()["cpu"]["logical_cores"]
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.startup()


def test_global_inference_limiter_allows_parallel_work_and_bounds_active_count() -> None:
    limiter = InferenceConcurrencyLimiter(2)
    release = threading.Event()
    two_active = threading.Event()
    state_lock = threading.Lock()
    active = 0
    observed_peak = 0

    def work() -> None:
        nonlocal active, observed_peak
        with limiter.slot():
            with state_lock:
                active += 1
                observed_peak = max(observed_peak, active)
                if active == 2:
                    two_active.set()
            assert release.wait(timeout=5)
            with state_lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(work) for _ in range(4)]
        assert two_active.wait(timeout=5)
        deadline = time.monotonic() + 5
        while limiter.summary()["waiting"] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert limiter.summary()["active"] == 2
        assert limiter.summary()["waiting"] == 2
        release.set()
        for future in futures:
            future.result(timeout=5)

    assert observed_peak == 2
    assert limiter.summary() == {
        "max_concurrency": 2,
        "active": 0,
        "waiting": 0,
        "peak_active": 2,
    }


def test_factory_and_summary_use_simple_public_contract() -> None:
    engine = create_engine(
        SimpleNamespace(
            inference_mode="mock",
            execution_provider="CPUExecutionProvider",
            embedding_dimension=16,
        )
    )
    assert engine.summary.embedding_dimension == 16
    summary = EngineSummary("id", "1", "a" * 64, 16, "1", "CPUExecutionProvider")
    assert summary.as_dict()["model_id"] == "id"
    with pytest.raises(ValueError, match="Unsupported inference_mode"):
        create_engine({"inference_mode": "remote"})


def _write_manifest(directory: Path) -> None:
    detector = directory / "detector.onnx"
    recognizer = directory / "recognizer.onnx"
    detector.write_bytes(b"detector")
    recognizer.write_bytes(b"recognizer")
    manifest = {
        "manifest_version": 1,
        "model_id": "buffalo_l",
        "model_version": "v0.7",
        "files": {"detector": detector.name, "recognizer": recognizer.name},
        "recognition": {
            "input_size": [112, 112],
            "embedding_dimension": 512,
            "preprocessing": "insightface-arcface-1",
        },
        "license": "MODEL.LICENSE",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    default_license = (
        Path(__file__).resolve().parents[2]
        / "backend"
        / "insightface_server"
        / "licensing"
        / "defaults"
        / "buffalo_l"
        / "MODEL.LICENSE"
    )
    (directory / "MODEL.LICENSE").write_bytes(default_license.read_bytes())


def test_onnx_engine_startup_is_once_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from insightface_server.inference.onnx_engine import OnnxInsightFaceEngine

    _write_manifest(tmp_path)
    engine = OnnxInsightFaceEngine(
        SimpleNamespace(
            models_dir=tmp_path,
            execution_provider="CPUExecutionProvider",
            detector_threshold=0.5,
            device_id=0,
        )
    )
    assert len(engine.summary.model_digest) == 64
    assert engine.summary.model_digest != engine.bundle.recognizer.sha256
    calls = 0

    def fake_startup() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(engine, "_startup_once", fake_startup)
    engine.startup()
    engine.startup()
    assert calls == 1


def test_onnx_engine_refuses_tampered_model_license(tmp_path: Path) -> None:
    from insightface_server.inference.onnx_engine import OnnxInsightFaceEngine

    _write_manifest(tmp_path)
    license_path = tmp_path / "MODEL.LICENSE"
    document = json.loads(license_path.read_text(encoding="utf-8"))
    document["grant"] = "commercial"
    document["customer"] = "Tampered Customer"
    license_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="signature verification failed"):
        OnnxInsightFaceEngine(
            SimpleNamespace(
                models_dir=tmp_path,
                execution_provider="CPUExecutionProvider",
                detector_threshold=0.5,
                device_id=0,
            )
        )
