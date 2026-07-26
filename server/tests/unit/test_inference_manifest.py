from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from insightface_server.models import load_manifest


def _models(tmp_path: Path) -> tuple[Path, Path]:
    detector = tmp_path / "detector.onnx"
    recognizer = tmp_path / "recognizer.onnx"
    detector.write_bytes(b"detector")
    recognizer.write_bytes(b"recognizer")
    (tmp_path / "MODEL.LICENSE").write_text("{}", encoding="utf-8")
    return detector, recognizer


def _manifest(detector: Path, recognizer: Path) -> dict[str, object]:
    return {
        "manifest_version": 1,
        "model_id": "test_model",
        "model_version": "v1",
        "display_name": "Test model",
        "files": {
            "detector": detector.name,
            "recognizer": recognizer.name,
        },
        "recognition": {
            "input_size": [112, 112],
            "embedding_dimension": 512,
            "preprocessing": "arcface-v1",
        },
        "license": "MODEL.LICENSE",
    }


def test_loads_compact_manifest_and_calculates_diagnostic_digest(tmp_path: Path) -> None:
    detector, recognizer = _models(tmp_path)
    manifest = _manifest(detector, recognizer)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    bundle = load_manifest(tmp_path)

    assert bundle.model_id == "test_model"
    assert bundle.model_version == "v1"
    assert bundle.detector.task == "face_detection"
    assert bundle.recognizer.embedding_dimension == 512
    assert bundle.recognizer.public_summary()["model_id"] == "test_model"
    assert bundle.recognizer.sha256 == hashlib.sha256(b"recognizer").hexdigest()
    assert bundle.license_path.name == "MODEL.LICENSE"


def test_converted_model_content_does_not_require_manifest_hash_update(tmp_path: Path) -> None:
    detector, recognizer = _models(tmp_path)
    manifest = _manifest(detector, recognizer)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    original = load_manifest(tmp_path).recognizer.sha256

    recognizer.write_bytes(b"converted-fp16-recognizer")
    converted = load_manifest(tmp_path)

    assert converted.model_id == "test_model"
    assert converted.recognizer.sha256 != original


def test_rejects_path_escape_and_wrong_license_filename(tmp_path: Path) -> None:
    detector, recognizer = _models(tmp_path)
    manifest = _manifest(detector, recognizer)
    files = manifest["files"]
    assert isinstance(files, dict)
    files["detector"] = "../escaped.onnx"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unsafe .onnx path"):
        load_manifest(tmp_path)

    manifest = _manifest(detector, recognizer)
    manifest["license"] = "license.json"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="MODEL.LICENSE"):
        load_manifest(tmp_path)


def test_recognition_contract_is_required(tmp_path: Path) -> None:
    detector, recognizer = _models(tmp_path)
    manifest = _manifest(detector, recognizer)
    recognition = manifest["recognition"]
    assert isinstance(recognition, dict)
    del recognition["embedding_dimension"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="recognition must contain exactly"):
        load_manifest(tmp_path)


def test_legacy_manifest_is_read_without_enforcing_declared_sha256(tmp_path: Path) -> None:
    detector, recognizer = _models(tmp_path)
    legacy = {
        "package": {"name": "buffalo_l", "release": "v0.7"},
        "models": [
            {
                "model_id": "scrfd-detection",
                "model_version": "1",
                "task": "face_detection",
                "file": detector.name,
                "input_size": [640, 640],
                "preprocessing_version": "insightface-scrfd-1",
                "sha256": "0" * 64,
            },
            {
                "model_id": "buffalo_l-recognition",
                "model_version": "1",
                "task": "face_recognition",
                "file": recognizer.name,
                "input_size": [112, 112],
                "embedding_dimension": 512,
                "preprocessing_version": "insightface-arcface-1",
                "sha256": "0" * 64,
            },
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(legacy), encoding="utf-8")
    bundle = load_manifest(tmp_path)
    assert bundle.legacy_manifest is True
    assert bundle.model_id == "buffalo_l"
