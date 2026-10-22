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


def _v2_manifest(detector: Path, recognizer: Path) -> dict[str, object]:
    return {
        "manifest_version": 2,
        "model_id": "raccoon_s",
        "display_name": "Raccoon S",
        "license": "MODEL.LICENSE",
        "tasks": {
            "detection": {
                "file": detector.name,
                "sha256": hashlib.sha256(b"detector").hexdigest(),
                "preprocessing": {"mean": 125.0, "std": 126.0},
            },
            "verification": {
                "file": "missing-verifier.onnx",
                "sha256": "not-checked-by-server",
                "preprocessing": "embedded",
                "expansion": -1,
            },
            "recognition": {
                "file": recognizer.name,
                "sha256": hashlib.sha256(b"recognizer").hexdigest(),
                "preprocessing": "embedded",
                "input_size": [112, 112],
                "embedding_dimension": 512,
                "preprocessing_version": "insightface-arcface-1",
            },
        },
        "ignored_root_extension": {"future": True},
    }


def test_loads_compact_manifest_and_calculates_diagnostic_digest(tmp_path: Path) -> None:
    detector, recognizer = _models(tmp_path)
    manifest = _manifest(detector, recognizer)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    bundle = load_manifest(tmp_path)

    assert bundle.model_id == "test_model"
    assert not hasattr(bundle, "model_version")
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


def test_v2_loads_only_active_models_and_ignores_verifier_and_extensions(
    tmp_path: Path,
) -> None:
    detector, recognizer = _models(tmp_path)
    manifest = _v2_manifest(detector, recognizer)
    manifest["model_version"] = "ignored-version"
    tasks = manifest["tasks"]
    assert isinstance(tasks, dict)
    tasks["future_task"] = "ignored"
    detection = tasks["detection"]
    assert isinstance(detection, dict)
    detection["future_metadata"] = {"ignored": True}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    bundle = load_manifest(tmp_path)

    assert bundle.model_id == "raccoon_s"
    assert not hasattr(bundle, "model_version")
    assert bundle.detector.preprocessing == "mean_std"
    assert bundle.detector.input_mean == 125.0
    assert bundle.detector.input_std == 126.0
    assert bundle.recognizer.preprocessing == "embedded"
    assert bundle.recognizer.input_mean == 0.0
    assert bundle.recognizer.input_std == 1.0
    assert bundle.models == (bundle.detector, bundle.recognizer)
    assert bundle.detector.sha256 == hashlib.sha256(b"detector").hexdigest()
    assert bundle.recognizer.sha256 == hashlib.sha256(b"recognizer").hexdigest()


@pytest.mark.parametrize("task", ("detection", "recognition"))
@pytest.mark.parametrize("declared", (None, "0" * 63, "A" * 64, "g" * 64))
def test_v2_rejects_malformed_active_task_sha256(
    tmp_path: Path,
    task: str,
    declared: object,
) -> None:
    detector, recognizer = _models(tmp_path)
    manifest = _v2_manifest(detector, recognizer)
    tasks = manifest["tasks"]
    assert isinstance(tasks, dict)
    active = tasks[task]
    assert isinstance(active, dict)
    active["sha256"] = declared
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match=rf"tasks\.{task}\.sha256 must be 64 lowercase hexadecimal characters",
    ):
        load_manifest(tmp_path)


@pytest.mark.parametrize("task", ("detection", "recognition"))
def test_v2_rejects_mismatched_active_task_sha256(tmp_path: Path, task: str) -> None:
    detector, recognizer = _models(tmp_path)
    manifest = _v2_manifest(detector, recognizer)
    tasks = manifest["tasks"]
    assert isinstance(tasks, dict)
    active = tasks[task]
    assert isinstance(active, dict)
    active["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match=rf"tasks\.{task}\.sha256 does not match model package file",
    ):
        load_manifest(tmp_path)


def test_v2_uses_server_defaults_when_optional_metadata_is_missing(tmp_path: Path) -> None:
    detector, recognizer = _models(tmp_path)
    manifest = _v2_manifest(detector, recognizer)
    tasks = manifest["tasks"]
    assert isinstance(tasks, dict)
    tasks["detection"] = {"file": detector.name}
    tasks["recognition"] = {"file": recognizer.name}
    del tasks["verification"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    bundle = load_manifest(tmp_path)

    assert bundle.detector.input_mean == 127.5
    assert bundle.detector.input_std == 128.0
    assert bundle.detector.preprocessing_version == "insightface-scrfd-1"
    assert bundle.recognizer.input_mean == 127.5
    assert bundle.recognizer.input_std == 127.5
    assert bundle.recognizer.input_size == (112, 112)
    assert bundle.recognizer.embedding_dimension == 512
    assert bundle.recognizer.preprocessing_version == "insightface-arcface-1"


@pytest.mark.parametrize("task", ("detection", "recognition"))
def test_v2_requires_both_active_tasks(tmp_path: Path, task: str) -> None:
    detector, recognizer = _models(tmp_path)
    manifest = _v2_manifest(detector, recognizer)
    tasks = manifest["tasks"]
    assert isinstance(tasks, dict)
    del tasks[task]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match=rf"tasks\.{task} must be an object"):
        load_manifest(tmp_path)


@pytest.mark.parametrize("task", ("detection", "recognition"))
def test_v2_rejects_unsafe_active_file(tmp_path: Path, task: str) -> None:
    detector, recognizer = _models(tmp_path)
    manifest = _v2_manifest(detector, recognizer)
    tasks = manifest["tasks"]
    assert isinstance(tasks, dict)
    active = tasks[task]
    assert isinstance(active, dict)
    active["file"] = "../escaped.onnx"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unsafe .onnx path"):
        load_manifest(tmp_path)


def test_v2_requires_distinct_active_paths(tmp_path: Path) -> None:
    detector, recognizer = _models(tmp_path)
    manifest = _v2_manifest(detector, recognizer)
    tasks = manifest["tasks"]
    assert isinstance(tasks, dict)
    recognition = tasks["recognition"]
    assert isinstance(recognition, dict)
    recognition["file"] = detector.name
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="distinct ONNX paths"):
        load_manifest(tmp_path)


def test_v1_rejects_v2_tasks_and_verifier(tmp_path: Path) -> None:
    detector, recognizer = _models(tmp_path)
    manifest = _manifest(detector, recognizer)
    manifest["tasks"] = {}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsupported fields: tasks"):
        load_manifest(tmp_path)

    manifest = _manifest(detector, recognizer)
    files = manifest["files"]
    assert isinstance(files, dict)
    files["verifier"] = "missing.onnx"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exactly detector and recognizer"):
        load_manifest(tmp_path)


@pytest.mark.parametrize(
    "preprocessing",
    ("external", {"mean": 127.5}, {"mean": 127.5, "std": 0.0}),
)
def test_v2_rejects_invalid_active_preprocessing(tmp_path: Path, preprocessing: object) -> None:
    detector, recognizer = _models(tmp_path)
    manifest = _v2_manifest(detector, recognizer)
    tasks = manifest["tasks"]
    assert isinstance(tasks, dict)
    detection = tasks["detection"]
    assert isinstance(detection, dict)
    detection["preprocessing"] = preprocessing
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="tasks.detection.preprocessing"):
        load_manifest(tmp_path)


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


def test_old_manifest_version_metadata_does_not_change_model_identity(tmp_path: Path) -> None:
    detector, recognizer = _models(tmp_path)
    manifest = _manifest(detector, recognizer)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    original = load_manifest(tmp_path)
    manifest['model_version'] = 'another-obsolete-version'
    path.write_text(json.dumps(manifest))
    assert load_manifest(tmp_path) == original
    manifest.pop('model_version')
    path.write_text(json.dumps(manifest))
    assert load_manifest(tmp_path) == original
    assert all('model_version' not in model.public_summary() for model in original.models)
