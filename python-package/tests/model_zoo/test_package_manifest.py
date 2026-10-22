import hashlib
import json

import pytest

from insightface.model_zoo.package_manifest import (
    MODEL_PACKAGE_TASKS,
    has_model_package_manifest,
    load_model_package,
    model_package_manifest_version,
    verify_model_artifact,
)


def _write_manifest(package, manifest):
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("name", ["raccoon_s", "raccoon_l", "custom_package"])
def test_v2_exposes_identity_and_declared_tasks(manifest_package_factory, name):
    package, _manifest = manifest_package_factory(name)

    descriptor = load_model_package(package)

    assert descriptor.name == name
    assert descriptor.model_id == name
    assert descriptor.manifest_version == 2
    assert descriptor.source_schema == "unified-v2"
    assert descriptor.license_path == package / "MODEL.LICENSE"
    assert tuple(descriptor.tasks) == MODEL_PACKAGE_TASKS
    assert has_model_package_manifest(package) is True
    assert model_package_manifest_version(package) == 2


def test_known_tasks_require_only_file_and_receive_defaults(
    manifest_package_factory,
):
    package, manifest = manifest_package_factory()
    manifest["tasks"] = {
        task: {"file": value["file"]} for task, value in manifest["tasks"].items()
    }
    _write_manifest(package, manifest)

    descriptor = load_model_package(package)

    assert descriptor.task("detection").metadata == {
        "preprocessing": {"mean": 127.5, "std": 128.0},
        "preprocessing_version": "insightface-scrfd-1",
    }
    assert descriptor.task("verification").metadata == {
        "preprocessing": "embedded",
        "expansion": 1.3,
    }
    assert descriptor.task("recognition").metadata == {
        "preprocessing": {"mean": 127.5, "std": 127.5},
        "preprocessing_version": "insightface-arcface-1",
        "input_size": (112, 112),
        "embedding_dimension": 512,
    }
    for task in MODEL_PACKAGE_TASKS:
        assert "sha256" not in descriptor.task(task).as_config()


def test_unknown_root_task_and_metadata_are_ignored(manifest_package_factory):
    package, manifest = manifest_package_factory()
    manifest.update(
        {
            "model_version": "ignored",
            "sha256": "ignored",
            "future_root": {"anything": True},
        }
    )
    manifest["tasks"]["future_task"] = "opaque future value"
    manifest["tasks"]["detection"].update(
        {
            "future_metadata": [1, 2, 3],
        }
    )
    _write_manifest(package, manifest)

    descriptor = load_model_package(package)

    assert tuple(descriptor.tasks) == MODEL_PACKAGE_TASKS
    assert set(descriptor.task("detection").metadata) == {
        "preprocessing",
        "preprocessing_version",
    }


@pytest.mark.parametrize("task", MODEL_PACKAGE_TASKS)
def test_task_sha256_is_optional_and_verified_when_declared(
    manifest_package_factory,
    task,
):
    package, manifest = manifest_package_factory()
    model_path = package / manifest["tasks"][task]["file"]
    expected = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest["tasks"][task]["sha256"] = expected
    _write_manifest(package, manifest)

    descriptor = load_model_package(package).task(task)

    assert descriptor.sha256 == expected
    assert descriptor.as_config()["sha256"] == expected
    assert verify_model_artifact(descriptor) == model_path


@pytest.mark.parametrize("task", MODEL_PACKAGE_TASKS)
@pytest.mark.parametrize(
    ("sha256", "error_type"),
    [
        (None, TypeError),
        (False, TypeError),
        ("", ValueError),
        ("a" * 63, ValueError),
        ("a" * 65, ValueError),
        ("A" * 64, ValueError),
        ("g" * 64, ValueError),
        (f"{'a' * 64} ", ValueError),
    ],
)
def test_task_sha256_requires_exact_lowercase_hex(
    manifest_package_factory,
    task,
    sha256,
    error_type,
):
    package, manifest = manifest_package_factory()
    manifest["tasks"][task]["sha256"] = sha256
    _write_manifest(package, manifest)

    with pytest.raises(error_type, match=rf"tasks\.{task}\.sha256"):
        load_model_package(package)


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("manifest_version", 1, ValueError),
        ("manifest_version", "2", ValueError),
        ("model_id", "../bad", ValueError),
        ("tasks", [], TypeError),
        ("license", "LICENSE.txt", ValueError),
    ],
)
def test_v2_validates_required_root_fields(
    manifest_package_factory,
    field,
    value,
    error_type,
):
    package, manifest = manifest_package_factory()
    manifest[field] = value
    _write_manifest(package, manifest)

    with pytest.raises(error_type):
        load_model_package(package)


@pytest.mark.parametrize("field", ["model_id", "tasks", "license"])
def test_v2_rejects_missing_required_root_fields(manifest_package_factory, field):
    package, manifest = manifest_package_factory()
    manifest.pop(field)
    _write_manifest(package, manifest)

    with pytest.raises(ValueError, match="missing required fields"):
        load_model_package(package)


@pytest.mark.parametrize("task", MODEL_PACKAGE_TASKS)
def test_known_v2_task_requires_file(manifest_package_factory, task):
    package, manifest = manifest_package_factory()
    manifest["tasks"][task].pop("file")
    _write_manifest(package, manifest)

    with pytest.raises(ValueError, match=rf"tasks\.{task}\.file is required"):
        load_model_package(package)


@pytest.mark.parametrize("task", MODEL_PACKAGE_TASKS)
@pytest.mark.parametrize(
    "preprocessing",
    ["embedded", {"mean": 12.5, "std": 64.0}],
)
def test_every_known_task_accepts_both_preprocessing_modes(
    manifest_package_factory,
    task,
    preprocessing,
):
    package, manifest = manifest_package_factory()
    manifest["tasks"][task]["preprocessing"] = preprocessing
    _write_manifest(package, manifest)

    descriptor = load_model_package(package).task(task)

    assert descriptor.metadata["preprocessing"] == preprocessing


@pytest.mark.parametrize("task", MODEL_PACKAGE_TASKS)
@pytest.mark.parametrize(
    ("preprocessing", "error_type"),
    [
        (None, TypeError),
        (False, TypeError),
        (1, TypeError),
        ([], TypeError),
        ("external", ValueError),
        ({}, ValueError),
        ({"mean": 0.0}, ValueError),
        ({"mean": 0.0, "std": 0.0}, ValueError),
        ({"mean": float("nan"), "std": 1.0}, ValueError),
    ],
)
def test_known_task_rejects_invalid_explicit_preprocessing(
    manifest_package_factory,
    task,
    preprocessing,
    error_type,
):
    package, manifest = manifest_package_factory()
    manifest["tasks"][task]["preprocessing"] = preprocessing
    _write_manifest(package, manifest)

    with pytest.raises(error_type, match=rf"tasks\.{task}\.preprocessing"):
        load_model_package(package)


@pytest.mark.parametrize(
    ("task", "field", "value", "error_type"),
    [
        ("detection", "preprocessing_version", "", ValueError),
        ("verification", "expansion", 0.0, ValueError),
        ("verification", "expansion", True, TypeError),
        ("recognition", "preprocessing_version", 1, TypeError),
        ("recognition", "input_size", [112], TypeError),
        ("recognition", "input_size", [112, 96], ValueError),
        ("recognition", "embedding_dimension", 0, ValueError),
    ],
)
def test_known_task_validates_recognized_optional_metadata(
    manifest_package_factory,
    task,
    field,
    value,
    error_type,
):
    package, manifest = manifest_package_factory()
    manifest["tasks"][task][field] = value
    _write_manifest(package, manifest)

    with pytest.raises(error_type):
        load_model_package(package)


def test_task_without_sha256_does_not_hash_or_compare_model_content(
    manifest_package_factory,
):
    package, _manifest = manifest_package_factory()
    descriptor = load_model_package(package)
    (package / "verifier.onnx").write_bytes(b"changed-after-parse")

    assert verify_model_artifact(descriptor.task("verification")) == (
        package / "verifier.onnx"
    )


def test_declared_sha256_detects_model_changed_after_parse(
    manifest_package_factory,
):
    package, manifest = manifest_package_factory()
    model_path = package / "verifier.onnx"
    manifest["tasks"]["verification"]["sha256"] = hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()
    _write_manifest(package, manifest)
    descriptor = load_model_package(package).task("verification")

    model_path.write_bytes(b"changed-after-parse")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_model_artifact(descriptor)


def test_selected_model_must_exist_when_resolved(manifest_package_factory):
    package, _manifest = manifest_package_factory()
    descriptor = load_model_package(package)
    (package / "verifier.onnx").unlink()

    with pytest.raises(FileNotFoundError, match="model file does not exist"):
        verify_model_artifact(descriptor.task("verification"))


def test_model_path_cannot_escape_package(manifest_package_factory):
    package, manifest = manifest_package_factory()
    manifest["tasks"]["detection"]["file"] = "../outside.onnx"
    _write_manifest(package, manifest)

    with pytest.raises(ValueError, match="escapes"):
        load_model_package(package)


def test_known_tasks_must_reference_distinct_files(manifest_package_factory):
    package, manifest = manifest_package_factory()
    manifest["tasks"]["verification"]["file"] = manifest["tasks"]["detection"]["file"]
    _write_manifest(package, manifest)

    with pytest.raises(ValueError, match="distinct files"):
        load_model_package(package)


def test_verification_rejects_package_root_replaced_by_symlink(
    manifest_package_factory,
    tmp_path,
):
    package, _manifest = manifest_package_factory()
    descriptor = load_model_package(package)
    original = tmp_path / "original-package"
    outside = tmp_path / "outside-package"
    package.rename(original)
    outside.mkdir()
    (outside / "detector.onnx").write_bytes(b"detector-bytes")
    try:
        package.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    with pytest.raises(ValueError, match="escapes"):
        verify_model_artifact(descriptor.task("detection"))


@pytest.mark.parametrize("version", [None, 1, 3])
def test_non_v2_manifest_is_not_discovered(manifest_package_factory, version):
    package, manifest = manifest_package_factory()
    if version is None:
        manifest.pop("manifest_version")
    else:
        manifest["manifest_version"] = version
    _write_manifest(package, manifest)

    assert has_model_package_manifest(package) is False
    assert model_package_manifest_version(package) == version
