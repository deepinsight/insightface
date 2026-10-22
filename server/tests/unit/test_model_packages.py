from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from insightface_server.licensing import verify_model_license
from insightface_server.models import load_manifest
from insightface_server.models.packages import (
    DEFAULT_LICENSES_DIR,
    MODEL_LICENSE_FILENAME,
    MODEL_ZOO_RELEASE_BASE_URL,
    PACKAGES,
    ModelPackage,
    ModelPackageError,
    PackageFile,
    _manifest,
    _verify_model_license,
    extract_required_models,
    install_package,
    license_notice,
    model_license_document,
    verify_installed,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_catalog_contains_all_supported_packages_with_signed_licenses() -> None:
    expected = {
        "buffalo_l": (
            "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
            ("det_10g.onnx", "w600k_r50.onnx"),
            datetime(2021, 9, 22, tzinfo=UTC),
        ),
        "buffalo_m": (
            "d98264bd8f2dc75cbc2ddce2a14e636e02bb857b3051c234b737bf3b614edca9",
            ("det_2.5g.onnx", "w600k_r50.onnx"),
            datetime(2021, 9, 22, tzinfo=UTC),
        ),
        "buffalo_s": (
            "d85a87f503f691807cd8bb97128bdf7a0660326cd9cd02657127fa978bab8b5e",
            ("det_500m.onnx", "w600k_mbf.onnx"),
            datetime(2021, 9, 22, tzinfo=UTC),
        ),
        "buffalo_sc": (
            "57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72",
            ("det_500m.onnx", "w600k_mbf.onnx"),
            datetime(2021, 9, 22, tzinfo=UTC),
        ),
        "antelopev2": (
            "8e182f14fc6e80b3bfa375b33eb6cff7ee05d8ef7633e738d1c89021dcf0c5c5",
            ("scrfd_10g_bnkps.onnx", "glintr100.onnx"),
            datetime(2021, 9, 22, tzinfo=UTC),
        ),
        "raccoon_s": (
            "f67a624ef8a4495899eb4359a8a6953f7b4c62a8399c5bc745c0e0f6582f898d",
            ("det_10g_wo.onnx", "w600k_mbf.onnx"),
            datetime(2026, 8, 29, tzinfo=UTC),
        ),
        "raccoon_l": (
            "70cd4f2f1de0a89dd0983bdac55a066a6178543f86e9ec87154f6f259bdded7e",
            ("det_10g_wo.onnx", "w600k_r50.onnx"),
            datetime(2026, 8, 29, tzinfo=UTC),
        ),
    }

    assert set(PACKAGES) == set(expected)
    for name, package in PACKAGES.items():
        archive_sha256, filenames, valid_from = expected[name]
        assert package.url == f"{MODEL_ZOO_RELEASE_BASE_URL}{name}.zip"
        assert package.archive_sha256 == archive_sha256
        assert tuple(item.filename for item in package.files) == filenames
        assert "model_version" not in _manifest(package, Path("."))
        assert tuple(item.task for item in package.files) == (
            "face_detection",
            "face_recognition",
        )
        assert package.files[1].embedding_dimension == 512
        license_info = verify_model_license(
            DEFAULT_LICENSES_DIR / name / MODEL_LICENSE_FILENAME,
            expected_model_id=name,
        )
        assert license_info.issuer == "InsightFace"
        assert license_info.grant == "non-commercial"
        assert license_info.valid_from == valid_from
        assert license_info.valid_until is None


def _package(tmp_path: Path) -> tuple[ModelPackage, Path]:
    detector = b"synthetic-detector"
    recognizer = b"synthetic-recognizer"
    archive_path = tmp_path / "test.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/det.onnx", detector)
        archive.writestr("nested/rec.onnx", recognizer)
        archive.writestr("ignored.onnx", b"not installed")
        archive.writestr("verifier.onnx", b"server does not install this task")
        archive.writestr("manifest.json", '{"manifest_version": 2}')
        archive.writestr("MODEL.LICENSE", "untrusted archive license")
    package = ModelPackage(
        name="buffalo_l",
        display_name="Synthetic Buffalo_L",
        url="https://example.invalid/test.zip",
        archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        files=(
            PackageFile(
                filename="det.onnx",
                sha256=_sha256(detector),
                model_id="test-detector",
                task="face_detection",
                input_size=(640, 640),
                preprocessing_version="test-detector-1",
                input_mean=127.5,
                input_std=128.0,
            ),
            PackageFile(
                filename="rec.onnx",
                sha256=_sha256(recognizer),
                model_id="test-recognizer",
                task="face_recognition",
                input_size=(112, 112),
                preprocessing_version="test-recognizer-1",
                input_mean=127.5,
                input_std=127.5,
                embedding_dimension=512,
            ),
        ),
    )
    return package, archive_path


def test_install_is_verified_atomic_and_idempotent(tmp_path: Path) -> None:
    package, archive = _package(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    downloads = 0

    def downloader(_package: ModelPackage, destination: Path) -> None:
        nonlocal downloads
        downloads += 1
        shutil.copyfile(archive, destination)

    assert install_package(package, models_dir, downloader=downloader) == "installed"
    bundle = load_manifest(models_dir)
    assert bundle.detector.sha256 == package.files[0].sha256
    assert bundle.recognizer.sha256 == package.files[1].sha256
    assert not (models_dir / "ignored.onnx").exists()
    assert not (models_dir / "verifier.onnx").exists()

    root_manifest_path = models_dir / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    relative_license = Path(root_manifest["license"])
    license_path = models_dir / relative_license
    assert license_path.read_text(encoding="utf-8") == model_license_document(package)
    assert root_manifest["manifest_version"] == 1
    assert root_manifest["model_id"] == "buffalo_l"
    assert _verify_model_license(package, models_dir).issuer == "InsightFace"

    # Re-running install upgrades a recognized legacy bundle without downloading again.
    license_path.unlink()
    bundle_dir = relative_license.parent
    legacy_manifest = {
        "package": {"name": package.name, "release": "legacy-release"},
        "models": [
            {
                "model_id": item.model_id,
                "model_version": "legacy-version",
                "task": item.task,
                "file": (bundle_dir / item.filename).as_posix(),
                "input_size": list(item.input_size),
                "preprocessing_version": item.preprocessing_version,
                "sha256": item.sha256,
                **(
                    {"embedding_dimension": item.embedding_dimension}
                    if item.embedding_dimension is not None
                    else {}
                ),
            }
            for item in package.files
        ],
    }
    root_manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")
    assert install_package(package, models_dir, downloader=downloader) == "already_installed"
    assert downloads == 1
    assert license_path.read_text(encoding="utf-8") == model_license_document(package)
    _verify_model_license(package, models_dir)

    installed, summaries, license_info = verify_installed(models_dir)
    assert installed == "buffalo_l"
    assert license_info.issuer == "InsightFace"
    assert [summary["task"] for summary in summaries] == [
        "face_detection",
        "face_recognition",
    ]


def test_extract_rejects_duplicate_required_filename(tmp_path: Path) -> None:
    package, _archive = _package(tmp_path)
    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("a/det.onnx", b"synthetic-detector")
        archive.writestr("b/det.onnx", b"synthetic-detector")
        archive.writestr("rec.onnx", b"synthetic-recognizer")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ModelPackageError, match="exactly one det.onnx"):
        extract_required_models(package, duplicate, output)


def test_extract_rejects_model_digest_mismatch(tmp_path: Path) -> None:
    package, archive = _package(tmp_path)
    bad_file = PackageFile(
        filename=package.files[0].filename,
        sha256="0" * 64,
        model_id=package.files[0].model_id,
        task="face_detection",
        input_size=(640, 640),
        preprocessing_version="test",
        input_mean=127.5,
        input_std=128.0,
    )
    bad_package = ModelPackage(
        name="bad",
        display_name="Bad",
        url=package.url,
        archive_sha256=package.archive_sha256,
        files=(bad_file, package.files[1]),
    )
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ModelPackageError, match="SHA-256 mismatch"):
        extract_required_models(bad_package, archive, output)


def test_license_notice_is_explicit(tmp_path: Path) -> None:
    package, _archive = _package(tmp_path)
    notice = license_notice(package)
    assert "non-commercial research use only" in notice
    assert "Commercial use requires a separate license" in notice
    assert "https://www.insightface.ai" in notice


def test_verify_rejects_missing_model_license(tmp_path: Path) -> None:
    package, archive = _package(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    def downloader(_package: ModelPackage, destination: Path) -> None:
        shutil.copyfile(archive, destination)

    install_package(package, models_dir, downloader=downloader)
    manifest = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))
    license_path = models_dir / manifest["license"]
    assert license_path.name == MODEL_LICENSE_FILENAME
    license_path.unlink()
    with pytest.raises(RuntimeError, match="license file is missing"):
        _verify_model_license(package, models_dir)
