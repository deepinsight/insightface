from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from insightface.model_zoo import model_license as model_license_module
from insightface.model_zoo.model_license import (
    STATUS_DEFAULT_NON_COMMERCIAL,
    STATUS_EXPIRED,
    STATUS_INVALID,
    STATUS_INVALID_MANIFEST,
    STATUS_NOT_ACTIVE,
    STATUS_VERIFIED_COMMERCIAL,
    STATUS_VERIFIED_NON_COMMERCIAL,
    ModelLicenseError,
    canonical_license_bytes,
    inspect_model_package_license,
    verify_model_license,
)


def _write_license(
    path: Path,
    private_key: Ed25519PrivateKey,
    **overrides: object,
) -> None:
    document: dict[str, object] = {
        "license_version": 1,
        "license_id": "test-license-1",
        "issuer": "InsightFace",
        "model_id": "test_model",
        "grant": "non-commercial",
        "valid_from": "2026-01-01T00:00:00Z",
        **overrides,
    }
    signature = private_key.sign(canonical_license_bytes(document))
    document["signature"] = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _inspection(
    package: Path,
    private_key: Ed25519PrivateKey,
    *,
    expected_model_id: str | None = None,
):
    return inspect_model_package_license(
        package,
        expected_model_id=expected_model_id,
        now=datetime(2026, 9, 4, tzinfo=timezone.utc),
        public_keys=(private_key.public_key(),),
    )


def test_strict_verifier_preserves_server_summary_contract(tmp_path: Path) -> None:
    private_key = Ed25519PrivateKey.generate()
    license_path = tmp_path / "MODEL.LICENSE"
    _write_license(license_path, private_key)

    result = verify_model_license(
        license_path,
        expected_model_id="test_model",
        now=datetime(2026, 9, 4, tzinfo=timezone.utc),
        public_keys=(private_key.public_key(),),
    )

    assert result.public_summary() == {
        "license_id": "test-license-1",
        "issuer": "InsightFace",
        "model_id": "test_model",
        "grant": "non-commercial",
        "customer": None,
        "reference": None,
        "valid_from": "2026-01-01T00:00:00Z",
        "valid_until": None,
        "signature_valid": True,
        "commercial_use_permitted": False,
    }


def test_manifestless_missing_license_defaults_to_non_commercial(
    tmp_path: Path,
) -> None:
    package = tmp_path / "buffalo_l"
    package.mkdir()
    private_key = Ed25519PrivateKey.generate()

    result = _inspection(package, private_key)

    assert result.status == STATUS_DEFAULT_NON_COMMERCIAL
    assert result.model_id == "buffalo_l"
    assert result.license_path == package / "MODEL.LICENSE"
    assert result.license is None
    assert result.defaulted is True
    assert result.error is None
    assert result.public_summary()["grant"] == "non-commercial"
    assert result.public_summary()["signature_valid"] is False
    assert result.public_summary()["commercial_use_permitted"] is False


def test_not_downloaded_selected_model_defaults_to_non_commercial(
    tmp_path: Path,
) -> None:
    package = tmp_path / "not-downloaded"
    private_key = Ed25519PrivateKey.generate()

    result = _inspection(
        package,
        private_key,
        expected_model_id="buffalo_l",
    )

    assert result.status == STATUS_DEFAULT_NON_COMMERCIAL
    assert result.model_id == "buffalo_l"
    assert result.license_path == package / "MODEL.LICENSE"


def test_manifestless_signed_license_is_verified(tmp_path: Path) -> None:
    package = tmp_path / "test_model"
    package.mkdir()
    private_key = Ed25519PrivateKey.generate()
    _write_license(package / "MODEL.LICENSE", private_key)

    result = _inspection(package, private_key)

    assert result.status == STATUS_VERIFIED_NON_COMMERCIAL
    assert result.verified is True
    assert result.signature_valid is True


@pytest.mark.parametrize("manifest_version", [1, 2])
def test_versioned_manifests_resolve_safe_nested_license(
    tmp_path: Path,
    manifest_version: int,
) -> None:
    package = tmp_path / f"version-{manifest_version}"
    package.mkdir()
    private_key = Ed25519PrivateKey.generate()
    license_path = package / "payload" / "MODEL.LICENSE"
    _write_license(license_path, private_key)
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": manifest_version,
                "model_id": "test_model",
                "license": "payload/MODEL.LICENSE",
                # License inspection intentionally ignores inference metadata.
                "tasks": "not inspected",
                "future_extension": {"allowed": True},
            }
        ),
        encoding="utf-8",
    )

    result = _inspection(package, private_key)

    assert result.status == STATUS_VERIFIED_NON_COMMERCIAL
    assert result.manifest_version == manifest_version
    assert result.license_path == license_path


def test_unversioned_server_manifest_resolves_colocated_license(
    tmp_path: Path,
) -> None:
    package = tmp_path / "server-legacy"
    payload = package / ".bundles" / "test_model-v1"
    payload.mkdir(parents=True)
    private_key = Ed25519PrivateKey.generate()
    _write_license(payload / "MODEL.LICENSE", private_key)
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "package": {"name": "test_model", "release": "v1"},
                "models": [
                    {"file": ".bundles/test_model-v1/det.onnx"},
                    {"file": ".bundles/test_model-v1/rec.onnx"},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _inspection(package, private_key)

    assert result.status == STATUS_VERIFIED_NON_COMMERCIAL
    assert result.manifest_path == package / "manifest.json"
    assert result.manifest_version is None
    assert result.license_path == payload / "MODEL.LICENSE"


def test_unversioned_server_manifest_ignores_conflicting_license_field(
    tmp_path: Path,
) -> None:
    package = tmp_path / "server-legacy"
    payload = package / ".bundles" / "test_model-v1"
    payload.mkdir(parents=True)
    private_key = Ed25519PrivateKey.generate()
    _write_license(payload / "MODEL.LICENSE", private_key)
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "package": {"name": "test_model", "release": "v1"},
                "license": "MODEL.LICENSE",
                "models": [
                    {"file": ".bundles/test_model-v1/det.onnx"},
                    {"file": ".bundles/test_model-v1/rec.onnx"},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _inspection(package, private_key)

    assert result.status == STATUS_VERIFIED_NON_COMMERCIAL
    assert result.license_path == payload / "MODEL.LICENSE"


def test_unversioned_server_manifest_uses_recognizer_identity_fallback(
    tmp_path: Path,
) -> None:
    package = tmp_path / "server-legacy"
    payload = package / ".bundles" / "test_model-v1"
    payload.mkdir(parents=True)
    private_key = Ed25519PrivateKey.generate()
    _write_license(payload / "MODEL.LICENSE", private_key)
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "ignored-root-extension",
                "package": {"name": 42, "release": "v1"},
                "models": [
                    {
                        "file": ".bundles/test_model-v1/det.onnx",
                        "task": "face_detection",
                        "model_id": "detector_component",
                    },
                    {
                        "file": ".bundles/test_model-v1/rec.onnx",
                        "task": "face_recognition",
                        "model_id": "test_model",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _inspection(package, private_key)

    assert result.status == STATUS_VERIFIED_NON_COMMERCIAL
    assert result.model_id == "test_model"


def test_missing_versioned_license_file_uses_explicit_default(tmp_path: Path) -> None:
    package = tmp_path / "v2"
    package.mkdir()
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 2,
                "model_id": "test_model",
                "license": "MODEL.LICENSE",
            }
        ),
        encoding="utf-8",
    )
    private_key = Ed25519PrivateKey.generate()

    result = _inspection(package, private_key)

    assert result.status == STATUS_DEFAULT_NON_COMMERCIAL
    assert result.manifest_version == 2
    assert result.grant == "non-commercial"


def test_license_inspection_does_not_add_a_stricter_manifest_contract(
    tmp_path: Path,
) -> None:
    package = tmp_path / "v2-with-future-metadata"
    package.mkdir()
    (package / "manifest.json").write_text(
        '{"manifest_version":2,"model_id":"test_model",'
        '"license":"MODEL.LICENSE","future_value":NaN,'
        '"future_value":{"accepted_by_loader":true}}',
        encoding="utf-8",
    )
    private_key = Ed25519PrivateKey.generate()

    result = _inspection(package, private_key)

    assert result.status == STATUS_DEFAULT_NON_COMMERCIAL


def test_manifest_or_license_non_file_never_defaults_to_non_commercial(
    tmp_path: Path,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    manifest_directory_package = tmp_path / "manifest-directory"
    (manifest_directory_package / "manifest.json").mkdir(parents=True)

    manifest_result = _inspection(
        manifest_directory_package,
        private_key,
        expected_model_id="test_model",
    )

    assert manifest_result.status == STATUS_INVALID_MANIFEST
    assert manifest_result.defaulted is False

    license_directory_package = tmp_path / "license-directory"
    license_directory_package.mkdir()
    (license_directory_package / "MODEL.LICENSE").mkdir()

    license_result = _inspection(
        license_directory_package,
        private_key,
        expected_model_id="test_model",
    )

    assert license_result.status == STATUS_INVALID
    assert license_result.defaulted is False


def test_manifestless_directory_name_does_not_limit_missing_license_fallback(
    tmp_path: Path,
) -> None:
    package = tmp_path / "Custom Model Directory"
    package.mkdir()
    private_key = Ed25519PrivateKey.generate()

    result = _inspection(package, private_key)

    assert result.status == STATUS_DEFAULT_NON_COMMERCIAL
    assert result.model_id == "Custom Model Directory"


def test_symlink_loops_return_invalid_inspection_instead_of_raising(
    tmp_path: Path,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    package = tmp_path / "test_model"
    package.mkdir()
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 2,
                "model_id": "test_model",
                "license": "MODEL.LICENSE",
            }
        ),
        encoding="utf-8",
    )
    try:
        (package / "MODEL.LICENSE").symlink_to("MODEL.LICENSE")
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    license_loop = _inspection(package, private_key)

    assert license_loop.status == STATUS_INVALID_MANIFEST
    assert license_loop.defaulted is False

    package_loop = tmp_path / "package-loop"
    package_loop.symlink_to(package_loop.name)

    directory_loop = _inspection(
        package_loop,
        private_key,
        expected_model_id="test_model",
    )

    assert directory_loop.status == STATUS_INVALID_MANIFEST
    assert directory_loop.defaulted is False


def test_unreadable_or_missing_trusted_key_resource_returns_invalid_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    package = tmp_path / "test_model"
    package.mkdir()
    _write_license(package / "MODEL.LICENSE", private_key)

    def missing_resource(_package: str):
        raise FileNotFoundError("packaged trusted key directory is missing")

    monkeypatch.setattr(model_license_module.resources, "files", missing_resource)

    result = inspect_model_package_license(
        package,
        now=datetime(2026, 9, 4, tzinfo=timezone.utc),
    )

    assert result.status == STATUS_INVALID
    assert result.defaulted is False
    assert "trusted InsightFace model-license keys" in result.message


def test_unexpandable_home_directory_returns_an_inspection() -> None:
    result = inspect_model_package_license(
        "~insightface-user-that-must-not-exist/models",
        expected_model_id="test_model",
    )

    assert result.status in {
        STATUS_DEFAULT_NON_COMMERCIAL,
        STATUS_INVALID_MANIFEST,
    }


@pytest.mark.parametrize(
    "manifest",
    [
        "not-json",
        json.dumps({"manifest_version": 3, "model_id": "test_model"}),
        json.dumps({"manifest_version": 2, "model_id": "test_model"}),
        json.dumps(
            {
                "manifest_version": 2,
                "model_id": "test_model",
                "license": "../MODEL.LICENSE",
            }
        ),
    ],
)
def test_invalid_manifest_never_falls_back_to_root_license(
    tmp_path: Path,
    manifest: str,
) -> None:
    package = tmp_path / "test_model"
    package.mkdir()
    private_key = Ed25519PrivateKey.generate()
    _write_license(package / "MODEL.LICENSE", private_key)
    (package / "manifest.json").write_text(manifest, encoding="utf-8")

    result = _inspection(package, private_key)

    assert result.status == STATUS_INVALID_MANIFEST
    assert result.license is None
    assert result.defaulted is False
    assert result.error
    assert result.public_summary()["grant"] is None


def test_existing_invalid_license_never_defaults(tmp_path: Path) -> None:
    package = tmp_path / "test_model"
    package.mkdir()
    (package / "MODEL.LICENSE").write_text("{}", encoding="utf-8")
    private_key = Ed25519PrivateKey.generate()

    result = _inspection(package, private_key)

    assert result.status == STATUS_INVALID
    assert result.defaulted is False
    assert result.grant is None
    assert result.error


def test_inspection_distinguishes_time_validity_and_commercial_grant(
    tmp_path: Path,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    package = tmp_path / "test_model"
    package.mkdir()
    license_path = package / "MODEL.LICENSE"

    _write_license(license_path, private_key, valid_from="2027-01-01T00:00:00Z")
    assert _inspection(package, private_key).status == STATUS_NOT_ACTIVE

    _write_license(
        license_path,
        private_key,
        valid_from="2025-01-01T00:00:00Z",
        valid_until="2026-01-01T00:00:00Z",
    )
    assert _inspection(package, private_key).status == STATUS_EXPIRED

    _write_license(
        license_path,
        private_key,
        grant="commercial",
        customer="Example Customer",
    )
    commercial = _inspection(package, private_key)
    assert commercial.status == STATUS_VERIFIED_COMMERCIAL
    assert commercial.commercial_use_permitted is True


def test_expected_model_id_checks_manifest_and_license_scope(tmp_path: Path) -> None:
    private_key = Ed25519PrivateKey.generate()
    package = tmp_path / "custom-directory-name"
    package.mkdir()
    _write_license(package / "MODEL.LICENSE", private_key)

    valid = _inspection(package, private_key, expected_model_id="test_model")
    assert valid.status == STATUS_VERIFIED_NON_COMMERCIAL

    wrong = _inspection(package, private_key, expected_model_id="another_model")
    assert wrong.status == STATUS_INVALID
    assert "not the active model" in wrong.message


def test_strict_verifier_rejects_tampering(tmp_path: Path) -> None:
    private_key = Ed25519PrivateKey.generate()
    license_path = tmp_path / "MODEL.LICENSE"
    _write_license(license_path, private_key)
    document = json.loads(license_path.read_text(encoding="utf-8"))
    document["grant"] = "commercial"
    document["customer"] = "Tampered Customer"
    license_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ModelLicenseError, match="signature verification failed"):
        verify_model_license(
            license_path,
            expected_model_id="test_model",
            now=datetime(2026, 9, 4, tzinfo=timezone.utc),
            public_keys=(private_key.public_key(),),
        )
