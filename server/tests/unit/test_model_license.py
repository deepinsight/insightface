from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from insightface.model_zoo import model_license as shared_model_license
from insightface_server.licensing import (
    ModelLicenseError,
    canonical_license_bytes,
    verify_model_license,
)
from insightface_server.licensing.model_license import _trusted_keys


def test_server_license_imports_are_shared_compatibility_aliases() -> None:
    assert ModelLicenseError is shared_model_license.ModelLicenseError
    assert canonical_license_bytes is shared_model_license.canonical_license_bytes
    assert verify_model_license is shared_model_license.verify_model_license
    assert _trusted_keys is shared_model_license._trusted_keys


def _write_license(
    path: Path,
    private_key: Ed25519PrivateKey,
    **overrides: object,
) -> None:
    document: dict[str, object] = {
        "license_version": 1,
        "license_id": "test-license-1",
        "issuer": "InsightFace",
        "model_id": "buffalo_l",
        "grant": "non-commercial",
        "valid_from": "2026-01-01T00:00:00Z",
        **overrides,
    }
    signature = private_key.sign(canonical_license_bytes(document))
    document["signature"] = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def test_verifies_signed_model_scoped_license(tmp_path: Path) -> None:
    private_key = Ed25519PrivateKey.generate()
    path = tmp_path / "MODEL.LICENSE"
    _write_license(path, private_key)

    result = verify_model_license(
        path,
        expected_model_id="buffalo_l",
        now=datetime(2026, 7, 22, tzinfo=UTC),
        public_keys=(private_key.public_key(),),
    )

    assert result.issuer == "InsightFace"
    assert result.model_id == "buffalo_l"
    assert result.grant == "non-commercial"
    assert result.commercial_use_permitted is False
    assert result.public_summary()["signature_valid"] is True


def test_rejects_tampering_wrong_model_and_expiration(tmp_path: Path) -> None:
    private_key = Ed25519PrivateKey.generate()
    path = tmp_path / "MODEL.LICENSE"
    _write_license(path, private_key)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["grant"] = "commercial"
    document["customer"] = "Tampered Customer"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ModelLicenseError, match="signature verification failed"):
        verify_model_license(
            path,
            expected_model_id="buffalo_l",
            now=datetime(2026, 7, 22, tzinfo=UTC),
            public_keys=(private_key.public_key(),),
        )

    _write_license(path, private_key, model_id="private_recognition")
    with pytest.raises(ModelLicenseError, match="not the active model"):
        verify_model_license(
            path,
            expected_model_id="buffalo_l",
            now=datetime(2026, 7, 22, tzinfo=UTC),
            public_keys=(private_key.public_key(),),
        )

    _write_license(path, private_key, issuer="Another Issuer")
    with pytest.raises(ModelLicenseError, match="issuer must be 'InsightFace'"):
        verify_model_license(
            path,
            expected_model_id="buffalo_l",
            now=datetime(2026, 7, 22, tzinfo=UTC),
            public_keys=(private_key.public_key(),),
        )

    _write_license(
        path,
        private_key,
        valid_from="2025-01-01T00:00:00Z",
        valid_until="2026-01-01T00:00:00Z",
    )
    with pytest.raises(ModelLicenseError, match="expired"):
        verify_model_license(
            path,
            expected_model_id="buffalo_l",
            now=datetime(2026, 7, 22, tzinfo=UTC),
            public_keys=(private_key.public_key(),),
        )


def test_commercial_license_requires_customer_and_preserves_optional_fields(
    tmp_path: Path,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    path = tmp_path / "MODEL.LICENSE"
    _write_license(path, private_key, grant="commercial")
    with pytest.raises(ModelLicenseError, match="identify the customer"):
        verify_model_license(
            path,
            expected_model_id="buffalo_l",
            now=datetime(2026, 7, 22, tzinfo=UTC),
            public_keys=(private_key.public_key(),),
        )

    _write_license(
        path,
        private_key,
        grant="commercial",
        customer="Example Customer",
        reference="CONTRACT-2026-001",
    )
    result = verify_model_license(
        path,
        expected_model_id="buffalo_l",
        now=datetime(2026, 7, 22, tzinfo=UTC),
        public_keys=(private_key.public_key(),),
    )
    assert result.commercial_use_permitted is True
    assert result.customer == "Example Customer"
    assert result.reference == "CONTRACT-2026-001"


def test_bundled_public_license_verifies_with_active_public_key() -> None:
    license_path = (
        Path(__file__).resolve().parents[2]
        / "backend"
        / "insightface_server"
        / "licensing"
        / "defaults"
        / "buffalo_l"
        / "MODEL.LICENSE"
    )
    result = verify_model_license(license_path, expected_model_id="buffalo_l")
    assert result.issuer == "InsightFace"
    assert result.grant == "non-commercial"


def test_trusted_keys_ignore_macos_appledouble_files(tmp_path: Path) -> None:
    bundled_key = (
        Path(__file__).resolve().parents[3]
        / "python-package"
        / "insightface"
        / "model_zoo"
        / "trusted_keys"
        / "insightface-model-license-public-ed25519.pem"
    )
    (tmp_path / bundled_key.name).write_bytes(bundled_key.read_bytes())
    (tmp_path / f"._{bundled_key.name}").write_bytes(b"AppleDouble metadata")

    assert len(_trusted_keys(tmp_path)) == 1
