"""Small, offline Ed25519 model-license verifier.

The license is a compliance credential for a logical ``model_id``.  It is
deliberately not bound to an ONNX digest so an operator may create an FP16,
INT8, optimized ONNX, or TensorRT derivative without asking for a new license.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

LICENSE_FILENAME = "MODEL.LICENSE"
LICENSE_VERSION = 1
LICENSE_ISSUER = "InsightFace"
GRANTS = frozenset({"non-commercial", "commercial"})
MAX_LICENSE_BYTES = 64 * 1024
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SIGNATURE = re.compile(r"^[A-Za-z0-9_-]+$")
_REQUIRED_FIELDS = frozenset(
    {
        "license_version",
        "license_id",
        "issuer",
        "model_id",
        "grant",
        "valid_from",
        "signature",
    }
)
_OPTIONAL_FIELDS = frozenset({"customer", "reference", "valid_until"})


class ModelLicenseError(RuntimeError):
    """A model license is missing, invalid, not active, or expired."""


@dataclass(frozen=True, slots=True)
class ModelLicense:
    license_id: str
    issuer: str
    model_id: str
    grant: str
    valid_from: datetime
    valid_until: datetime | None
    customer: str | None = None
    reference: str | None = None

    @property
    def commercial_use_permitted(self) -> bool:
        return self.grant == "commercial"

    def public_summary(self) -> dict[str, object]:
        return {
            "license_id": self.license_id,
            "issuer": self.issuer,
            "model_id": self.model_id,
            "grant": self.grant,
            "customer": self.customer,
            "reference": self.reference,
            "valid_from": _format_time(self.valid_from),
            "valid_until": (
                _format_time(self.valid_until) if self.valid_until is not None else None
            ),
            "signature_valid": True,
            "commercial_use_permitted": self.commercial_use_permitted,
        }


def _format_time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ModelLicenseError(f"Duplicate field in model license: {key}")
        result[key] = value
    return result


def _read_document(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_LICENSE_BYTES:
            raise ModelLicenseError(
                f"Model license must be between 1 and {MAX_LICENSE_BYTES} bytes: {path}"
            )
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-standard JSON number: {value}")
            ),
        )
    except ModelLicenseError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ModelLicenseError(f"Unable to read model license {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ModelLicenseError("Model license root must be a JSON object")
    return raw


def _required_string(document: dict[str, Any], field: str, *, maximum: int = 256) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ModelLicenseError(
            f"Model license field {field!r} must be a non-empty string of at most "
            f"{maximum} characters"
        )
    return value.strip()


def _optional_string(document: dict[str, Any], field: str) -> str | None:
    if field not in document:
        return None
    return _required_string(document, field)


def _parse_time(document: dict[str, Any], field: str, *, required: bool) -> datetime | None:
    if field not in document:
        if required:
            raise ModelLicenseError(f"Model license is missing required field: {field}")
        return None
    value = _required_string(document, field, maximum=32)
    if not value.endswith("Z"):
        raise ModelLicenseError(f"Model license field {field!r} must use UTC and end in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ModelLicenseError(f"Invalid UTC timestamp in {field!r}: {value}") from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ModelLicenseError(f"Model license field {field!r} must use UTC")
    return parsed


def _signature_bytes(value: str) -> bytes:
    if not _SIGNATURE.fullmatch(value):
        raise ModelLicenseError("Model license signature must use unpadded base64url")
    try:
        padding = "=" * (-len(value) % 4)
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ModelLicenseError("Invalid model license signature encoding") from exc
    if len(decoded) != 64:
        raise ModelLicenseError("An Ed25519 model license signature must be 64 bytes")
    return decoded


def canonical_license_bytes(document: dict[str, Any]) -> bytes:
    """Canonicalize all signed fields, excluding only ``signature``."""

    unsigned = {key: value for key, value in document.items() if key != "signature"}
    try:
        return rfc8785.dumps(unsigned)
    except rfc8785.CanonicalizationError as exc:
        raise ModelLicenseError(f"Model license cannot be canonicalized: {exc}") from exc


def _trusted_keys(directory: Path | None = None) -> tuple[Ed25519PublicKey, ...]:
    key_dir = directory or Path(__file__).with_name("trusted_keys")
    paths = sorted(
        path for path in key_dir.glob("*.pem") if not path.name.startswith(".")
    )
    if not paths:
        raise ModelLicenseError(f"No trusted InsightFace model-license keys found in {key_dir}")
    keys: list[Ed25519PublicKey] = []
    for path in paths:
        try:
            loaded = serialization.load_pem_public_key(path.read_bytes())
        except (OSError, ValueError, TypeError) as exc:
            raise ModelLicenseError(f"Invalid trusted public key {path.name}: {exc}") from exc
        if not isinstance(loaded, Ed25519PublicKey):
            raise ModelLicenseError(f"Trusted key {path.name} is not an Ed25519 public key")
        keys.append(loaded)
    return tuple(keys)


def verify_model_license(
    path: str | Path,
    *,
    expected_model_id: str,
    now: datetime | None = None,
    public_keys: Iterable[Ed25519PublicKey] | None = None,
) -> ModelLicense:
    """Verify signature, model scope, issuer, grant, and validity period."""

    license_path = Path(path)
    if not license_path.is_file():
        raise ModelLicenseError(f"Required model license file is missing: {license_path}")
    document = _read_document(license_path)
    fields = frozenset(document)
    missing = sorted(_REQUIRED_FIELDS - fields)
    unknown = sorted(fields - _REQUIRED_FIELDS - _OPTIONAL_FIELDS)
    if missing:
        raise ModelLicenseError(f"Model license is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ModelLicenseError(f"Model license contains unsupported fields: {', '.join(unknown)}")
    if document.get("license_version") != LICENSE_VERSION:
        raise ModelLicenseError(
            f"Unsupported model license version: {document.get('license_version')!r}"
        )

    license_id = _required_string(document, "license_id")
    issuer = _required_string(document, "issuer")
    if issuer != LICENSE_ISSUER:
        raise ModelLicenseError(
            f"Model license issuer must be {LICENSE_ISSUER!r}; found {issuer!r}"
        )
    model_id = _required_string(document, "model_id", maximum=128)
    if not _MODEL_ID.fullmatch(model_id):
        raise ModelLicenseError("Model license model_id has an invalid format")
    if model_id != expected_model_id:
        raise ModelLicenseError(
            f"Model license is for {model_id!r}, not the active model {expected_model_id!r}"
        )
    grant = _required_string(document, "grant", maximum=32)
    if grant not in GRANTS:
        raise ModelLicenseError(
            f"Model license grant must be one of: {', '.join(sorted(GRANTS))}"
        )
    customer = _optional_string(document, "customer")
    reference = _optional_string(document, "reference")
    if grant == "commercial" and customer is None:
        raise ModelLicenseError("A commercial model license must identify the customer")

    valid_from = _parse_time(document, "valid_from", required=True)
    valid_until = _parse_time(document, "valid_until", required=False)
    assert valid_from is not None
    if valid_until is not None and valid_until <= valid_from:
        raise ModelLicenseError("Model license valid_until must be later than valid_from")

    signature = _signature_bytes(_required_string(document, "signature", maximum=128))
    signed = canonical_license_bytes(document)
    keys = tuple(public_keys) if public_keys is not None else _trusted_keys()
    if not keys:
        raise ModelLicenseError("No trusted InsightFace model-license keys are configured")
    for key in keys:
        try:
            key.verify(signature, signed)
            break
        except InvalidSignature:
            continue
    else:
        raise ModelLicenseError("Model license signature verification failed")

    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        raise ModelLicenseError("License verification time must be timezone-aware")
    current = current.astimezone(UTC)
    if current < valid_from:
        raise ModelLicenseError(
            f"Model license is not active until {_format_time(valid_from)}"
        )
    if valid_until is not None and current >= valid_until:
        raise ModelLicenseError(f"Model license expired at {_format_time(valid_until)}")

    return ModelLicense(
        license_id=license_id,
        issuer=issuer,
        model_id=model_id,
        grant=grant,
        valid_from=valid_from,
        valid_until=valid_until,
        customer=customer,
        reference=reference,
    )
