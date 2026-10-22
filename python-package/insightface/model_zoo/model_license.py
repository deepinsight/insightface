"""Shared inspection and verification for InsightFace model licenses.

The strict :func:`verify_model_license` entry point verifies a concrete signed
``MODEL.LICENSE`` file.  :func:`inspect_model_package_license` is the
directory-level UI/API entry point: it locates the license in legacy, Server
V1, and model-package V2 layouts and returns a non-throwing result.

An absent license intentionally means the package falls back to the public
non-commercial terms.  A license that exists but is malformed, untrusted,
expired, or scoped to another model never falls back.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any


LICENSE_FILENAME = "MODEL.LICENSE"
LICENSE_VERSION = 1
LICENSE_ISSUER = "InsightFace"
GRANTS = frozenset({"non-commercial", "commercial"})
MAX_LICENSE_BYTES = 64 * 1024

STATUS_VERIFIED_NON_COMMERCIAL = "verified_non_commercial"
STATUS_VERIFIED_COMMERCIAL = "verified_commercial"
STATUS_DEFAULT_NON_COMMERCIAL = "default_non_commercial"
STATUS_INVALID = "invalid"
STATUS_INVALID_MANIFEST = "invalid_manifest"
STATUS_NOT_ACTIVE = "not_active"
STATUS_EXPIRED = "expired"
STATUS_DEPENDENCY_MISSING = "dependency_missing"

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
_UTC = timezone.utc


class ModelLicenseError(RuntimeError):
    """A model license is missing, invalid, not active, or expired."""


class ModelLicenseDependencyError(ModelLicenseError):
    """The optional dependencies needed for signed-license verification are absent."""


class ModelLicenseNotActiveError(ModelLicenseError):
    """The signed license has not reached its activation time."""


class ModelLicenseExpiredError(ModelLicenseError):
    """The signed license is past its expiration time."""


class ModelLicenseManifestError(ModelLicenseError):
    """The package manifest cannot safely identify a model license."""


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
        """Return the stable Server-compatible signed-license summary."""

        return {
            "license_id": self.license_id,
            "issuer": self.issuer,
            "model_id": self.model_id,
            "grant": self.grant,
            "customer": self.customer,
            "reference": self.reference,
            "valid_from": _format_time(self.valid_from),
            "valid_until": (
                _format_time(self.valid_until)
                if self.valid_until is not None
                else None
            ),
            "signature_valid": True,
            "commercial_use_permitted": self.commercial_use_permitted,
        }


@dataclass(frozen=True, slots=True)
class ModelLicenseInspection:
    """Non-throwing result for one model-package directory.

    ``license`` is populated only for a verified signed credential.  The
    ``default_non_commercial`` state is an explicit fallback policy, not a
    claim that a signature was present.
    """

    package_path: Path
    model_id: str
    license_path: Path | None
    status: str
    message: str
    license: ModelLicense | None = None
    manifest_path: Path | None = None
    manifest_version: int | None = None

    @property
    def grant(self) -> str | None:
        if self.license is not None:
            return self.license.grant
        if self.status == STATUS_DEFAULT_NON_COMMERCIAL:
            return "non-commercial"
        return None

    @property
    def signature_valid(self) -> bool:
        return self.license is not None

    @property
    def commercial_use_permitted(self) -> bool:
        return bool(
            self.license is not None
            and self.license.commercial_use_permitted
        )

    @property
    def defaulted(self) -> bool:
        return self.status == STATUS_DEFAULT_NON_COMMERCIAL

    @property
    def verified(self) -> bool:
        return self.status in {
            STATUS_VERIFIED_NON_COMMERCIAL,
            STATUS_VERIFIED_COMMERCIAL,
        }

    @property
    def error(self) -> str | None:
        if self.verified or self.defaulted:
            return None
        return self.message

    def public_summary(self) -> dict[str, object]:
        """Return a presentation-friendly summary with a stable common shape."""

        if self.license is not None:
            summary = self.license.public_summary()
        else:
            summary = {
                "license_id": None,
                "issuer": None,
                "model_id": self.model_id,
                "grant": self.grant,
                "customer": None,
                "reference": None,
                "valid_from": None,
                "valid_until": None,
                "signature_valid": False,
                "commercial_use_permitted": False,
            }
        summary.update(
            {
                "status": self.status,
                "defaulted": self.defaulted,
                "message": self.message,
            }
        )
        return summary


def _format_time(value: datetime) -> str:
    return value.astimezone(_UTC).isoformat().replace("+00:00", "Z")


def _dependencies() -> tuple[Any, Any, Any, type[Any]]:
    try:
        import rfc8785
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise ModelLicenseDependencyError(
            "Signed model-license verification requires the 'cryptography' "
            "and 'rfc8785' packages; install insightface[gui]"
        ) from exc
    return rfc8785, InvalidSignature, serialization, Ed25519PublicKey


def _reject_license_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ModelLicenseError(f"Duplicate field in model license: {key}")
        result[key] = value
    return result


def _read_license_document(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_LICENSE_BYTES:
            raise ModelLicenseError(
                f"Model license must be between 1 and {MAX_LICENSE_BYTES} bytes: {path}"
            )
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_license_duplicate_keys,
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


def _read_manifest_document(path: Path) -> dict[str, Any]:
    """Read only the fields needed to locate a license.

    Deliberately mirror the permissive JSON behavior of the existing model
    loaders here. License presentation must not impose a second, stricter
    manifest contract on packages that those loaders already accept.
    """

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelLicenseManifestError(
            f"Unable to read model package manifest {path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ModelLicenseManifestError("Model package manifest root must be an object")
    return raw


def _required_string(
    document: Mapping[str, Any], field: str, *, maximum: int = 256
) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ModelLicenseError(
            f"Model license field {field!r} must be a non-empty string of at most "
            f"{maximum} characters"
        )
    return value.strip()


def _optional_string(document: Mapping[str, Any], field: str) -> str | None:
    if field not in document:
        return None
    return _required_string(document, field)


def _parse_time(
    document: Mapping[str, Any], field: str, *, required: bool
) -> datetime | None:
    if field not in document:
        if required:
            raise ModelLicenseError(f"Model license is missing required field: {field}")
        return None
    value = _required_string(document, field, maximum=32)
    if not value.endswith("Z"):
        raise ModelLicenseError(
            f"Model license field {field!r} must use UTC and end in Z"
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ModelLicenseError(f"Invalid UTC timestamp in {field!r}: {value}") from exc
    if parsed.utcoffset() != _UTC.utcoffset(parsed):
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


def canonical_license_bytes(document: Mapping[str, Any]) -> bytes:
    """Canonicalize all signed fields, excluding only ``signature``."""

    rfc8785, _invalid_signature, _serialization, _key_type = _dependencies()
    unsigned = {key: value for key, value in document.items() if key != "signature"}
    try:
        return rfc8785.dumps(unsigned)
    except rfc8785.CanonicalizationError as exc:
        raise ModelLicenseError(f"Model license cannot be canonicalized: {exc}") from exc


def _trusted_keys(directory: Path | None = None) -> tuple[Any, ...]:
    """Load bundled keys, retaining ``directory`` for Server test compatibility."""

    _rfc8785, _invalid_signature, serialization, Ed25519PublicKey = _dependencies()
    if directory is not None:
        location = str(directory)
        try:
            key_files: Iterable[Any] = sorted(
                path
                for path in Path(directory).glob("*.pem")
                if not path.name.startswith(".")
            )
        except OSError as exc:
            raise ModelLicenseError(
                f"Unable to read trusted InsightFace model-license keys from "
                f"{location}: {exc}"
            ) from exc
    else:
        location = f"{__package__}.trusted_keys"
        try:
            key_root = resources.files(__package__).joinpath("trusted_keys")
            key_files = sorted(
                (
                    entry
                    for entry in key_root.iterdir()
                    if entry.name.endswith(".pem")
                    and not entry.name.startswith(".")
                ),
                key=lambda entry: entry.name,
            )
        except (OSError, TypeError) as exc:
            raise ModelLicenseError(
                f"Unable to read trusted InsightFace model-license keys from "
                f"{location}: {exc}"
            ) from exc
    keys: list[Any] = []
    for key_file in key_files:
        try:
            loaded = serialization.load_pem_public_key(key_file.read_bytes())
        except (OSError, ValueError, TypeError) as exc:
            raise ModelLicenseError(
                f"Invalid trusted public key {key_file.name}: {exc}"
            ) from exc
        if not isinstance(loaded, Ed25519PublicKey):
            raise ModelLicenseError(
                f"Trusted key {key_file.name} is not an Ed25519 public key"
            )
        keys.append(loaded)
    if not keys:
        raise ModelLicenseError(
            f"No trusted InsightFace model-license keys found in {location}"
        )
    return tuple(keys)


def verify_model_license(
    path: str | Path,
    *,
    expected_model_id: str,
    now: datetime | None = None,
    public_keys: Iterable[Any] | None = None,
) -> ModelLicense:
    """Verify signature, model scope, issuer, grant, and validity period."""

    _rfc8785, InvalidSignature, _serialization, _key_type = _dependencies()
    license_path = Path(path)
    if not license_path.is_file():
        raise ModelLicenseError(f"Required model license file is missing: {license_path}")
    document = _read_license_document(license_path)
    fields = frozenset(document)
    missing = sorted(_REQUIRED_FIELDS - fields)
    unknown = sorted(fields - _REQUIRED_FIELDS - _OPTIONAL_FIELDS)
    if missing:
        raise ModelLicenseError(
            f"Model license is missing required fields: {', '.join(missing)}"
        )
    if unknown:
        raise ModelLicenseError(
            f"Model license contains unsupported fields: {', '.join(unknown)}"
        )
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
            f"Model license is for {model_id!r}, not the active model "
            f"{expected_model_id!r}"
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

    current = now or datetime.now(_UTC)
    if current.tzinfo is None:
        raise ModelLicenseError("License verification time must be timezone-aware")
    current = current.astimezone(_UTC)
    if current < valid_from:
        raise ModelLicenseNotActiveError(
            f"Model license is not active until {_format_time(valid_from)}"
        )
    if valid_until is not None and current >= valid_until:
        raise ModelLicenseExpiredError(
            f"Model license expired at {_format_time(valid_until)}"
        )

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


def _manifest_model_id(
    manifest: Mapping[str, Any], package_path: Path, expected_model_id: str | None
) -> str:
    if "manifest_version" not in manifest:
        value: Any = None
        package = manifest.get("package")
        if isinstance(package, Mapping):
            package_name = package.get("name")
            if isinstance(package_name, str) and package_name.strip():
                value = package_name
        if value is None:
            models = manifest.get("models")
            if isinstance(models, list):
                recognizer = next(
                    (
                        item
                        for item in models
                        if isinstance(item, Mapping)
                        and item.get("task") == "face_recognition"
                    ),
                    None,
                )
                if recognizer is not None:
                    value = recognizer.get("model_id")
        if not isinstance(value, str) or not value.strip():
            value = expected_model_id or package_path.name or "unknown"
        result = value.strip()
    else:
        value = manifest.get("model_id")
        if not isinstance(value, str) or not _MODEL_ID.fullmatch(value.strip()):
            raise ModelLicenseManifestError(
                "Model package manifest model_id is invalid"
            )
        result = value.strip()
    if expected_model_id is not None and result != expected_model_id:
        raise ModelLicenseManifestError(
            f"Model package manifest is for {result!r}, not the selected model "
            f"{expected_model_id!r}"
        )
    return result


def _safe_manifest_license_path(package_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ModelLicenseManifestError(
            "Model package manifest license must be a non-empty string"
        )
    filename = value.strip()
    if "\\" in filename or "\x00" in filename:
        raise ModelLicenseManifestError(
            "Model package manifest license must be a safe relative POSIX path"
        )
    relative = PurePosixPath(filename)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or any(":" in part for part in relative.parts)
        or relative.name != LICENSE_FILENAME
    ):
        raise ModelLicenseManifestError(
            f"Model package manifest license must safely reference {LICENSE_FILENAME}"
        )
    candidate = package_path / Path(*relative.parts)
    try:
        candidate_exists = not _path_is_missing(
            candidate,
            label="model license path",
            error_type=ModelLicenseManifestError,
        )
        result = candidate.resolve(strict=candidate_exists)
    except (OSError, RuntimeError) as exc:
        raise ModelLicenseManifestError(
            "Model package manifest license path cannot be resolved"
        ) from exc
    try:
        result.relative_to(package_path)
    except ValueError as exc:
        raise ModelLicenseManifestError(
            "Model package manifest license escapes the package directory"
        ) from exc
    return result


def _path_is_missing(
    path: Path,
    *,
    label: str,
    error_type: type[ModelLicenseError] = ModelLicenseError,
) -> bool:
    """Return true only for an actually absent directory entry.

    ``Path.exists()`` intentionally suppresses permission errors and treats
    broken symlinks as absent. Neither case may activate the non-commercial
    missing-file fallback.
    """

    try:
        path.lstat()
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise error_type(f"Unable to inspect {label} {path}: {exc}") from exc
    return False


def _legacy_license_path(manifest: Mapping[str, Any], package_path: Path) -> Path:
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ModelLicenseManifestError(
            "Unversioned Server manifest must contain a non-empty models list"
        )
    parents: set[Path] = set()
    for index, model in enumerate(models):
        if not isinstance(model, Mapping):
            raise ModelLicenseManifestError(
                f"Unversioned Server manifest models[{index}] must be an object"
            )
        filename = model.get("file")
        if not isinstance(filename, str) or not filename.strip():
            raise ModelLicenseManifestError(
                f"Unversioned Server manifest models[{index}].file is invalid"
            )
        relative = PurePosixPath(filename.strip())
        if (
            "\\" in filename
            or "\x00" in filename
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or any(":" in part for part in relative.parts)
        ):
            raise ModelLicenseManifestError(
                f"Unversioned Server manifest models[{index}].file is unsafe"
            )
        try:
            parent = (package_path / Path(*relative.parts)).resolve().parent
        except (OSError, RuntimeError) as exc:
            raise ModelLicenseManifestError(
                f"Unversioned Server manifest models[{index}].file cannot be resolved"
            ) from exc
        try:
            parent.relative_to(package_path)
        except ValueError as exc:
            raise ModelLicenseManifestError(
                f"Unversioned Server manifest models[{index}].file escapes the package"
            ) from exc
        parents.add(parent)
    if len(parents) != 1:
        raise ModelLicenseManifestError(
            "Unversioned Server manifest model files must share one package directory"
        )
    return parents.pop() / LICENSE_FILENAME


def _package_license_location(
    model_dir: str | Path, expected_model_id: str | None
) -> tuple[Path, Path | None, int | None, str, Path]:
    try:
        package_path = Path(model_dir).resolve()
    except (OSError, RuntimeError) as exc:
        raise ModelLicenseManifestError(
            f"Model package directory cannot be resolved: {model_dir}"
        ) from exc
    if not package_path.is_dir():
        if _path_is_missing(
            package_path,
            label="model package directory",
            error_type=ModelLicenseManifestError,
        ) and expected_model_id is not None:
            return (
                package_path,
                None,
                None,
                expected_model_id,
                package_path / LICENSE_FILENAME,
            )
        raise ModelLicenseManifestError(
            f"Model package directory does not exist: {package_path}"
        )
    manifest_path = package_path / "manifest.json"
    if _path_is_missing(
        manifest_path,
        label="model package manifest",
        error_type=ModelLicenseManifestError,
    ):
        model_id = expected_model_id or package_path.name or "unknown"
        return package_path, None, None, model_id, package_path / LICENSE_FILENAME
    if not manifest_path.is_file():
        raise ModelLicenseManifestError(
            f"Model package manifest is not a regular file: {manifest_path}"
        )

    manifest = _read_manifest_document(manifest_path)
    model_id = _manifest_model_id(manifest, package_path, expected_model_id)
    if "manifest_version" not in manifest:
        return (
            package_path,
            manifest_path,
            None,
            model_id,
            _legacy_license_path(manifest, package_path),
        )
    version = manifest.get("manifest_version")
    if type(version) is not int or version not in {1, 2}:
        raise ModelLicenseManifestError(
            f"Unsupported model package manifest_version: {version!r}"
        )
    if "license" not in manifest:
        raise ModelLicenseManifestError(
            f"Model package manifest V{version} is missing the license field"
        )
    return (
        package_path,
        manifest_path,
        version,
        model_id,
        _safe_manifest_license_path(package_path, manifest["license"]),
    )


def inspect_model_package_license(
    model_dir: str | Path,
    *,
    expected_model_id: str | None = None,
    now: datetime | None = None,
    public_keys: Iterable[Any] | None = None,
) -> ModelLicenseInspection:
    """Inspect a model-package directory without raising expected data errors.

    Supported layouts are manifest-less ModelZoo directories, unversioned
    legacy Server manifests, Server manifest V1, and model-package manifest
    V2.  The function deliberately does not import or initialize FaceAnalysis.
    """

    try:
        raw_package_path = Path(model_dir).expanduser()
    except (OSError, RuntimeError) as exc:
        raw_package_path = Path(str(model_dir))
        package_path = raw_package_path.absolute()
        return ModelLicenseInspection(
            package_path=package_path,
            model_id=expected_model_id or package_path.name or "unknown",
            license_path=None,
            status=STATUS_INVALID_MANIFEST,
            message=f"Model package directory cannot be expanded: {exc}",
            manifest_path=package_path / "manifest.json",
        )
    try:
        package_path = raw_package_path.resolve()
    except (OSError, RuntimeError):
        package_path = raw_package_path.absolute()
    fallback_model_id = expected_model_id or package_path.name or "unknown"
    try:
        (
            package_path,
            manifest_path,
            manifest_version,
            model_id,
            license_path,
        ) = _package_license_location(raw_package_path, expected_model_id)
    except (ModelLicenseManifestError, OSError, RuntimeError) as exc:
        return ModelLicenseInspection(
            package_path=package_path,
            model_id=fallback_model_id,
            license_path=None,
            status=STATUS_INVALID_MANIFEST,
            message=str(exc),
            manifest_path=(package_path / "manifest.json"),
        )

    try:
        license_missing = _path_is_missing(
            license_path,
            label="model license",
        )
    except ModelLicenseError as exc:
        return ModelLicenseInspection(
            package_path=package_path,
            manifest_path=manifest_path,
            manifest_version=manifest_version,
            model_id=model_id,
            license_path=license_path,
            status=STATUS_INVALID,
            message=str(exc),
        )
    if license_missing:
        return ModelLicenseInspection(
            package_path=package_path,
            manifest_path=manifest_path,
            manifest_version=manifest_version,
            model_id=model_id,
            license_path=license_path,
            status=STATUS_DEFAULT_NON_COMMERCIAL,
            message=(
                f"{LICENSE_FILENAME} is absent; defaulting to non-commercial use"
            ),
        )

    try:
        model_license = verify_model_license(
            license_path,
            expected_model_id=model_id,
            now=now,
            public_keys=public_keys,
        )
    except ModelLicenseNotActiveError as exc:
        status = STATUS_NOT_ACTIVE
        message = str(exc)
    except ModelLicenseExpiredError as exc:
        status = STATUS_EXPIRED
        message = str(exc)
    except ModelLicenseDependencyError as exc:
        status = STATUS_DEPENDENCY_MISSING
        message = str(exc)
    except ModelLicenseError as exc:
        status = STATUS_INVALID
        message = str(exc)
    else:
        status = (
            STATUS_VERIFIED_COMMERCIAL
            if model_license.commercial_use_permitted
            else STATUS_VERIFIED_NON_COMMERCIAL
        )
        return ModelLicenseInspection(
            package_path=package_path,
            manifest_path=manifest_path,
            manifest_version=manifest_version,
            model_id=model_id,
            license_path=license_path,
            status=status,
            message=(
                "Verified commercial model license"
                if model_license.commercial_use_permitted
                else "Verified non-commercial model license"
            ),
            license=model_license,
        )
    return ModelLicenseInspection(
        package_path=package_path,
        manifest_path=manifest_path,
        manifest_version=manifest_version,
        model_id=model_id,
        license_path=license_path,
        status=status,
        message=message,
    )


inspect_model_license = inspect_model_package_license


__all__ = [
    "GRANTS",
    "LICENSE_FILENAME",
    "LICENSE_ISSUER",
    "LICENSE_VERSION",
    "MAX_LICENSE_BYTES",
    "ModelLicense",
    "ModelLicenseDependencyError",
    "ModelLicenseError",
    "ModelLicenseExpiredError",
    "ModelLicenseInspection",
    "ModelLicenseManifestError",
    "ModelLicenseNotActiveError",
    "STATUS_DEFAULT_NON_COMMERCIAL",
    "STATUS_DEPENDENCY_MISSING",
    "STATUS_EXPIRED",
    "STATUS_INVALID",
    "STATUS_INVALID_MANIFEST",
    "STATUS_NOT_ACTIVE",
    "STATUS_VERIFIED_COMMERCIAL",
    "STATUS_VERIFIED_NON_COMMERCIAL",
    "canonical_license_bytes",
    "inspect_model_license",
    "inspect_model_package_license",
    "verify_model_license",
]
