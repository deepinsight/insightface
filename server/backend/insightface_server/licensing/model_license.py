"""Compatibility exports for the shared InsightFace model-license contract.

The implementation and trusted public keys live in the Python package so the
GUI and Server apply one verification policy. This module deliberately
preserves the Server's historical import path.
"""

from insightface.model_zoo.model_license import (
    GRANTS,
    LICENSE_FILENAME,
    LICENSE_ISSUER,
    LICENSE_VERSION,
    MAX_LICENSE_BYTES,
    STATUS_DEFAULT_NON_COMMERCIAL,
    STATUS_DEPENDENCY_MISSING,
    STATUS_EXPIRED,
    STATUS_INVALID,
    STATUS_INVALID_MANIFEST,
    STATUS_NOT_ACTIVE,
    STATUS_VERIFIED_COMMERCIAL,
    STATUS_VERIFIED_NON_COMMERCIAL,
    ModelLicense,
    ModelLicenseDependencyError,
    ModelLicenseError,
    ModelLicenseExpiredError,
    ModelLicenseInspection,
    ModelLicenseManifestError,
    ModelLicenseNotActiveError,
    canonical_license_bytes,
    inspect_model_license,
    inspect_model_package_license,
    verify_model_license,
)
from insightface.model_zoo.model_license import (
    _trusted_keys as _trusted_keys,
)

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
