"""Verification for InsightFace-issued model licenses."""

from .model_license import (
    LICENSE_FILENAME,
    ModelLicense,
    ModelLicenseError,
    canonical_license_bytes,
    verify_model_license,
)

__all__ = [
    "LICENSE_FILENAME",
    "ModelLicense",
    "ModelLicenseError",
    "canonical_license_bytes",
    "verify_model_license",
]
