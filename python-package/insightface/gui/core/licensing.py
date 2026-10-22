"""License center helpers."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from ...model_zoo.model_license import (
    STATUS_DEFAULT_NON_COMMERCIAL,
    STATUS_DEPENDENCY_MISSING,
    STATUS_EXPIRED,
    STATUS_INVALID,
    STATUS_INVALID_MANIFEST,
    STATUS_NOT_ACTIVE,
    STATUS_VERIFIED_COMMERCIAL,
    STATUS_VERIFIED_NON_COMMERCIAL,
    ModelLicenseInspection,
    inspect_model_package_license,
)

from .constants import (
    COMMERCIAL_NOTICE,
    DEFAULT_LICENSE_STATUS,
    LICENSE_NOTICE,
    RESPONSIBLE_USE_NOTICE,
)
from .i18n import tr
from .model_packages import GUI_MODEL_PACKAGES


_LICENSE_CACHE_LOCK = threading.RLock()
_LICENSE_CACHE: dict[
    tuple[str, str | None],
    tuple[
        tuple[Any, ...],
        Path | None,
        tuple[Any, ...],
        int,
        ModelLicenseInspection,
    ],
] = {}
_MAX_LICENSE_CACHE_ENTRIES = 64


@dataclass(frozen=True)
class ModelLicenseDisplay:
    """Presentation state derived from one model package license inspection."""

    status_text: str
    detail_text: str
    inspection: ModelLicenseInspection
    is_error: bool = False

    def tooltip(self, language: str | None = None) -> str:
        lines = [tr(self.detail_text, language)]
        message = str(self.inspection.message or "").strip()
        if self.is_error and message and message != self.detail_text:
            lines.append(message)
        if self.inspection.license_path is not None:
            lines.append(f"MODEL.LICENSE: {self.inspection.license_path}")
        else:
            lines.append(f"Model directory: {self.inspection.package_path}")
        return "\n".join(lines)


def resolve_configured_model_dir(config, *, model_name: str | None = None) -> Path:
    """Resolve a GUI model directory with the same rules as ``FaceEngine``.

    Built-in catalog choices intentionally ignore a stale custom directory,
    matching ``create_face_engine``.  Merely resolving the path never creates a
    ``FaceAnalysis`` instance or an ONNX Runtime Session.
    """

    selected_name = str(model_name if model_name is not None else config.model_name)
    custom_value = ""
    if model_name is None and selected_name not in GUI_MODEL_PACKAGES:
        custom_value = str(getattr(config, "custom_model_dir", "") or "")
    if custom_value:
        custom_path = Path(os.path.expanduser(custom_value))
        if custom_path.exists():
            return custom_path
    direct_path = Path(os.path.expanduser(selected_name))
    if direct_path.exists() and direct_path.is_dir():
        return direct_path
    model_root = Path(
        os.path.expanduser(
            str(getattr(config, "model_root", "") or "~/.insightface")
        )
    )
    return model_root / "models" / selected_name


def _path_stamp(path: Path | None) -> tuple[Any, ...]:
    if path is None:
        return (False,)
    try:
        stat = path.stat()
    except OSError:
        return (False,)
    return (
        True,
        path.is_dir(),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def invalidate_model_license_cache() -> None:
    """Discard GUI license inspections after model selection or file changes."""

    with _LICENSE_CACHE_LOCK:
        _LICENSE_CACHE.clear()


def inspect_configured_model_license(
    config,
    *,
    model_name: str | None = None,
) -> ModelLicenseInspection:
    """Inspect the configured package without initializing inference models."""

    selected_name = str(model_name if model_name is not None else config.model_name)
    model_dir = resolve_configured_model_dir(config, model_name=model_name)
    expected_model_id = selected_name if selected_name in GUI_MODEL_PACKAGES else None
    try:
        cache_path = model_dir.expanduser().resolve()
    except (OSError, RuntimeError):
        cache_path = model_dir.absolute()
    key = (str(cache_path), expected_model_id)
    manifest_path = model_dir / "manifest.json"
    directory_and_manifest_stamp = (
        _path_stamp(model_dir),
        _path_stamp(manifest_path),
    )
    # Verification includes time-sensitive activation and expiry checks.  A
    # minute bucket bounds staleness even when files themselves are unchanged.
    minute_bucket = int(time.time() // 60)
    with _LICENSE_CACHE_LOCK:
        cached = _LICENSE_CACHE.get(key)
        if cached is not None:
            cached_stamp, license_path, license_stamp, checked_minute, inspection = cached
            if (
                cached_stamp == directory_and_manifest_stamp
                and license_stamp == _path_stamp(license_path)
                and checked_minute == minute_bucket
            ):
                return inspection

    inspection = inspect_model_package_license(
        model_dir,
        expected_model_id=expected_model_id,
    )
    license_stamp = _path_stamp(inspection.license_path)
    with _LICENSE_CACHE_LOCK:
        if len(_LICENSE_CACHE) >= _MAX_LICENSE_CACHE_ENTRIES:
            _LICENSE_CACHE.pop(next(iter(_LICENSE_CACHE)))
        _LICENSE_CACHE[key] = (
            directory_and_manifest_stamp,
            inspection.license_path,
            license_stamp,
            minute_bucket,
            inspection,
        )
    return inspection


def model_license_display(
    config,
    *,
    model_name: str | None = None,
) -> ModelLicenseDisplay:
    inspection = inspect_configured_model_license(config, model_name=model_name)
    status = inspection.status
    if status == STATUS_VERIFIED_COMMERCIAL:
        return ModelLicenseDisplay(
            status_text="Commercial",
            detail_text="Signed model license verified for commercial use.",
            inspection=inspection,
        )
    if status == STATUS_VERIFIED_NON_COMMERCIAL:
        return ModelLicenseDisplay(
            status_text=DEFAULT_LICENSE_STATUS,
            detail_text="Signed model license verified for non-commercial use.",
            inspection=inspection,
        )
    if status == STATUS_DEFAULT_NON_COMMERCIAL:
        return ModelLicenseDisplay(
            status_text=DEFAULT_LICENSE_STATUS,
            detail_text=(
                "MODEL.LICENSE was not found. Non-commercial use is assumed by default."
            ),
            inspection=inspection,
        )
    if status == STATUS_INVALID_MANIFEST:
        status_text = "Invalid model manifest"
        detail_text = "The current model manifest is invalid."
    elif status == STATUS_NOT_ACTIVE:
        status_text = "Model license not active"
        detail_text = "The current model license is not active yet."
    elif status == STATUS_EXPIRED:
        status_text = "Model license expired"
        detail_text = "The current model license has expired."
    elif status == STATUS_DEPENDENCY_MISSING:
        status_text = "License verification unavailable"
        detail_text = "The current model license could not be verified."
    elif status == STATUS_INVALID:
        status_text = "Invalid model license"
        detail_text = "The current model license is invalid."
    else:
        status_text = "License verification unavailable"
        detail_text = "The current model license could not be verified."
    return ModelLicenseDisplay(
        status_text=status_text,
        detail_text=detail_text,
        inspection=inspection,
        is_error=True,
    )


def current_model_license_display(context) -> ModelLicenseDisplay:
    return model_license_display(context.config)


def find_license_text(start: str | Path) -> str:
    root = Path(start).resolve()
    candidates = [root / "LICENSE", root / "LICENSE.md", root.parent / "LICENSE", root.parent / "LICENSE.md"]
    for path in candidates:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            return text[:4000]
    return "Please refer to the LICENSE file in this repository for the code license."


def allowed_usage_summary() -> Dict[str, str]:
    return {
        "Personal research": "yes",
        "Internal evaluation": "depends on model license",
        "Commercial production": "requires commercial model license",
        "Redistribution": "requires explicit permission",
        "SaaS / API usage": "requires commercial agreement",
        "Face swap commercial usage": "requires commercial agreement",
    }


def license_summary_text(status: str, model_name: str, provider: str, workspace: str) -> str:
    lines = [
        "# InsightFace License Summary",
        "",
        f"- Current status: {status}",
        f"- Model: {model_name}",
        f"- Provider: {provider}",
        f"- Workspace: {workspace}",
        "",
        LICENSE_NOTICE,
        COMMERCIAL_NOTICE,
        RESPONSIBLE_USE_NOTICE,
        "This tool does not provide legal advice.",
    ]
    return "\n".join(lines)
