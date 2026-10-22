"""Shared GUI model choices and local package compatibility checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ...model_zoo.package_manifest import (
    DETECTION_TASK,
    MODEL_PACKAGE_MANIFEST,
    RECOGNITION_TASK,
    SUPPORTED_MANIFEST_PACKAGES,
    VERIFICATION_TASK,
    load_model_package,
)


# Keep one ordered catalog for the first-launch wizard, runtime settings, and
# download actions.  The first entry is also the default for a new GUI config.
PERSON_MODEL_PACKAGES = ("cheetah_s", "cheetah_l")
GUI_MODEL_PACKAGES = (
    *PERSON_MODEL_PACKAGES,
    *SUPPORTED_MANIFEST_PACKAGES,
    "buffalo_l",
    "buffalo_m",
    "buffalo_s",
    "buffalo_sc",
    "antelopev2",
)
PRIVATEFRAME_MODEL_PACKAGES = frozenset(SUPPORTED_MANIFEST_PACKAGES)
CUSTOM_MODEL_CHOICE = "__custom_model_directory__"


@dataclass(frozen=True)
class PrivateFrameModelStatus:
    """Read-only readiness result for the current global GUI model selection."""

    model_name: str
    model_root: Path
    package_path: Path
    state: str
    can_start: bool
    message: str

    @property
    def installed(self) -> bool:
        return self.state == "ready"


def model_package_path(model_name: str, model_root: str | Path) -> Path:
    return Path(model_root).expanduser().resolve() / "models" / str(model_name)


def is_gui_model_package_asset(*, name: str, source: str) -> bool:
    """Return whether a download-catalog row may become the global GUI model."""

    path = Path(str(name))
    return (
        str(source).casefold() == "insightface"
        and path.suffix.casefold() == ".zip"
        and path.stem in GUI_MODEL_PACKAGES
    )


@dataclass(frozen=True)
class PersonModelStatus:
    model_name: str
    model_root: Path
    package_path: Path
    state: str
    can_start: bool
    message: str

    @property
    def installed(self) -> bool:
        return self.state == "ready"


@lru_cache(maxsize=8)
def _validate_person_package_cached(name: str, root: str, signature: tuple) -> None:
    # The SDK remains the authority for all four model contracts and hashes.
    # File metadata only avoids rehashing unchanged weights on GUI refresh.
    from ...model_zoo.person_package import load_person_package

    del signature
    load_person_package(name, root)


def inspect_person_model(model_name: str, model_root: str | Path) -> PersonModelStatus:
    """Validate a local Cheetah package without network access or inference."""
    name = str(model_name or "").strip()
    root_text = str(model_root or "").strip()
    root = Path(root_text or ".").expanduser().resolve()
    folder = root / "models" / name

    def status(state, can_start, message):
        return PersonModelStatus(name, root, folder, state, can_start, message)

    if name not in PERSON_MODEL_PACKAGES:
        return status("unsupported", False,
                      "Person Analysis supports only cheetah_s or cheetah_l. "
                      "Open Models and select a Cheetah package.")
    if not root_text:
        return status("invalid", False, "Person Analysis model root is empty.")
    for path in (root, root / "models", folder):
        if (path.exists() or path.is_symlink()) and not path.is_dir():
            return status("invalid", False, f"Model path is not a directory: {path}")
    if not folder.exists():
        return status("missing", True,
                      f"{name} is not installed. Starting will try the GitHub model-zoo "
                      f"download. You can also install its complete package at {folder}.")
    try:
        signature = tuple(
            (str(path.relative_to(folder)), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            for path in sorted(folder.rglob("*")) if path.is_file()
            for stat in (path.stat(),)
        )
        _validate_person_package_cached(name, str(root), signature)
    except Exception as exc:
        return status("invalid", False,
                      f"Person Analysis cannot use {folder}: {exc}. "
                      "Repair the local package; it will not be overwritten automatically.")
    return status("ready", True, f"Ready: {name} from {folder}.")


def ensure_person_model(
    model_name: str,
    model_root: str | Path,
    cache_dir: str | Path | None = None,
    progress=None,
    is_cancelled=None,
) -> Path:
    """Use valid local weights, downloading only a completely absent package.

    Run this in a worker thread. Cancellation is checked between synchronous
    download operations; it does not interrupt a pending HTTP read.
    """
    from .model_downloads import ModelAsset, download_model_asset
    from ...utils.storage import model_zoo_download_url

    if is_cancelled is not None and is_cancelled():
        raise RuntimeError("Person model preparation cancelled.")
    status = inspect_person_model(model_name, model_root)
    if status.installed:
        return status.package_path
    if not status.can_start:
        raise RuntimeError(status.message)
    asset = ModelAsset(
        name=f"{status.model_name}.zip",
        browser_download_url=model_zoo_download_url(f"{status.model_name}.zip"),
    )
    return download_model_asset(
        asset, status.model_root,
        cache_dir if cache_dir is not None else status.model_root / "gui" / "cache",
        progress=progress, is_cancelled=is_cancelled,
    )


def person_model_providers(choice: str) -> list[str]:
    """Resolve Cheetah providers without silently substituting explicit CUDA."""
    import onnxruntime

    choice = str(choice or "Auto").strip()
    if choice not in {"Auto", "CPU", "CUDA"}:
        raise ValueError("Person Analysis supports Auto, CPU, or CUDA providers.")
    available = onnxruntime.get_available_providers()
    primary = (
        "CUDAExecutionProvider"
        if choice == "CUDA" or (choice == "Auto" and "CUDAExecutionProvider" in available)
        else "CPUExecutionProvider"
    )
    if primary not in available:
        raise RuntimeError(f"Person Analysis requested provider is unavailable: {primary}")
    providers = [primary]
    if primary != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return providers


def person_provider_runtime_display(choice: str, language: str | None = None) -> tuple[str, str]:
    """Describe the same CPU/CUDA policy used by the Person Analysis runner."""
    from .i18n import tr

    def localized(text):
        return tr(text, language) if language is not None else text

    try:
        providers = person_model_providers(choice)
    except (RuntimeError, ValueError) as exc:
        return localized("Unavailable"), str(exc)
    return providers[0], localized(
        "Configured selection: {selection}. Resolved provider chain: {chain}."
    ).format(selection=choice, chain=" → ".join(providers))


def inspect_privateframe_model(
    model_name: str,
    model_root: str | Path,
    *,
    require_recognition: bool = False,
) -> PrivateFrameModelStatus:
    """Inspect a Raccoon package without downloading it or creating Sessions.

    A completely absent supported package remains runnable because PrivateFrame
    intentionally lets ModelZoo download it on first use.  Once a package
    directory exists, however, it must be a usable V2 package; silently falling
    back or downloading over a partial/corrupt directory would hide a bad local
    installation.
    """

    normalized_name = str(model_name or "").strip()
    root_text = str(model_root or "").strip()
    root = Path(root_text or ".").expanduser().resolve()
    package_path = root / "models" / normalized_name
    if normalized_name not in PRIVATEFRAME_MODEL_PACKAGES:
        return PrivateFrameModelStatus(
            model_name=normalized_name,
            model_root=root,
            package_path=package_path,
            state="unsupported",
            can_start=False,
            message=(
                "PrivateFrame supports only raccoon_s or raccoon_l. "
                "Open Models and select a Raccoon package."
            ),
        )
    if not root_text:
        return _invalid_privateframe_status(
            normalized_name,
            root,
            package_path,
            "the model root is empty",
        )
    if root.exists() and not root.is_dir():
        return _invalid_privateframe_status(
            normalized_name,
            root,
            package_path,
            f"the model root is not a directory: {root}",
        )
    models_path = root / "models"
    if models_path.exists() and not models_path.is_dir():
        return _invalid_privateframe_status(
            normalized_name,
            root,
            package_path,
            f"the models path is not a directory: {models_path}",
        )
    if not package_path.exists():
        return PrivateFrameModelStatus(
            model_name=normalized_name,
            model_root=root,
            package_path=package_path,
            state="missing",
            can_start=True,
            message=(
                f"{normalized_name} is not installed under {root / 'models'}. "
                "It will be downloaded there on first use."
            ),
        )
    if not package_path.is_dir():
        return _invalid_privateframe_status(
            normalized_name,
            root,
            package_path,
            f"the package path is not a directory: {package_path}",
        )

    manifest_path = package_path / MODEL_PACKAGE_MANIFEST
    if not manifest_path.is_file():
        return _invalid_privateframe_status(
            normalized_name,
            root,
            package_path,
            f"the V2 manifest is missing: {manifest_path}",
        )
    try:
        package = load_model_package(package_path)
    except Exception as exc:
        return _invalid_privateframe_status(
            normalized_name,
            root,
            package_path,
            str(exc),
        )
    if package.name != normalized_name:
        return _invalid_privateframe_status(
            normalized_name,
            root,
            package_path,
            f"manifest model_id is {package.name!r}",
        )

    required_tasks = [DETECTION_TASK, VERIFICATION_TASK]
    if require_recognition:
        required_tasks.append(RECOGNITION_TASK)
    missing_tasks = [task for task in required_tasks if task not in package.tasks]
    if missing_tasks:
        return _invalid_privateframe_status(
            normalized_name,
            root,
            package_path,
            "missing required task(s): " + ", ".join(missing_tasks),
        )
    missing_files = []
    for task in required_tasks:
        descriptor = package.tasks[task]
        if not descriptor.path.is_file():
            missing_files.append(str(descriptor.path))
    if missing_files:
        return _invalid_privateframe_status(
            normalized_name,
            root,
            package_path,
            "missing model file(s): " + ", ".join(missing_files),
        )
    try:
        for task in required_tasks:
            descriptor = package.tasks[task]
            if descriptor.sha256 is None:
                continue
            stat = descriptor.path.stat()
            actual_sha256 = _cached_sha256(
                str(descriptor.path),
                stat.st_size,
                stat.st_mtime_ns,
            )
            if actual_sha256 != descriptor.sha256:
                raise RuntimeError(
                    f"model SHA-256 mismatch for {descriptor.path}: expected "
                    f"{descriptor.sha256}, got {actual_sha256}"
                )
    except (OSError, RuntimeError) as exc:
        return _invalid_privateframe_status(
            normalized_name,
            root,
            package_path,
            str(exc),
        )
    return PrivateFrameModelStatus(
        model_name=normalized_name,
        model_root=root,
        package_path=package_path,
        state="ready",
        can_start=True,
        message=f"Ready: {normalized_name} from {package_path}.",
    )


def _invalid_privateframe_status(
    model_name: str,
    model_root: Path,
    package_path: Path,
    reason: str,
) -> PrivateFrameModelStatus:
    return PrivateFrameModelStatus(
        model_name=model_name,
        model_root=model_root,
        package_path=package_path,
        state="invalid",
        can_start=False,
        message=f"PrivateFrame cannot use {model_name}: {reason}",
    )


@lru_cache(maxsize=32)
def _cached_sha256(path: str, size: int, mtime_ns: int) -> str:
    """Hash an unchanged artifact once across repeated GUI page refreshes."""

    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CUSTOM_MODEL_CHOICE",
    "GUI_MODEL_PACKAGES",
    "PRIVATEFRAME_MODEL_PACKAGES",
    "PERSON_MODEL_PACKAGES",
    "PersonModelStatus",
    "PrivateFrameModelStatus",
    "inspect_privateframe_model",
    "inspect_person_model",
    "ensure_person_model",
    "person_model_providers",
    "person_provider_runtime_display",
    "is_gui_model_package_asset",
    "model_package_path",
]
