"""Model catalog and downloads, including local-first Person Analysis startup."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ...utils.storage import (
    MODEL_ZOO_RELEASE_DOWNLOAD_URL,
    MODEL_ZOO_RELEASE_TAG,
    model_zoo_download_url,
)
from .. import __version__
from .utils import utc_now_iso

GITHUB_MODEL_ZOO_DOWNLOAD_URL = MODEL_ZOO_RELEASE_DOWNLOAD_URL
GITHUB_RELEASES_URL = (
    "https://github.com/deepinsight/insightface/releases/tag/model-zoo"
)
GITHUB_MODEL_ZOO_RELEASE_API = (
    "https://api.github.com/repos/deepinsight/insightface/releases/tags/model-zoo"
)
# Compatibility alias for integrations that imported the previous constant.
GITHUB_LATEST_RELEASE_API = GITHUB_MODEL_ZOO_RELEASE_API
FALLBACK_RELEASE_TAG = MODEL_ZOO_RELEASE_TAG
FALLBACK_RELEASE_NAME = "InsightFace Model Zoo"
GFPGAN_MODEL_NAME = "GFPGANv1.4.onnx"
GFPGAN_DOWNLOAD_URL = (
    "https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/"
    "Models/GFPGANv1.4.onnx"
)

INSIGHTFACE_MODEL_ASSET_NAMES = (
    "antelopev2.zip",
    "buffalo_l.zip",
    "buffalo_m.zip",
    "buffalo_s.zip",
    "buffalo_sc.zip",
    "cheetah_l.zip",
    "cheetah_s.zip",
    "inswapper_128.onnx",
    "raccoon_l.zip",
    "raccoon_s.zip",
    "scrfd_person_2.5g.onnx",
)
_INSIGHTFACE_RELEASE_DOWNLOAD_ROOT = (
    "https://github.com/deepinsight/insightface/releases/download/"
)


@dataclass
class ModelAsset:
    name: str
    browser_download_url: str
    tag_name: str = FALLBACK_RELEASE_TAG
    release_name: str = FALLBACK_RELEASE_NAME
    size: int = 0
    content_type: str = ""
    updated_at: str = ""
    source: str = "InsightFace"

    @property
    def stem(self) -> str:
        if self.name.endswith(".zip"):
            return self.name[:-4]
        if self.name.endswith(".onnx"):
            return self.name[:-5]
        return Path(self.name).stem

    @property
    def kind(self) -> str:
        if self.source.lower() != "insightface":
            return "third-party restore model"
        if self.name.endswith(".zip"):
            return "model package"
        if self.name.endswith(".onnx"):
            return "onnx model"
        return "asset"


def fallback_model_assets() -> list[ModelAsset]:
    """Return the bundled catalog for the dedicated model-zoo release."""

    assets = [
        ModelAsset(
            name=name,
            browser_download_url=model_zoo_download_url(name),
            tag_name=FALLBACK_RELEASE_TAG,
            release_name=FALLBACK_RELEASE_NAME,
        )
        for name in INSIGHTFACE_MODEL_ASSET_NAMES
    ]
    return merge_required_assets(assets)


def third_party_model_assets() -> list[ModelAsset]:
    return [
        ModelAsset(
            name=GFPGAN_MODEL_NAME,
            browser_download_url=GFPGAN_DOWNLOAD_URL,
            tag_name="third-party",
            release_name="harisreedhar/Face-Upscalers-ONNX GFPGAN v1.4",
            size=340 * 1024 * 1024,
            content_type="application/octet-stream",
            source="third party",
        )
    ]


def merge_required_assets(assets: Iterable[ModelAsset]) -> list[ModelAsset]:
    merged = [_normalize_insightface_asset(asset) for asset in assets]
    seen = {asset.name for asset in merged}
    for name in INSIGHTFACE_MODEL_ASSET_NAMES:
        if name not in seen:
            merged.append(
                ModelAsset(
                    name=name,
                    browser_download_url=model_zoo_download_url(name),
                    tag_name=FALLBACK_RELEASE_TAG,
                    release_name=FALLBACK_RELEASE_NAME,
                )
            )
            seen.add(name)
    for asset in third_party_model_assets():
        if asset.name not in seen:
            merged.append(asset)
            seen.add(asset.name)
    return merged


def _normalize_insightface_asset(asset: ModelAsset) -> ModelAsset:
    """Point official assets at the canonical release, including cached ones."""

    is_official = asset.name in INSIGHTFACE_MODEL_ASSET_NAMES or (
        asset.browser_download_url.startswith(_INSIGHTFACE_RELEASE_DOWNLOAD_ROOT)
    )
    if asset.source.casefold() != "insightface" or not is_official:
        return asset
    release_name = asset.release_name
    if asset.tag_name != MODEL_ZOO_RELEASE_TAG:
        release_name = FALLBACK_RELEASE_NAME
    return replace(
        asset,
        browser_download_url=model_zoo_download_url(asset.name),
        tag_name=MODEL_ZOO_RELEASE_TAG,
        release_name=release_name or FALLBACK_RELEASE_NAME,
    )


def cache_file(gui_cache_dir: str | os.PathLike[str]) -> Path:
    path = Path(gui_cache_dir).expanduser() / "model_download_urls.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def asset_from_dict(data: dict) -> ModelAsset:
    name = str(data.get("name", ""))
    download_url = str(data.get("browser_download_url", ""))
    source = data.get("source")
    if source is None:
        source = (
            "InsightFace"
            if name in INSIGHTFACE_MODEL_ASSET_NAMES
            or download_url.startswith(_INSIGHTFACE_RELEASE_DOWNLOAD_ROOT)
            else "third party"
        )
    return _normalize_insightface_asset(
        ModelAsset(
            name=name,
            browser_download_url=download_url,
            tag_name=str(
                data.get("tag_name", data.get("release_tag", FALLBACK_RELEASE_TAG))
            ),
            release_name=str(data.get("release_name", FALLBACK_RELEASE_NAME)),
            size=int(data.get("size") or 0),
            content_type=str(data.get("content_type", "")),
            updated_at=str(data.get("updated_at", "")),
            source=str(source),
        )
    )


def load_cached_assets(gui_cache_dir: str | os.PathLike[str]) -> list[ModelAsset]:
    path = cache_file(gui_cache_dir)
    if not path.exists():
        return fallback_model_assets()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assets = [
            asset_from_dict(item)
            for item in payload.get("assets", [])
            if item.get("browser_download_url")
        ]
        assets = merge_required_assets(assets)
        normalized = [asdict(asset) for asset in assets]
        if payload.get("assets") != normalized:
            try:
                save_cached_assets(
                    gui_cache_dir,
                    assets,
                    source=GITHUB_MODEL_ZOO_RELEASE_API,
                )
            except OSError:
                # A read-only cache must not prevent the in-memory migration.
                pass
        return assets
    except (AttributeError, OSError, TypeError, ValueError):
        return fallback_model_assets()


def save_cached_assets(
    gui_cache_dir: str | os.PathLike[str],
    assets: Iterable[ModelAsset],
    source: str = "",
) -> Path:
    path = cache_file(gui_cache_dir)
    payload = {
        "source": source or GITHUB_MODEL_ZOO_RELEASE_API,
        "refreshed_at": utc_now_iso(),
        "assets": [asdict(asset) for asset in assets],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def refresh_model_assets(
    gui_cache_dir: str | os.PathLike[str],
    timeout: int = 20,
) -> tuple[list[ModelAsset], str]:
    request = urllib.request.Request(
        GITHUB_MODEL_ZOO_RELEASE_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"InsightFace-Evaluation-Studio/{__version__}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        assets = fallback_model_assets()
        save_cached_assets(
            gui_cache_dir,
            assets,
            source=f"fallback after refresh error: {exc}",
        )
        return (
            assets,
            f"Refresh failed; using bundled {FALLBACK_RELEASE_TAG} URLs. Error: {exc}",
        )

    release_name = payload.get("name") or FALLBACK_RELEASE_NAME
    assets = []
    for item in payload.get("assets", []):
        name = item.get("name") or ""
        if not name.endswith((".zip", ".onnx")):
            continue
        assets.append(
            ModelAsset(
                name=name,
                browser_download_url=model_zoo_download_url(name),
                tag_name=MODEL_ZOO_RELEASE_TAG,
                release_name=release_name,
                size=int(item.get("size") or 0),
                content_type=item.get("content_type") or "",
                updated_at=item.get("updated_at") or "",
            )
        )
    if not assets:
        assets = fallback_model_assets()
        message = (
            "The model-zoo release returned no model assets; using bundled "
            f"{FALLBACK_RELEASE_TAG} URLs."
        )
    else:
        assets = merge_required_assets(assets)
        message = f"Refreshed {len(assets)} asset(s), including third-party restore models."
    save_cached_assets(gui_cache_dir, assets, source=GITHUB_MODEL_ZOO_RELEASE_API)
    return assets, message


def installed_model_asset_path(
    asset: ModelAsset,
    model_root: str | os.PathLike[str],
) -> Path | None:
    """Return the installed asset path, or ``None`` for absent/partial assets."""

    root = Path(model_root).expanduser()
    if asset.name.endswith(".zip"):
        target = root / "models" / asset.stem
        from .model_packages import PERSON_MODEL_PACKAGES, inspect_person_model

        if asset.stem in PERSON_MODEL_PACKAGES:
            status = inspect_person_model(asset.stem, root)
            return status.package_path if status.installed else None
        if target.exists() and any(target.glob("*.onnx")):
            return target
        return None
    target_file = root / "models" / asset.stem / asset.name
    if target_file.exists():
        return target_file
    legacy_file = root / "models" / asset.name
    if legacy_file.exists():
        return legacy_file
    return None


def is_model_asset_installed(
    asset: ModelAsset,
    model_root: str | os.PathLike[str],
) -> bool:
    """Return whether a complete local copy of a catalog asset is available."""

    return installed_model_asset_path(asset, model_root) is not None


def local_model_status(asset: ModelAsset, model_root: str | os.PathLike[str]) -> str:
    from .model_packages import PERSON_MODEL_PACKAGES, inspect_person_model

    if asset.stem in PERSON_MODEL_PACKAGES:
        status = inspect_person_model(asset.stem, model_root)
        if status.state == "invalid":
            return status.message
    installed_path = installed_model_asset_path(asset, model_root)
    if installed_path is not None:
        return f"installed: {installed_path}"
    if asset.name.endswith(".zip"):
        target = Path(model_root).expanduser() / "models" / asset.stem
        if target.exists():
            return f"folder exists: {target}"
    return "not installed"


def is_model_package_installed(model_name: str, model_root: str | os.PathLike[str]) -> bool:
    from .model_packages import PERSON_MODEL_PACKAGES, inspect_person_model

    if model_name in PERSON_MODEL_PACKAGES:
        return inspect_person_model(model_name, model_root).installed
    target = Path(model_root).expanduser() / "models" / model_name
    return target.exists() and any(target.glob("*.onnx"))


def list_installed_swap_models(model_root: str | os.PathLike[str]) -> list[Path]:
    root = Path(model_root).expanduser() / "models"
    if not root.exists():
        return []
    paths = []
    for path in root.rglob("*.onnx"):
        name = path.name.lower()
        parent = path.parent.name.lower()
        if "swap" in name or "inswapper" in name or "swap" in parent or "inswapper" in parent:
            paths.append(path)
    return sorted(paths)


def list_installed_gfpgan_models(model_root: str | os.PathLike[str]) -> list[Path]:
    root = Path(model_root).expanduser() / "models"
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.onnx") if "gfpgan" in path.name.lower())


def _content_range_total(value: str | None) -> int:
    if not value or "/" not in value:
        return 0
    total = value.rsplit("/", 1)[-1].strip()
    if not total or total == "*":
        return 0
    try:
        return int(total)
    except ValueError:
        return 0


def _download_with_retries(
    url: str,
    destination: Path,
    asset_name: str,
    expected_size: int = 0,
    progress: Callable[[int, int, str], None] | None = None,
    retries: int = 4,
) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {
            "Accept": "application/octet-stream,*/*",
            "User-Agent": f"Mozilla/5.0 InsightFace-Evaluation-Studio/{__version__}",
        }
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                content_length = int(response.headers.get("Content-Length") or 0)
                content_range_total = _content_range_total(response.headers.get("Content-Range"))
                if existing and status != 206:
                    existing = 0
                    mode = "wb"
                elif existing:
                    mode = "ab"
                else:
                    mode = "wb"
                if content_range_total:
                    total = content_range_total
                elif status == 206:
                    total = existing + content_length
                else:
                    total = content_length or expected_size
                downloaded = existing if mode == "ab" else 0
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if progress:
                            progress(downloaded, total or downloaded, f"Downloading {asset_name}")
            partial.replace(destination)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (404, 410):
                raise RuntimeError(
                    f"{asset_name} is not available at {url} (HTTP {exc.code}). "
                    "The release asset may not have been published yet."
                ) from exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 416 and partial.exists():
                partial.replace(destination)
                return
            last_error = exc
            if progress:
                progress(
                    existing,
                    expected_size or existing or 1,
                    f"Download interrupted; retrying {asset_name} ({attempt}/{retries})",
                )
            if attempt < retries:
                time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"Download failed for {asset_name} after {retries} attempt(s): {last_error}")


def download_model_asset(
    asset: ModelAsset,
    model_root: str | os.PathLike[str],
    gui_cache_dir: str | os.PathLike[str],
    progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Path:
    from .model_packages import PERSON_MODEL_PACKAGES

    if asset.name.endswith(".zip") and asset.stem in PERSON_MODEL_PACKAGES:
        return _download_person_model_asset(
            asset, model_root, gui_cache_dir, progress, is_cancelled,
        )
    cache_dir = Path(gui_cache_dir).expanduser() / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_root_path = Path(model_root).expanduser() / "models"
    model_root_path.mkdir(parents=True, exist_ok=True)

    archive_path = cache_dir / asset.name
    _download_with_retries(
        asset.browser_download_url,
        archive_path,
        asset.name,
        expected_size=asset.size,
        progress=progress,
    )

    if asset.name.endswith(".zip"):
        target_dir = model_root_path / asset.stem
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{asset.stem}-install-",
                dir=model_root_path,
            )
        )
        backup_dir: Path | None = None
        install_committed = False
        try:
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(staging_dir)
            if not any(staging_dir.rglob("*.onnx")):
                raise RuntimeError(
                    f"Downloaded package {asset.name} contains no ONNX models."
                )
            if target_dir.exists():
                backup_dir = Path(
                    tempfile.mkdtemp(
                        prefix=f".{asset.stem}-previous-",
                        dir=model_root_path,
                    )
                )
                backup_dir.rmdir()
                target_dir.replace(backup_dir)
            try:
                staging_dir.replace(target_dir)
                install_committed = True
            except Exception:
                if backup_dir is not None and not target_dir.exists():
                    backup_dir.replace(target_dir)
                    backup_dir = None
                raise
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            if (
                install_committed
                and backup_dir is not None
                and backup_dir.exists()
            ):
                shutil.rmtree(backup_dir, ignore_errors=True)
        if progress:
            progress(1, 1, f"Extracted to {target_dir}")
        return target_dir

    target_dir = model_root_path / asset.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / asset.name
    staging_file = target_dir / f".{asset.name}.installing"
    try:
        shutil.copy2(archive_path, staging_file)
        staging_file.replace(target_file)
    finally:
        staging_file.unlink(missing_ok=True)
    if progress:
        progress(1, 1, f"Saved to {target_file}")
    return target_file


def _download_person_model_asset(asset, model_root, gui_cache_dir, progress, is_cancelled):
    """Validate Cheetah in staging, preserving every existing local package."""
    from .model_packages import inspect_person_model
    from ...model_zoo.person_package import load_person_package

    def check_cancelled():
        if is_cancelled is not None and is_cancelled():
            raise RuntimeError("Person model preparation cancelled.")

    def report(current, total, message):
        check_cancelled()
        if progress is not None:
            progress(current, total, message)

    check_cancelled()
    status = inspect_person_model(asset.stem, model_root)
    if status.installed:
        return status.package_path
    if not status.can_start:
        raise RuntimeError(status.message)
    cache = Path(gui_cache_dir).expanduser() / "models"
    cache.mkdir(parents=True, exist_ok=True)
    status.package_path.parent.mkdir(parents=True, exist_ok=True)
    archive = cache / asset.name
    url = model_zoo_download_url(asset.name)
    try:
        _download_with_retries(url, archive, asset.name, expected_size=asset.size, progress=report)
        check_cancelled()
        with tempfile.TemporaryDirectory(
            prefix=f".{asset.stem}-install-", dir=status.package_path.parent,
        ) as temporary:
            staging_root = Path(temporary)
            staging_package = staging_root / "models" / asset.stem
            staging_package.mkdir(parents=True)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(staging_package)
            # Accept both a package-content zip and a zip with one named folder.
            wrapped_package = staging_package / asset.stem
            if not (staging_package / "manifest.json").exists() and (wrapped_package / "manifest.json").is_file():
                unpacked = staging_root / "unpacked"
                wrapped_package.replace(unpacked)
                shutil.rmtree(staging_package)
                unpacked.replace(staging_package)
            check_cancelled()
            load_person_package(asset.stem, staging_root)
            check_cancelled()
            # A concurrent installer or user may have created a local package
            # during the download. Never replace it with our staged result.
            if status.package_path.exists() or status.package_path.is_symlink():
                current = inspect_person_model(asset.stem, status.model_root)
                if current.installed:
                    return current.package_path
                raise RuntimeError(f"Local package appeared during download: {status.package_path}. It was not replaced.")
            staging_package.replace(status.package_path)
        report(1, 1, f"Installed and validated {asset.name} at {status.package_path}")
        return status.package_path
    except Exception as exc:
        raise RuntimeError(
            f"Could not prepare {asset.stem}: {exc} "
            f"Install the complete package (manifest.json and its four models) at {status.package_path}."
        ) from exc
