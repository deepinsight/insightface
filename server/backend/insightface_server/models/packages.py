"""Explicitly installed model packages for InsightFace Server.

Official downloads are checksum-verified before installation. Runtime model
licenses intentionally bind only the logical ``model_id`` so customers may use
converted FP16, INT8, optimized ONNX, or TensorRT artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..licensing import LICENSE_FILENAME, ModelLicense, verify_model_license
from .manifest import load_manifest, sha256_file

MODEL_LICENSE_SUMMARY = "Non-commercial research use only."
MODEL_LICENSE_ID = "non-commercial-research-only"
MODEL_LICENSE_FILENAME = LICENSE_FILENAME
MODEL_LICENSE_URL = "https://github.com/deepinsight/insightface#license"
COMMERCIAL_LICENSE_URL = "https://www.insightface.ai"
DEFAULT_LICENSES_DIR = (
    Path(__file__).resolve().parents[1] / "licensing" / "defaults"
)
MAX_ARCHIVE_BYTES = 400 * 1024 * 1024
MAX_MODEL_BYTES = 256 * 1024 * 1024
MODEL_ZOO_RELEASE_BASE_URL = (
    "https://github.com/deepinsight/insightface/releases/download/model-zoo/"
)


def _model_zoo_release_url(package_name: str) -> str:
    return f"{MODEL_ZOO_RELEASE_BASE_URL}{package_name}.zip"


class ModelPackageError(RuntimeError):
    """A safe, user-facing model package operation failed."""


@dataclass(frozen=True, slots=True)
class PackageFile:
    filename: str
    sha256: str
    model_id: str
    task: str
    input_size: tuple[int, int]
    preprocessing_version: str
    input_mean: float
    input_std: float
    embedding_dimension: int | None = None


@dataclass(frozen=True, slots=True)
class ModelPackage:
    name: str
    display_name: str
    url: str
    archive_sha256: str
    files: tuple[PackageFile, ...]
    license_summary: str = MODEL_LICENSE_SUMMARY
    license_id: str = MODEL_LICENSE_ID
    license_url: str = MODEL_LICENSE_URL
    commercial_license_url: str = COMMERCIAL_LICENSE_URL


BUFFALO_L = ModelPackage(
    name="buffalo_l",
    display_name="Buffalo_L",
    url=_model_zoo_release_url("buffalo_l"),
    archive_sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
    files=(
        PackageFile(
            filename="det_10g.onnx",
            sha256="5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
            model_id="scrfd-detection",
            task="face_detection",
            input_size=(640, 640),
            preprocessing_version="insightface-scrfd-1",
            input_mean=127.5,
            input_std=128.0,
        ),
        PackageFile(
            filename="w600k_r50.onnx",
            sha256="4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
            model_id="buffalo_l-recognition",
            task="face_recognition",
            input_size=(112, 112),
            preprocessing_version="insightface-arcface-1",
            input_mean=127.5,
            input_std=127.5,
            embedding_dimension=512,
        ),
    ),
)

BUFFALO_M = ModelPackage(
    name="buffalo_m",
    display_name="Buffalo_M",
    url=_model_zoo_release_url("buffalo_m"),
    archive_sha256="d98264bd8f2dc75cbc2ddce2a14e636e02bb857b3051c234b737bf3b614edca9",
    files=(
        PackageFile(
            filename="det_2.5g.onnx",
            sha256="041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9",
            model_id="buffalo_m-detection",
            task="face_detection",
            input_size=(640, 640),
            preprocessing_version="insightface-scrfd-1",
            input_mean=127.5,
            input_std=128.0,
        ),
        PackageFile(
            filename="w600k_r50.onnx",
            sha256="4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
            model_id="buffalo_m-recognition",
            task="face_recognition",
            input_size=(112, 112),
            preprocessing_version="insightface-arcface-1",
            input_mean=127.5,
            input_std=127.5,
            embedding_dimension=512,
        ),
    ),
)

BUFFALO_S = ModelPackage(
    name="buffalo_s",
    display_name="Buffalo_S",
    url=_model_zoo_release_url("buffalo_s"),
    archive_sha256="d85a87f503f691807cd8bb97128bdf7a0660326cd9cd02657127fa978bab8b5e",
    files=(
        PackageFile(
            filename="det_500m.onnx",
            sha256="5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
            model_id="buffalo_s-detection",
            task="face_detection",
            input_size=(640, 640),
            preprocessing_version="insightface-scrfd-1",
            input_mean=127.5,
            input_std=128.0,
        ),
        PackageFile(
            filename="w600k_mbf.onnx",
            sha256="9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
            model_id="buffalo_s-recognition",
            task="face_recognition",
            input_size=(112, 112),
            preprocessing_version="insightface-arcface-1",
            input_mean=127.5,
            input_std=127.5,
            embedding_dimension=512,
        ),
    ),
)

BUFFALO_SC = ModelPackage(
    name="buffalo_sc",
    display_name="Buffalo_SC",
    url=_model_zoo_release_url("buffalo_sc"),
    archive_sha256="57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72",
    files=(
        PackageFile(
            filename="det_500m.onnx",
            sha256="5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
            model_id="buffalo_sc-detection",
            task="face_detection",
            input_size=(640, 640),
            preprocessing_version="insightface-scrfd-1",
            input_mean=127.5,
            input_std=128.0,
        ),
        PackageFile(
            filename="w600k_mbf.onnx",
            sha256="9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
            model_id="buffalo_sc-recognition",
            task="face_recognition",
            input_size=(112, 112),
            preprocessing_version="insightface-arcface-1",
            input_mean=127.5,
            input_std=127.5,
            embedding_dimension=512,
        ),
    ),
)

ANTELOPEV2 = ModelPackage(
    name="antelopev2",
    display_name="AntelopeV2",
    url=_model_zoo_release_url("antelopev2"),
    archive_sha256="8e182f14fc6e80b3bfa375b33eb6cff7ee05d8ef7633e738d1c89021dcf0c5c5",
    files=(
        PackageFile(
            filename="scrfd_10g_bnkps.onnx",
            sha256="5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
            model_id="antelopev2-detection",
            task="face_detection",
            input_size=(640, 640),
            preprocessing_version="insightface-scrfd-1",
            input_mean=127.5,
            input_std=128.0,
        ),
        PackageFile(
            filename="glintr100.onnx",
            sha256="4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf",
            model_id="antelopev2-recognition",
            task="face_recognition",
            input_size=(112, 112),
            preprocessing_version="insightface-arcface-1",
            input_mean=127.5,
            input_std=127.5,
            embedding_dimension=512,
        ),
    ),
)

RACCOON_S = ModelPackage(
    name="raccoon_s",
    display_name="Raccoon S",
    url=_model_zoo_release_url("raccoon_s"),
    archive_sha256="f67a624ef8a4495899eb4359a8a6953f7b4c62a8399c5bc745c0e0f6582f898d",
    files=(
        PackageFile(
            filename="det_10g_wo.onnx",
            sha256="9f68c41d8150ec20939e5527bca5686d2fb921b5a4c1f2554d618e70c73e6b5b",
            model_id="raccoon_s-detection",
            task="face_detection",
            input_size=(640, 640),
            preprocessing_version="insightface-scrfd-1",
            input_mean=127.5,
            input_std=128.0,
        ),
        PackageFile(
            filename="w600k_mbf.onnx",
            sha256="9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
            model_id="raccoon_s-recognition",
            task="face_recognition",
            input_size=(112, 112),
            preprocessing_version="insightface-arcface-1",
            input_mean=127.5,
            input_std=127.5,
            embedding_dimension=512,
        ),
    ),
)

RACCOON_L = ModelPackage(
    name="raccoon_l",
    display_name="Raccoon L",
    url=_model_zoo_release_url("raccoon_l"),
    archive_sha256="70cd4f2f1de0a89dd0983bdac55a066a6178543f86e9ec87154f6f259bdded7e",
    files=(
        PackageFile(
            filename="det_10g_wo.onnx",
            sha256="9f68c41d8150ec20939e5527bca5686d2fb921b5a4c1f2554d618e70c73e6b5b",
            model_id="raccoon_l-detection",
            task="face_detection",
            input_size=(640, 640),
            preprocessing_version="insightface-scrfd-1",
            input_mean=127.5,
            input_std=128.0,
        ),
        PackageFile(
            filename="w600k_r50.onnx",
            sha256="4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
            model_id="raccoon_l-recognition",
            task="face_recognition",
            input_size=(112, 112),
            preprocessing_version="insightface-arcface-1",
            input_mean=127.5,
            input_std=127.5,
            embedding_dimension=512,
        ),
    ),
)

PACKAGES: dict[str, ModelPackage] = {
    package.name: package
    for package in (
        BUFFALO_L,
        BUFFALO_M,
        BUFFALO_S,
        BUFFALO_SC,
        ANTELOPEV2,
        RACCOON_S,
        RACCOON_L,
    )
}


def package_for_name(name: str) -> ModelPackage:
    try:
        return PACKAGES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(PACKAGES))
        raise ModelPackageError(
            f"Unknown model package {name!r}. Supported packages: {supported}"
        ) from exc


def license_notice(package: ModelPackage) -> str:
    return "\n".join(
        (
            "LICENSE NOTICE",
            f"{package.name} is provided for non-commercial research use only.",
            "Commercial use requires a separate license.",
            f"Licensing information: {package.commercial_license_url}",
        )
    )


def model_license_document(package: ModelPackage) -> str:
    """Return the InsightFace-signed default license for a catalog package."""

    path = DEFAULT_LICENSES_DIR / package.name / MODEL_LICENSE_FILENAME
    try:
        document = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ModelPackageError(
            f"Bundled default license is unavailable for {package.name}: {exc}"
        ) from exc
    return document


def _package_matches_installed(package: ModelPackage, models_dir: Path) -> bool:
    try:
        bundle = load_manifest(models_dir)
    except RuntimeError:
        return False
    return bundle.model_id == package.name


def installed_package_name(models_dir: Path) -> str | None:
    try:
        bundle = load_manifest(models_dir)
    except RuntimeError:
        return None
    return bundle.model_id if bundle.model_id in PACKAGES else None


def _requests_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def download_archive(package: ModelPackage, destination: Path) -> None:
    """Download one pinned archive with bounds, retry, and SHA-256 validation."""

    hasher = hashlib.sha256()
    downloaded = 0
    next_report = 10
    try:
        with _requests_session() as session:
            response = session.get(
                package.url,
                stream=True,
                timeout=(15, 300),
                headers={"User-Agent": "InsightFace-Server-Model-Installer/0.3.1"},
            )
            response.raise_for_status()
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > MAX_ARCHIVE_BYTES:
                raise ModelPackageError(
                    f"Model archive exceeds the {MAX_ARCHIVE_BYTES // 1024 // 1024} MiB limit"
                )
            total = int(declared) if declared and declared.isdigit() else 0
            with destination.open("xb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > MAX_ARCHIVE_BYTES:
                        raise ModelPackageError(
                            f"Model archive exceeds the {MAX_ARCHIVE_BYTES // 1024 // 1024} MiB limit"
                        )
                    output.write(chunk)
                    hasher.update(chunk)
                    if total:
                        progress = min(100, int(downloaded * 100 / total))
                        if progress >= next_report:
                            print(f"Download progress: {progress}%")
                            next_report += 10
                output.flush()
                os.fsync(output.fileno())
    except (OSError, requests.RequestException, ValueError) as exc:
        if isinstance(exc, ModelPackageError):
            raise
        raise ModelPackageError(f"Unable to download {package.name}: {exc}") from exc

    actual = hasher.hexdigest()
    if actual != package.archive_sha256:
        raise ModelPackageError(
            f"Archive SHA-256 mismatch: expected {package.archive_sha256}, actual {actual}"
        )


def _safe_member_for_filename(archive: zipfile.ZipFile, filename: str) -> zipfile.ZipInfo:
    matches = []
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        mode = member.external_attr >> 16
        if path.name == filename and not member.is_dir():
            if stat.S_ISLNK(mode):
                raise ModelPackageError(f"Archive entry must not be a symlink: {member.filename}")
            matches.append(member)
    if len(matches) != 1:
        raise ModelPackageError(
            f"Archive must contain exactly one {filename}; found {len(matches)}"
        )
    member = matches[0]
    if member.file_size > MAX_MODEL_BYTES:
        raise ModelPackageError(f"Archive entry is too large: {filename}")
    return member


def _copy_checked(source: IO[bytes], destination: Path, expected_sha256: str) -> None:
    hasher = hashlib.sha256()
    written = 0
    with destination.open("xb") as output:
        while chunk := source.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_MODEL_BYTES:
                raise ModelPackageError(f"Extracted model is too large: {destination.name}")
            output.write(chunk)
            hasher.update(chunk)
        output.flush()
        os.fsync(output.fileno())
    actual = hasher.hexdigest()
    if actual != expected_sha256:
        raise ModelPackageError(
            f"SHA-256 mismatch for {destination.name}: expected {expected_sha256}, actual {actual}"
        )


def extract_required_models(package: ModelPackage, archive_path: Path, output_dir: Path) -> None:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for item in package.files:
                member = _safe_member_for_filename(archive, item.filename)
                with archive.open(member) as source:
                    _copy_checked(source, output_dir / item.filename, item.sha256)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, ModelPackageError):
            raise
        raise ModelPackageError(f"Invalid model archive: {exc}") from exc


def _manifest(package: ModelPackage, relative_bundle: Path) -> dict[str, object]:
    by_task = {item.task: item for item in package.files}
    detector = by_task.get("face_detection")
    recognizer = by_task.get("face_recognition")
    if detector is None or recognizer is None or recognizer.embedding_dimension is None:
        raise ModelPackageError(
            f"Package {package.name} must contain one detector and one recognizer"
        )
    return {
        "manifest_version": 1,
        "model_id": package.name,
        "display_name": package.display_name,
        "files": {
            "detector": (relative_bundle / detector.filename).as_posix(),
            "recognizer": (relative_bundle / recognizer.filename).as_posix(),
        },
        "recognition": {
            "input_size": list(recognizer.input_size),
            "embedding_dimension": recognizer.embedding_dimension,
            "preprocessing": recognizer.preprocessing_version,
        },
        "license": (relative_bundle / MODEL_LICENSE_FILENAME).as_posix(),
    }


def _write_json_fsync(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def _write_text_fsync(path: Path, payload: str) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _managed_license_location(
    package: ModelPackage, models_dir: Path
) -> tuple[Path, Path]:
    root = models_dir.expanduser().resolve()
    bundle = load_manifest(root)
    if bundle.model_id != package.name:
        raise ModelPackageError(
            f"Installed model_id is {bundle.model_id!r}, not {package.name!r}"
        )
    try:
        relative_license = bundle.license_path.relative_to(root)
    except ValueError as exc:  # pragma: no cover - load_manifest already confines paths
        raise ModelPackageError("Installed model license escapes the model directory") from exc
    return root / relative_license, relative_license


LicenseProvider = Callable[[ModelPackage], str]


def _ensure_model_license(
    package: ModelPackage,
    models_dir: Path,
    *,
    license_provider: LicenseProvider,
) -> None:
    """Install a default license and migrate a recognized legacy manifest."""

    root = models_dir.expanduser().resolve()
    owner = root.stat()
    bundle = load_manifest(root)
    license_path, relative_license = _managed_license_location(package, root)
    if license_path.exists():
        verify_model_license(license_path, expected_model_id=package.name)
    else:
        _write_text_fsync(license_path, license_provider(package))
        verify_model_license(license_path, expected_model_id=package.name)

    if bundle.legacy_manifest:
        parents = {model.path.parent for model in bundle.models}
        if len(parents) != 1:
            raise ModelPackageError(
                f"Installed {package.name} model files must share one package directory"
            )
        relative_bundle = parents.pop().relative_to(root)
        payload = _manifest(package, relative_bundle)
        if payload["license"] != relative_license.as_posix():
            raise ModelPackageError("Legacy model license location is inconsistent")
        temporary_manifest = root / f".manifest-{uuid.uuid4().hex}.json"
        try:
            _write_json_fsync(temporary_manifest, payload)
            os.replace(temporary_manifest, root / "manifest.json")
        finally:
            temporary_manifest.unlink(missing_ok=True)

    _restore_owner(license_path, owner.st_uid, owner.st_gid)
    _restore_owner(root / "manifest.json", owner.st_uid, owner.st_gid)
    license_path.chmod(0o644)
    (root / "manifest.json").chmod(0o644)


def _verify_model_license(package: ModelPackage, models_dir: Path) -> ModelLicense:
    root = models_dir.expanduser().resolve()
    license_path, _relative_license = _managed_license_location(package, root)
    return verify_model_license(license_path, expected_model_id=package.name)


def _restore_owner(root: Path, uid: int, gid: int) -> None:
    if not hasattr(os, "chown") or os.geteuid() != 0:
        return
    paths = (root, *root.rglob("*")) if root.is_dir() else (root,)
    for path in paths:
        os.chown(path, uid, gid, follow_symlinks=False)


Downloader = Callable[[ModelPackage, Path], None]


def install_package(
    package: ModelPackage,
    models_dir: Path,
    *,
    downloader: Downloader = download_archive,
    license_provider: LicenseProvider = model_license_document,
) -> str:
    """Install a verified package and publish its manifest last."""

    models_dir = models_dir.expanduser().resolve()
    if not models_dir.is_dir():
        raise ModelPackageError(f"Configured model directory does not exist: {models_dir}")
    if _package_matches_installed(package, models_dir):
        _ensure_model_license(
            package, models_dir, license_provider=license_provider
        )
        return "already_installed"
    manifest_path = models_dir / "manifest.json"
    if manifest_path.exists():
        raise ModelPackageError(
            "A different or invalid manifest.json already exists. Back up the model directory "
            "and remove the conflicting bundle before installing."
        )

    owner = models_dir.stat()
    bundle_name = f"{package.name}-{package.archive_sha256[:12]}"
    bundles_dir = models_dir / ".bundles"
    bundles_dir.mkdir(mode=0o755, exist_ok=True)
    destination = bundles_dir / bundle_name

    with tempfile.TemporaryDirectory(prefix=".install-", dir=models_dir) as temporary:
        temporary_dir = Path(temporary)
        archive_path = temporary_dir / f"{package.name}.zip"
        payload_dir = temporary_dir / "payload"
        payload_dir.mkdir()
        print(f"Downloading {package.name} from {package.url}")
        downloader(package, archive_path)
        print("Archive SHA-256 verified.")
        extract_required_models(package, archive_path, payload_dir)

        _write_text_fsync(
            payload_dir / MODEL_LICENSE_FILENAME, license_provider(package)
        )
        local_manifest = _manifest(package, Path("."))
        _write_json_fsync(payload_dir / "manifest.json", local_manifest)
        load_manifest(payload_dir)

        if destination.exists():
            destination_valid = destination.is_dir() and all(
                (destination / item.filename).is_file()
                and sha256_file(destination / item.filename) == item.sha256
                for item in package.files
            )
            if not destination_valid:
                raise ModelPackageError(f"Conflicting package directory already exists: {destination}")
        else:
            os.replace(payload_dir, destination)
        load_manifest(destination)

        relative_bundle = Path(".bundles") / bundle_name
        manifest_payload = _manifest(package, relative_bundle)
        temporary_manifest = models_dir / f".manifest-{uuid.uuid4().hex}.json"
        try:
            _write_json_fsync(temporary_manifest, manifest_payload)
            os.replace(temporary_manifest, manifest_path)
        finally:
            temporary_manifest.unlink(missing_ok=True)

    _restore_owner(bundles_dir, owner.st_uid, owner.st_gid)
    _restore_owner(manifest_path, owner.st_uid, owner.st_gid)
    manifest_path.chmod(0o644)
    for item in package.files:
        (destination / item.filename).chmod(0o644)
    (destination / MODEL_LICENSE_FILENAME).chmod(0o644)
    load_manifest(models_dir)
    return "installed"


def verify_installed(
    models_dir: Path,
) -> tuple[str | None, tuple[dict[str, object], ...], ModelLicense]:
    bundle = load_manifest(models_dir)
    installed = installed_package_name(models_dir)
    license_info = verify_model_license(
        bundle.license_path, expected_model_id=bundle.model_id
    )
    return (
        installed,
        tuple(model.public_summary() for model in bundle.models),
        license_info,
    )
