"""Descriptors for the optional task-aware model-package manifest V2.

Only an explicit ``manifest_version: 2`` activates manifest routing. Older,
unversioned, V1, and unknown-version documents are intentionally ignored by
FaceAnalysis so their ONNX files retain the historical shape-based routing.
Once a package explicitly opts into V2, malformed known fields fail before any
inference Session is created. A task may also declare its ONNX SHA-256; when it
does, the selected artifact is verified immediately before Session creation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping


MODEL_PACKAGE_MANIFEST = "manifest.json"
MODEL_LICENSE_FILENAME = "MODEL.LICENSE"
MODEL_PACKAGE_MANIFEST_VERSION = 2
EMBEDDED_PREPROCESSING = "embedded"
DETECTION_TASK = "detection"
VERIFICATION_TASK = "verification"
RECOGNITION_TASK = "recognition"
MODEL_PACKAGE_TASKS = (
    DETECTION_TASK,
    VERIFICATION_TASK,
    RECOGNITION_TASK,
)

# Package selection/download remains intentionally narrower than the generic
# V2 schema parser.
SUPPORTED_MANIFEST_PACKAGES = ("raccoon_s", "raccoon_l")

_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROOT_REQUIRED_FIELDS = {
    "manifest_version",
    "model_id",
    "tasks",
    "license",
}
_DEFAULT_PREPROCESSING = {
    DETECTION_TASK: {"mean": 127.5, "std": 128.0},
    VERIFICATION_TASK: EMBEDDED_PREPROCESSING,
    RECOGNITION_TASK: {"mean": 127.5, "std": 127.5},
}


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _model_id(value: Any) -> str:
    result = _required_string(value, "model_id")
    if not _MODEL_ID.fullmatch(result):
        raise ValueError("model_id has an invalid format")
    return result


def _expected_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not _SHA256.fullmatch(value):
        raise ValueError(
            f"{field} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        requirement = "finite and positive" if positive else "finite"
        raise ValueError(f"{field} must be {requirement}")
    return result


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _positive_pair(value: Any, field: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{field} must contain two positive integers")
    result = (
        _positive_integer(value[0], f"{field}[0]"),
        _positive_integer(value[1], f"{field}[1]"),
    )
    if result[0] != result[1]:
        raise ValueError(f"{field} must be square")
    return result


def normalize_preprocessing(value: Any, field: str = "preprocessing") -> Any:
    """Validate and freeze embedded or scalar mean/std normalization."""

    if isinstance(value, str):
        if value != EMBEDDED_PREPROCESSING:
            raise ValueError(f'{field} must be "embedded" or a mean/std object')
        return EMBEDDED_PREPROCESSING
    if not isinstance(value, Mapping):
        raise TypeError(f'{field} must be "embedded" or a mean/std object')
    if set(value) != {"mean", "std"}:
        raise ValueError(f"{field} keys must be exactly ['mean', 'std']")
    return MappingProxyType(
        {
            "mean": _finite_number(value["mean"], f"{field}.mean"),
            "std": _finite_number(
                value["std"],
                f"{field}.std",
                positive=True,
            ),
        }
    )


def _safe_package_path(
    package_path: Path,
    value: Any,
    *,
    field: str,
    suffix: str,
    basename: str | None = None,
) -> tuple[str, Path]:
    filename = _required_string(value, field)
    if "\\" in filename or "\x00" in filename:
        raise ValueError(f"{field} must use a safe relative POSIX path")
    relative = PurePosixPath(filename)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or any(":" in part for part in relative.parts)
    ):
        raise ValueError(f"{field} escapes the package or is not a safe relative path")
    if basename is not None and relative.name != basename:
        raise ValueError(f"{field} must reference {basename}")
    if relative.suffix.lower() != suffix.lower():
        raise ValueError(f"{field} must end in {suffix}")
    resolved = (package_path / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(package_path)
    except ValueError as error:
        raise ValueError(f"{field} escapes the package directory") from error
    return filename, resolved


def _model_path(package_path: Path, task: str, value: Any) -> tuple[str, Path]:
    return _safe_package_path(
        package_path,
        value,
        field=f"tasks.{task}.file",
        suffix=".onnx",
    )


def _license_path(package_path: Path, value: Any) -> Path:
    _filename, path = _safe_package_path(
        package_path,
        value,
        field="license",
        suffix=".license",
        basename=MODEL_LICENSE_FILENAME,
    )
    return path


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ModelTaskDescriptor:
    """One known V2 task and its optional artifact digest."""

    task: str
    file: str
    path: Path
    package_path: Path
    metadata: Mapping[str, Any]
    sha256: str | None = None

    def as_config(self) -> dict[str, Any]:
        config = {
            "file": self.file,
            "path": str(self.path),
            **_thaw(self.metadata),
        }
        if self.sha256 is not None:
            config["sha256"] = self.sha256
        return config


@dataclass(frozen=True)
class ModelPackageDescriptor:
    """A parsed V2 package whose selected ONNX files remain unopened."""

    name: str
    path: Path
    manifest_path: Path
    tasks: Mapping[str, ModelTaskDescriptor]
    manifest_version: int = MODEL_PACKAGE_MANIFEST_VERSION
    display_name: str | None = None
    license_path: Path | None = None
    source_schema: str = "unified-v2"

    @property
    def model_id(self) -> str:
        return self.name

    def task(self, name: str) -> ModelTaskDescriptor:
        try:
            return self.tasks[str(name)]
        except KeyError as error:
            raise ValueError(f"model_task must be one of {list(self.tasks)}") from error

    def verify_task(self, name: str) -> ModelTaskDescriptor:
        descriptor = self.task(name)
        verify_model_artifact(descriptor)
        return descriptor


def _manifest_path(path: str | Path) -> tuple[Path, Path]:
    package_path = Path(path).expanduser().resolve()
    manifest_path = (package_path / MODEL_PACKAGE_MANIFEST).resolve()
    try:
        manifest_path.relative_to(package_path)
    except ValueError as error:
        raise ValueError(
            "model package manifest escapes the package directory"
        ) from error
    return package_path, manifest_path


def _read_document(manifest_path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid model package manifest: {manifest_path}") from error
    if not isinstance(raw, Mapping):
        raise TypeError("model package manifest must be an object")
    return raw


def model_package_manifest_version(path: str | Path) -> int | None:
    """Return a readable manifest's integer version without validating it."""

    package_path = Path(path).expanduser()
    manifest_path = package_path / MODEL_PACKAGE_MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, Mapping):
        return None
    version = raw.get("manifest_version")
    return version if type(version) is int else None


def has_model_package_manifest(path: str | Path) -> bool:
    """Return whether a directory explicitly opts into manifest V2 routing."""

    return model_package_manifest_version(path) == MODEL_PACKAGE_MANIFEST_VERSION


def _task_metadata(task: str, raw: Mapping[str, Any]) -> Mapping[str, Any]:
    preprocessing = normalize_preprocessing(
        raw.get("preprocessing", _DEFAULT_PREPROCESSING[task]),
        f"tasks.{task}.preprocessing",
    )
    values: dict[str, Any] = {"preprocessing": preprocessing}
    if task == DETECTION_TASK:
        values["preprocessing_version"] = _required_string(
            raw.get("preprocessing_version", "insightface-scrfd-1"),
            "tasks.detection.preprocessing_version",
        )
    elif task == VERIFICATION_TASK:
        values["expansion"] = _finite_number(
            raw.get("expansion", 1.3),
            "tasks.verification.expansion",
            positive=True,
        )
    else:
        values["preprocessing_version"] = _required_string(
            raw.get("preprocessing_version", "insightface-arcface-1"),
            "tasks.recognition.preprocessing_version",
        )
        values["input_size"] = _positive_pair(
            raw.get("input_size", [112, 112]),
            "tasks.recognition.input_size",
        )
        values["embedding_dimension"] = _positive_integer(
            raw.get("embedding_dimension", 512),
            "tasks.recognition.embedding_dimension",
        )
    return MappingProxyType(values)


def load_model_package(path: str | Path) -> ModelPackageDescriptor:
    """Parse an explicit model-package manifest V2."""

    package_path, manifest_path = _manifest_path(path)
    if not package_path.is_dir():
        raise FileNotFoundError(
            f"model package directory does not exist: {package_path}"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"model package manifest does not exist: {manifest_path}"
        )
    raw = _read_document(manifest_path)
    version = raw.get("manifest_version")
    if type(version) is not int or version != MODEL_PACKAGE_MANIFEST_VERSION:
        raise ValueError(
            f"manifest_version must be {MODEL_PACKAGE_MANIFEST_VERSION}; "
            f"received {version!r}"
        )
    missing = sorted(_ROOT_REQUIRED_FIELDS - set(raw))
    if missing:
        raise ValueError(
            f"model package manifest is missing required fields: {missing}"
        )

    package_name = _model_id(raw["model_id"])
    display_name = _required_string(
        raw.get("display_name", package_name),
        "display_name",
    )
    license_path = _license_path(package_path, raw["license"])
    raw_tasks = raw["tasks"]
    if not isinstance(raw_tasks, Mapping):
        raise TypeError("tasks must be an object")

    tasks: dict[str, ModelTaskDescriptor] = {}
    for task in MODEL_PACKAGE_TASKS:
        if task not in raw_tasks:
            continue
        task_raw = raw_tasks[task]
        if not isinstance(task_raw, Mapping):
            raise TypeError(f"tasks.{task} must be an object")
        if "file" not in task_raw:
            raise ValueError(f"tasks.{task}.file is required")
        filename, model_path = _model_path(package_path, task, task_raw["file"])
        expected_sha256 = (
            _expected_sha256(task_raw["sha256"], f"tasks.{task}.sha256")
            if "sha256" in task_raw
            else None
        )
        tasks[task] = ModelTaskDescriptor(
            task=task,
            file=filename,
            path=model_path,
            package_path=package_path,
            metadata=_task_metadata(task, task_raw),
            sha256=expected_sha256,
        )

    paths = [descriptor.path for descriptor in tasks.values()]
    if len(set(paths)) != len(paths):
        raise ValueError("known model package tasks must reference distinct files")
    return ModelPackageDescriptor(
        name=package_name,
        path=package_path,
        manifest_path=manifest_path,
        tasks=MappingProxyType(tasks),
        display_name=display_name,
        license_path=license_path,
    )


def verify_model_artifact(descriptor: ModelTaskDescriptor) -> Path:
    """Resolve and, when declared, verify a task before Session construction."""

    path = descriptor.path.resolve()
    try:
        path.relative_to(descriptor.package_path)
    except ValueError as error:
        raise ValueError(
            f"{descriptor.task}.file escapes its package directory"
        ) from error
    if not path.is_file():
        raise FileNotFoundError(f"model file does not exist: {path}")
    if descriptor.sha256 is not None:
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != descriptor.sha256:
            raise RuntimeError(
                f"model SHA-256 mismatch for {path}: expected "
                f"{descriptor.sha256}, got {actual_sha256}"
            )
    return path


load_package_manifest = load_model_package
ModelPackage = ModelPackageDescriptor
ModelArtifact = ModelTaskDescriptor


__all__ = [
    "DETECTION_TASK",
    "EMBEDDED_PREPROCESSING",
    "MODEL_LICENSE_FILENAME",
    "MODEL_PACKAGE_MANIFEST",
    "MODEL_PACKAGE_MANIFEST_VERSION",
    "MODEL_PACKAGE_TASKS",
    "RECOGNITION_TASK",
    "SUPPORTED_MANIFEST_PACKAGES",
    "VERIFICATION_TASK",
    "ModelArtifact",
    "ModelPackage",
    "ModelPackageDescriptor",
    "ModelTaskDescriptor",
    "has_model_package_manifest",
    "load_model_package",
    "load_package_manifest",
    "model_package_manifest_version",
    "normalize_preprocessing",
    "verify_model_artifact",
]
