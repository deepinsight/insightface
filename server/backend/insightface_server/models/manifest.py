from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..licensing import LICENSE_FILENAME

DETECTION_TASK = "face_detection"
RECOGNITION_TASK = "face_recognition"
MANIFEST_VERSION = 1
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One ONNX component described by ``/models/manifest.json``.

    ``sha256`` is calculated from the active artifact for diagnostics and the
    Collection embedding contract. It is not compared with the manifest and is
    never part of model-license authorization.
    """

    model_id: str
    model_version: str
    task: str
    path: Path
    input_size: tuple[int, int]
    embedding_dimension: int | None
    preprocessing_version: str
    sha256: str
    input_mean: float
    input_std: float

    def public_summary(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "task": self.task,
            "file": self.path.name,
            "input_size": list(self.input_size),
            "embedding_dimension": self.embedding_dimension,
            "preprocessing_version": self.preprocessing_version,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ModelBundle:
    model_id: str
    model_version: str
    display_name: str
    license_path: Path
    detector: ModelSpec
    recognizer: ModelSpec
    legacy_manifest: bool = False

    @property
    def models(self) -> tuple[ModelSpec, ModelSpec]:
        return (self.detector, self.recognizer)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _required_string(raw: dict[str, Any], key: str, *, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Manifest field {key!r} must be a non-empty string in {context}")
    return value.strip()


def _safe_file(
    models_dir: Path, filename: str, *, suffix: str, must_exist: bool = True
) -> Path:
    root = models_dir.resolve()
    path = (root / filename).resolve()
    if root not in path.parents or path.suffix.lower() != suffix:
        raise RuntimeError(f"Unsafe {suffix} path in manifest: {filename}")
    if must_exist and not path.is_file():
        raise RuntimeError(f"Model package file does not exist: {path}")
    return path


def _positive_pair(value: object, *, field: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise RuntimeError(f"{field} must contain two positive integers")
    return value[0], value[1]


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{field} must be a positive integer")
    return value


def _new_manifest(raw: dict[str, Any], models_dir: Path) -> ModelBundle:
    allowed = {
        "manifest_version",
        "model_id",
        "model_version",
        "display_name",
        "files",
        "recognition",
        "license",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise RuntimeError(f"manifest.json contains unsupported fields: {', '.join(unknown)}")
    if raw.get("manifest_version") != MANIFEST_VERSION:
        raise RuntimeError(
            f"manifest_version must be {MANIFEST_VERSION}; found {raw.get('manifest_version')!r}"
        )
    model_id = _required_string(raw, "model_id", context="manifest.json")
    if not _MODEL_ID.fullmatch(model_id):
        raise RuntimeError("manifest model_id has an invalid format")
    model_version = _required_string(raw, "model_version", context="manifest.json")
    display = raw.get("display_name", model_id)
    if not isinstance(display, str) or not display.strip():
        raise RuntimeError("display_name must be a non-empty string when provided")

    files = raw.get("files")
    if not isinstance(files, dict) or set(files) != {"detector", "recognizer"}:
        raise RuntimeError("files must contain exactly detector and recognizer")
    detector_name = _required_string(files, "detector", context="files")
    recognizer_name = _required_string(files, "recognizer", context="files")
    detector_path = _safe_file(models_dir, detector_name, suffix=".onnx")
    recognizer_path = _safe_file(models_dir, recognizer_name, suffix=".onnx")

    recognition = raw.get("recognition")
    if not isinstance(recognition, dict):
        raise RuntimeError("recognition must be an object")
    if set(recognition) != {"input_size", "embedding_dimension", "preprocessing"}:
        raise RuntimeError(
            "recognition must contain exactly input_size, embedding_dimension, and preprocessing"
        )
    recognition_size = _positive_pair(
        recognition.get("input_size"), field="recognition.input_size"
    )
    if recognition_size[0] != recognition_size[1]:
        raise RuntimeError("Phase-one recognition input_size must be square")
    dimension = _positive_integer(
        recognition.get("embedding_dimension"), field="recognition.embedding_dimension"
    )
    preprocessing = _required_string(
        recognition, "preprocessing", context="recognition"
    )

    license_name = _required_string(raw, "license", context="manifest.json")
    if Path(license_name).name != LICENSE_FILENAME:
        raise RuntimeError(f"The model license filename must be {LICENSE_FILENAME}")
    license_path = _safe_file(
        models_dir, license_name, suffix=".license", must_exist=False
    )

    detector = ModelSpec(
        model_id=model_id,
        model_version=model_version,
        task=DETECTION_TASK,
        path=detector_path,
        # Active SCRFD resolutions are startup configuration, not manifest data.
        input_size=(640, 640),
        embedding_dimension=None,
        preprocessing_version="insightface-scrfd-1",
        sha256=sha256_file(detector_path),
        input_mean=127.5,
        input_std=128.0,
    )
    recognizer = ModelSpec(
        model_id=model_id,
        model_version=model_version,
        task=RECOGNITION_TASK,
        path=recognizer_path,
        input_size=recognition_size,
        embedding_dimension=dimension,
        preprocessing_version=preprocessing,
        sha256=sha256_file(recognizer_path),
        input_mean=127.5,
        input_std=127.5,
    )
    return ModelBundle(
        model_id=model_id,
        model_version=model_version,
        display_name=display.strip(),
        license_path=license_path,
        detector=detector,
        recognizer=recognizer,
    )


def _legacy_spec(raw: object, models_dir: Path) -> ModelSpec:
    if not isinstance(raw, dict):
        raise RuntimeError("Every models entry in legacy manifest.json must be an object")
    filename = _required_string(raw, "file", context="legacy models entry")
    task = _required_string(raw, "task", context=filename)
    if task not in {DETECTION_TASK, RECOGNITION_TASK}:
        raise RuntimeError(f"Unsupported model task {task!r}")
    path = _safe_file(models_dir, filename, suffix=".onnx")
    size = _positive_pair(raw.get("input_size"), field=f"input_size for {filename}")
    dimension_value = raw.get("embedding_dimension")
    dimension = (
        None
        if dimension_value is None
        else _positive_integer(dimension_value, field=f"embedding_dimension for {filename}")
    )
    default_std = 128.0 if task == DETECTION_TASK else 127.5
    try:
        input_mean = float(raw.get("input_mean", 127.5))
        input_std = float(raw.get("input_std", default_std))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid input_mean/input_std for {filename}") from exc
    if not math.isfinite(input_mean) or not math.isfinite(input_std) or input_std <= 0:
        raise RuntimeError(f"Invalid input_mean/input_std for {filename}")
    return ModelSpec(
        model_id=_required_string(raw, "model_id", context=filename),
        model_version=_required_string(raw, "model_version", context=filename),
        task=task,
        path=path,
        input_size=size,
        embedding_dimension=dimension,
        preprocessing_version=_required_string(
            raw, "preprocessing_version", context=filename
        ),
        # A legacy declared digest is intentionally not enforced. It is only
        # recalculated for diagnostics and Collection compatibility.
        sha256=sha256_file(path),
        input_mean=input_mean,
        input_std=input_std,
    )


def _legacy_manifest(raw: dict[str, Any], models_dir: Path) -> ModelBundle:
    entries = raw.get("models")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("manifest.json must use manifest_version 1")
    specs = [_legacy_spec(entry, models_dir) for entry in entries]
    by_task = {spec.task: spec for spec in specs}
    if len(specs) != 2 or len(by_task) != 2 or set(by_task) != {DETECTION_TASK, RECOGNITION_TASK}:
        raise RuntimeError(
            "Phase one requires exactly one face_detection and one face_recognition model"
        )
    recognizer = by_task[RECOGNITION_TASK]
    if recognizer.embedding_dimension is None:
        raise RuntimeError("face_recognition must declare embedding_dimension")
    package = raw.get("package")
    package_name = package.get("name") if isinstance(package, dict) else None
    model_id = package_name if isinstance(package_name, str) and package_name else recognizer.model_id
    parents = {spec.path.parent for spec in specs}
    if len(parents) != 1:
        raise RuntimeError("Legacy model files must share one package directory")
    license_path = parents.pop() / LICENSE_FILENAME
    return ModelBundle(
        model_id=model_id,
        model_version=(
            str(package.get("release"))
            if isinstance(package, dict) and package.get("release")
            else recognizer.model_version
        ),
        display_name=model_id,
        license_path=license_path,
        detector=by_task[DETECTION_TASK],
        recognizer=recognizer,
        legacy_manifest=True,
    )


def load_manifest(models_dir: str | Path) -> ModelBundle:
    """Load the current compact manifest, with read-only legacy migration support."""

    root = Path(models_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Required model manifest is missing: {manifest_path}")
    try:
        raw = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-standard JSON number: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Invalid model manifest: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("Model manifest root must be an object")
    if "manifest_version" in raw:
        return _new_manifest(raw, root)
    return _legacy_manifest(raw, root)
