"""Minimal manifest-driven model-pack discovery."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from ...model_zoo.package_manifest import (
    DETECTION_TASK,
    EMBEDDED_PREPROCESSING,
    MODEL_PACKAGE_MANIFEST,
    MODEL_PACKAGE_TASKS,
    RECOGNITION_TASK,
    SUPPORTED_MANIFEST_PACKAGES,
    VERIFICATION_TASK,
    load_model_package,
    normalize_preprocessing,
)
from ...utils import ensure_available

SUPPORTED_MODEL_PACKAGES = SUPPORTED_MANIFEST_PACKAGES

DEFAULT_INSIGHTFACE_ROOT = "~/.insightface"

_MODEL_CONFIG_KEYS = {"name", "root", DETECTION_TASK}
_DETECTION_TUNABLE_KEYS = {
    "nms_iou_threshold",
    "max_detections",
}
_TASK_CONFIG_KEYS = {
    DETECTION_TASK: {
        "file",
        "path",
        "sha256",
        "preprocessing",
        "preprocessing_version",
    },
    VERIFICATION_TASK: {
        "file",
        "path",
        "sha256",
        "preprocessing",
        "expansion",
    },
    RECOGNITION_TASK: {
        "file",
        "path",
        "sha256",
        "preprocessing",
        "preprocessing_version",
        "input_size",
        "embedding_dimension",
    },
}


def validate_model_package_selection(config: Mapping[str, Any]) -> str:
    """Validate the public model-package selector without resolving downloads."""

    models = config.get("models")
    if not isinstance(models, Mapping):
        raise TypeError("models configuration must be a mapping")
    extra = set(models) - _MODEL_CONFIG_KEYS
    if extra:
        raise ValueError(f"models contains unsupported keys: {sorted(extra)}")
    package_name = models.get("name")
    if not isinstance(package_name, str):
        raise TypeError("models.name must be a string")
    package_name = package_name.strip()
    if package_name not in SUPPORTED_MODEL_PACKAGES:
        raise ValueError("models.name must be raccoon_s or raccoon_l")

    root = models.get("root", DEFAULT_INSIGHTFACE_ROOT)
    if not isinstance(root, str):
        raise TypeError("models.root must be a string")
    if not root.strip():
        raise ValueError("models.root must be a non-empty path")

    detection = models.get(DETECTION_TASK, {})
    if not isinstance(detection, Mapping):
        raise TypeError("models.detection must be a mapping")
    detection_extra = set(detection) - _DETECTION_TUNABLE_KEYS
    if detection_extra:
        raise ValueError(
            "models.detection contains unsupported keys: " f"{sorted(detection_extra)}"
        )
    if "nms_iou_threshold" in detection:
        threshold = detection["nms_iou_threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("models.detection.nms_iou_threshold must be a number")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(
                "models.detection.nms_iou_threshold must be between 0 and 1"
            )
    if "max_detections" in detection:
        maximum = detection["max_detections"]
        if isinstance(maximum, bool) or not isinstance(maximum, int):
            raise TypeError("models.detection.max_detections must be an integer")
        if maximum < 1:
            raise ValueError("models.detection.max_detections must be positive")
    return package_name


def _required_tasks(config: Mapping[str, Any]) -> tuple[str, ...]:
    recognition = config.get("recognition", {})
    mode = (
        str(recognition.get("mode", "all"))
        if isinstance(recognition, Mapping)
        else "all"
    )
    tasks = [DETECTION_TASK, VERIFICATION_TASK]
    if mode != "all":
        tasks.append(RECOGNITION_TASK)
    return tuple(tasks)


def _task_config(package: Any, task: str) -> dict[str, Any]:
    descriptor = package.task(task)
    declared = descriptor.as_config()
    # V2 metadata may grow without turning an unrelated package annotation into
    # a PrivateFrame configuration error. Keep only values this runtime uses.
    return {
        key: deepcopy(value)
        for key, value in declared.items()
        if key in _TASK_CONFIG_KEYS[task]
    }


def _materialize_package_models(
    config: dict[str, Any],
) -> None:
    package_name = validate_model_package_selection(config)
    models = config["models"]
    model_root = str(models.get("root", DEFAULT_INSIGHTFACE_ROOT))
    package_path = ensure_available(
        "models",
        package_name,
        root=model_root,
    )
    package = load_model_package(package_path)
    if package.name != package_name:
        raise RuntimeError(
            f"resolved model package name {package.name} does not match {package_name}"
        )
    required_tasks = _required_tasks(config)
    missing = [task for task in required_tasks if task not in package.tasks]
    if missing:
        raise ValueError(
            f"PrivateFrame model package {package.name!r} is missing required "
            f"task(s): {', '.join(missing)}"
        )
    declared_tasks = {
        task: _task_config(package, task)
        for task in required_tasks
    }

    detection_settings = models.get(DETECTION_TASK, {})
    tunable_detection_settings = deepcopy(dict(detection_settings))
    declared_tasks[DETECTION_TASK].update(tunable_detection_settings)

    models.clear()
    models.update(
        {
            "name": package.name,
            "root": model_root,
            "manifest_path": str(package.manifest_path),
            **declared_tasks,
        }
    )


def materialize_model_package(config: dict[str, Any]) -> None:
    """Resolve one named raccoon package from its configured InsightFace root."""

    _materialize_package_models(config)


def declared_model_sha256(
    model: Mapping[str, Any],
    *,
    field: str = "model.sha256",
) -> str | None:
    """Return and validate an optional manifest-declared model digest."""

    if "sha256" not in model:
        return None
    value = model["sha256"]
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            f"{field} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def verify_model_file(model: Mapping[str, Any]) -> Path:
    """Return an existing model path without hashing it again.

    ModelZoo verifies a declared digest immediately before Session creation.
    This lightweight check is used while assembling result metadata, after the
    model has already passed that inference-boundary verification.
    """

    declared_model_sha256(model)
    path = Path(str(model["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    return path


__all__ = [
    "DEFAULT_INSIGHTFACE_ROOT",
    "DETECTION_TASK",
    "EMBEDDED_PREPROCESSING",
    "MODEL_PACKAGE_MANIFEST",
    "MODEL_PACKAGE_TASKS",
    "RECOGNITION_TASK",
    "SUPPORTED_MODEL_PACKAGES",
    "VERIFICATION_TASK",
    "declared_model_sha256",
    "materialize_model_package",
    "normalize_preprocessing",
    "validate_model_package_selection",
    "verify_model_file",
]
