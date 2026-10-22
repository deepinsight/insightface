from __future__ import annotations

import json
from pathlib import Path

import pytest

from insightface.app import face_analysis
from insightface.app.face_analysis import FaceAnalysis
from insightface.model_zoo.package_manifest import (
    has_model_package_manifest,
    load_model_package,
)


class _LegacyModel:
    input_shape = [1, 3, 112, 112]
    input_mean = 127.5
    input_std = 127.5

    def __init__(self, taskname: str):
        self.taskname = taskname


def _write_versioned_package(tmp_path: Path, version: int | None) -> Path:
    package = tmp_path / f"package-{version}"
    package.mkdir()
    detector = package / "a_detector.onnx"
    recognizer = package / "b_recognizer.onnx"
    detector.write_bytes(b"detector")
    recognizer.write_bytes(b"recognizer")
    manifest = {
        "model_id": "buffalo_l",
        "files": {
            "detector": detector.name,
            "recognizer": recognizer.name,
        },
    }
    if version is not None:
        manifest["manifest_version"] = version
    (package / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return package


@pytest.mark.parametrize("version", [None, 1, 3])
def test_face_analysis_ignores_non_v2_manifest_and_shape_routes_onnx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: int | None,
) -> None:
    package = _write_versioned_package(tmp_path, version)
    calls: list[tuple[str, dict]] = []

    def get_model(path, **kwargs):
        filename = Path(path).name
        calls.append((filename, kwargs))
        task = "detection" if filename.startswith("a_") else "recognition"
        return _LegacyModel(task)

    monkeypatch.setattr(face_analysis.model_zoo, "get_model", get_model)
    monkeypatch.setattr(
        face_analysis,
        "load_model_package",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-V2 manifest entered descriptor routing")
        ),
    )

    analysis = FaceAnalysis(package)

    assert has_model_package_manifest(package) is False
    assert analysis.model_package is None
    assert tuple(analysis.models) == ("detection", "recognition")
    assert [name for name, _kwargs in calls] == [
        "a_detector.onnx",
        "b_recognizer.onnx",
    ]
    assert all("model_task" not in kwargs for _name, kwargs in calls)


def test_load_model_package_itself_rejects_server_v1(tmp_path: Path) -> None:
    package = _write_versioned_package(tmp_path, 1)

    with pytest.raises(ValueError, match="manifest_version must be 2"):
        load_model_package(package)


def test_explicit_but_damaged_v2_fails_without_shape_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "broken-v2"
    package.mkdir()
    (package / "detector.onnx").write_bytes(b"detector")
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 2,
                "model_id": "broken_v2",
                "tasks": {"detection": {}},
                "license": "MODEL.LICENSE",
            }
        ),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        face_analysis.model_zoo,
        "get_model",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(ValueError, match=r"tasks\.detection\.file is required"):
        FaceAnalysis(package)
    assert calls == []
