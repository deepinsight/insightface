from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from insightface.app.privateframe import models as private_models


class _SessionOptions:
    def __init__(self) -> None:
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0
        self.log_severity_level = 0


class _Session:
    def __init__(self, providers: tuple[str, ...]) -> None:
        self._providers = providers

    def get_providers(self) -> list[str]:
        return list(self._providers)


class _Model:
    def __init__(self, providers: tuple[str, ...]) -> None:
        self.session = _Session(providers)
        self.nms_thresh = -1.0


class _Analysis:
    def __init__(self, providers: tuple[str, ...]) -> None:
        self.models = {
            "detection": _Model(providers),
            "verification": _Model(providers),
        }
        self.det_model = self.models["detection"]


def _runtime(*providers: str) -> dict[str, Any]:
    return {
        "providers": list(providers),
        "intra_op_threads": 3,
        "inter_op_threads": 2,
    }


def _config(tmp_path: Path, runtime: dict[str, Any]) -> dict[str, Any]:
    package_path = tmp_path / "raccoon_s"
    package_path.mkdir()
    contents = {
        "detection": ("detector.onnx", b"detector"),
        "verification": ("verifier.onnx", b"verifier"),
        "recognition": ("recognizer.onnx", b"recognizer"),
    }
    manifest = {
        "manifest_version": 2,
        "model_id": "raccoon_s",
        "tasks": {
            "detection": {
                "file": "detector.onnx",
                "preprocessing": {"mean": 127.5, "std": 128.0},
            },
            "verification": {
                "file": "verifier.onnx",
                "expansion": 1.3,
                "preprocessing": "embedded",
            },
            "recognition": {
                "file": "recognizer.onnx",
                "preprocessing": {"mean": 127.5, "std": 127.5},
            },
        },
        "license": "MODEL.LICENSE",
    }
    for filename, content in contents.values():
        (package_path / filename).write_bytes(content)
    manifest_path = package_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "models": {
            "name": "raccoon_s",
            "manifest_path": str(manifest_path),
            "detection": {
                "nms_iou_threshold": 0.42,
                "max_detections": 5,
            },
        },
        "runtime": runtime,
        "recognition": {"mode": "all"},
    }


def test_make_face_analysis_forwards_session_options_and_provider_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[dict[str, Any]] = []
    analysis = _Analysis(("CPUExecutionProvider",))

    def construct(**kwargs: Any) -> _Analysis:
        captured.append(kwargs)
        return analysis

    monkeypatch.setattr(private_models.ort, "SessionOptions", _SessionOptions)
    monkeypatch.setattr(private_models, "FaceAnalysis", construct)

    result = private_models.make_face_analysis(
        _config(tmp_path, _runtime("CPUExecutionProvider"))
    )

    assert result is analysis
    assert len(captured) == 1
    call = captured[0]
    assert call["providers"] == ["CPUExecutionProvider"]
    assert call["allowed_modules"] == ("detection", "verification")
    assert call["static_shape_sessions"] is True
    assert set(call) == {
        "name",
        "allowed_modules",
        "providers",
        "sess_options",
        "static_shape_sessions",
    }
    options = call["sess_options"]
    assert options.intra_op_num_threads == 3
    assert options.inter_op_num_threads == 2
    assert options.log_severity_level == 3
    assert analysis.models["detection"].nms_thresh == pytest.approx(0.42)


def test_cuda_primary_provider_silent_fallback_fails_closed() -> None:
    analysis = _Analysis(("CPUExecutionProvider",))

    with pytest.raises(
        RuntimeError,
        match=r"CUDAExecutionProvider.*activated \['CPUExecutionProvider'\]",
    ):
        private_models._validate_primary_provider(
            analysis.models["detection"],
            _runtime("CUDAExecutionProvider", "CPUExecutionProvider"),
        )
