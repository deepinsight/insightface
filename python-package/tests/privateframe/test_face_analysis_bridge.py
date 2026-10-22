from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
from insightface.app import face_analysis as face_analysis_module
from insightface.app.privateframe import models as private_models
from insightface.app.privateframe import recognition, revalidation, scan, streaming


class _Session:
    def __init__(self, providers: tuple[str, ...] = ("CPUExecutionProvider",)) -> None:
        self.providers = providers

    def get_providers(self) -> list[str]:
        return list(self.providers)


class _Recognizer:
    taskname = "recognition"
    model_file = "recognizer.onnx"
    input_shape: ClassVar[list[int]] = [1, 3, 112, 112]
    input_size = (112, 112)
    input_mean = 33.0
    input_std = 44.0

    def __init__(self) -> None:
        self.session = _Session()
        self.calls = 0

    def get_feat(self, _aligned: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.asarray([[3.0, 4.0]], dtype=np.float32)


class _TaskModel:
    def __init__(self, task: str) -> None:
        self.task = task
        self.taskname = task
        self.model_file = {
            "detection": "detector.onnx",
            "verification": "verifier.onnx",
        }[task]
        self.input_shape = [1, 3, 112, 112]
        self.session = _Session()
        if task == "detection":
            self.input_mean = 0.0
            self.input_std = 1.0
            self.nms_thresh = -1.0


class _FakeAnalysis:
    def __init__(self, *, include_recognizer: bool = True) -> None:
        self.detector = _TaskModel("detection")
        self.verifier = _TaskModel("verification")
        self.recognizer = _Recognizer()
        self.models: dict[str, Any] = {
            "detection": self.detector,
            "verification": self.verifier,
        }
        if include_recognizer:
            self.models["recognition"] = self.recognizer
        self.det_model = self.detector
        self.detector.detect = self._detect
        self.verifier.verify = self._verify
        self.detect_calls: list[dict[str, Any]] = []
        self.verify_calls: list[tuple[np.ndarray, list[list[float]]]] = []
        self.detections: list[dict[str, Any]] | None = None

    def _detect(
        self,
        image: np.ndarray,
        input_size: Any = None,
        max_num: int = 0,
        metric: str = "default",
        det_thresh: float | None = None,
    ) -> tuple[np.ndarray, None]:
        assert max_num == 0
        assert metric == "default"
        self.detect_calls.append(
            {
                "image": image,
                "input_sizes": tuple(input_size),
                "confidence_threshold": det_thresh,
            }
        )
        detections = self.detections
        if detections is None:
            detections = [
                {
                    "box": [0.0, 0.0, float(image.shape[1]), float(image.shape[0])],
                    "confidence": 0.9,
                }
            ]
        rows = [[*value["box"], float(value["confidence"])] for value in detections]
        return np.asarray(rows, dtype=np.float32), None

    def _verify(
        self,
        frame: np.ndarray,
        boxes: list[list[float]],
    ) -> list[dict[str, float]]:
        self.verify_calls.append((frame, boxes))
        return [{"face_probability": 0.75} for _box in boxes]


def _manifest_config(
    tmp_path: Path,
    *,
    mode: str,
    include_recognizer_file: bool = True,
    verifier_preprocessing: str | dict[str, float] = "embedded",
) -> dict[str, Any]:
    package = tmp_path / "raccoon_s"
    package.mkdir()
    contents = {
        "detection": b"detector-model",
        "verification": b"verifier-model",
        "recognition": b"recognizer-model",
    }
    manifest = {
        "manifest_version": 2,
        "model_id": "raccoon_s",
        "tasks": {
            "detection": {
                "file": "detector.onnx",
                "preprocessing": {"mean": 11.0, "std": 22.0},
            },
            "verification": {
                "file": "verifier.onnx",
                "expansion": 1.3,
                "preprocessing": verifier_preprocessing,
            },
            "recognition": {
                "file": "recognizer.onnx",
                "preprocessing": {"mean": 33.0, "std": 44.0},
            },
        },
        "license": "MODEL.LICENSE",
    }
    for task, content in contents.items():
        if task != "recognition" or include_recognizer_file:
            (package / manifest["tasks"][task]["file"]).write_bytes(content)
    manifest_path = package / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "models": {
            "name": "raccoon_s",
            "manifest_path": str(manifest_path),
            "detection": {
                "nms_iou_threshold": 0.37,
                "max_detections": 3,
            },
        },
        "runtime": {
            "providers": ["CPUExecutionProvider"],
            "intra_op_threads": 2,
            "inter_op_threads": 1,
        },
        "recognition": {"mode": mode},
    }


def _patch_standard_model_zoo(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], list[str]]:
    task_models: dict[str, Any] = {
        "detection": _TaskModel("detection"),
        "verification": _TaskModel("verification"),
        "recognition": _Recognizer(),
    }
    calls: list[str] = []

    def get_model(_name: Any, **kwargs: Any) -> Any:
        task = str(kwargs["model_task"])
        calls.append(task)
        return task_models[task]

    monkeypatch.setattr(face_analysis_module.model_zoo, "get_model", get_model)
    return task_models, calls


def test_make_face_analysis_uses_allowed_modules_as_the_loading_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_models, model_calls = _patch_standard_model_zoo(monkeypatch)
    constructor_calls: list[dict[str, Any]] = []
    real_face_analysis = private_models.FaceAnalysis

    def construct(*args: Any, **kwargs: Any) -> Any:
        assert args == ()
        constructor_calls.append(kwargs)
        return real_face_analysis(**kwargs)

    monkeypatch.setattr(private_models, "FaceAnalysis", construct)
    monkeypatch.setattr(
        face_analysis_module,
        "load_model_package",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("FaceAnalysis reparsed the supplied V2 descriptor")
        ),
    )
    config = _manifest_config(tmp_path, mode="blur_only")

    analysis = private_models.make_face_analysis(config)

    assert len(constructor_calls) == 1
    call = constructor_calls[0]
    package = call["name"]
    assert package.path == Path(config["models"]["manifest_path"]).parent
    assert package.manifest_version == 2
    assert analysis.model_package is package
    assert call["allowed_modules"] == (
        "detection",
        "verification",
        "recognition",
    )
    assert call["providers"] == ["CPUExecutionProvider"]
    assert call["static_shape_sessions"] is True
    assert set(call) == {
        "name",
        "allowed_modules",
        "providers",
        "sess_options",
        "static_shape_sessions",
    }
    assert model_calls == ["detection", "verification", "recognition"]
    assert analysis.models == {
        "detection": task_models["detection"],
        "verification": task_models["verification"],
        "recognition": task_models["recognition"],
    }
    assert task_models["detection"].nms_thresh == pytest.approx(0.37)
    assert dict(package.task("detection").metadata["preprocessing"]) == {
        "mean": 11.0,
        "std": 22.0,
    }
    assert package.task("verification").metadata["preprocessing"] == "embedded"


def test_verification_can_select_external_mean_std_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _task_models, model_calls = _patch_standard_model_zoo(monkeypatch)
    config = _manifest_config(
        tmp_path,
        mode="all",
        verifier_preprocessing={"mean": 0.5, "std": 0.25},
    )

    analysis = private_models.make_face_analysis(config)

    assert model_calls == ["detection", "verification"]
    assert dict(
        analysis.model_package.task("verification").metadata["preprocessing"]
    ) == {"mean": 0.5, "std": 0.25}


def test_mode_all_excludes_and_never_touches_missing_recognizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_models, model_calls = _patch_standard_model_zoo(monkeypatch)
    config = _manifest_config(
        tmp_path,
        mode="all",
        include_recognizer_file=False,
    )

    analysis = private_models.make_face_analysis(config)

    assert model_calls == ["detection", "verification"]
    assert analysis.models == {
        "detection": task_models["detection"],
        "verification": task_models["verification"],
    }
    assert "recognition" not in analysis.models


def test_face_analysis_loads_each_allowed_manifest_module_in_constructor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_models, model_calls = _patch_standard_model_zoo(monkeypatch)
    config = _manifest_config(
        tmp_path,
        mode="all",
        include_recognizer_file=False,
    )

    package = private_models._load_selected_model_package(config["models"])
    analysis = face_analysis_module.FaceAnalysis(
        name=package,
        allowed_modules=("detection", "verification"),
        providers=["CPUExecutionProvider"],
    )

    assert model_calls == ["detection", "verification"]
    assert analysis.models == {
        "detection": task_models["detection"],
        "verification": task_models["verification"],
    }


def test_privateframe_does_not_require_a_manifest_snapshot_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _task_models, model_calls = _patch_standard_model_zoo(monkeypatch)
    config = _manifest_config(tmp_path, mode="all")
    manifest_path = Path(config["models"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["future_release_annotation"] = {"build": 7}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    analysis = private_models.make_face_analysis(config)

    assert set(analysis.models) == {"detection", "verification"}
    assert model_calls == ["detection", "verification"]


def test_privateframe_package_name_fails_before_face_analysis_or_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _manifest_config(tmp_path, mode="all")
    config["models"]["name"] = "raccoon_l"
    constructor_calls: list[bool] = []
    monkeypatch.setattr(
        private_models,
        "FaceAnalysis",
        lambda **_kwargs: constructor_calls.append(True),
    )

    with pytest.raises(RuntimeError, match="model package name mismatch"):
        private_models.make_face_analysis(config)

    assert constructor_calls == []


def test_detect_faces_normalizes_sizes_stably_sorts_and_caps() -> None:
    analysis = _FakeAnalysis()
    analysis.detections = [
        {"box": [1, 1, 2, 2], "confidence": 0.5},
        {"box": [2, 2, 3, 3], "confidence": 0.9},
        {"box": [3, 3, 4, 4], "confidence": 0.5},
        {"box": [4, 4, 5, 5], "confidence": 0.7},
    ]
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    result = private_models.detect_faces(
        analysis,
        image,
        input_sizes=[128, (640, 320)],
        confidence_threshold=0.25,
        max_detections=3,
    )

    assert analysis.detect_calls == [
        {
            "image": image,
            "input_sizes": ((128, 128), (640, 320)),
            "confidence_threshold": 0.25,
        }
    ]
    assert [value["box"] for value in result] == [
        [2.0, 2.0, 3.0, 3.0],
        [4.0, 4.0, 5.0, 5.0],
        [1.0, 1.0, 2.0, 2.0],
    ]


def test_gallery_passes_maximum_and_two_square_scales() -> None:
    analysis = _FakeAnalysis()
    analysis.detections = [
        {"box": [0, 0, 4, 4], "confidence": 0.4},
        {"box": [0, 0, 5, 5], "confidence": 0.8},
        {"box": [0, 0, 6, 6], "confidence": 0.6},
    ]
    image = np.zeros((12, 16, 3), dtype=np.uint8)

    result = recognition.detect_gallery_faces_upright(
        analysis,
        image,
        max_detections=2,
    )

    assert analysis.detect_calls[-1]["input_sizes"] == (
        (640, 640),
        (128, 128),
    )
    assert [value["confidence"] for value in result] == pytest.approx([0.8, 0.6])


def test_legacy_arcface_get_feat_is_normalized_and_failures_close() -> None:
    aligned = np.zeros((112, 112, 3), dtype=np.uint8)
    recognizer = _Recognizer()

    embedding = recognition._embed_aligned_face(recognizer, aligned)

    np.testing.assert_allclose(
        embedding,
        np.asarray([0.6, 0.8], dtype=np.float32),
    )
    assert recognizer.calls == 1

    zero = _Recognizer()
    zero.get_feat = lambda _image: np.asarray([[0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-zero L2 norm"):
        recognition._embed_aligned_face(zero, aligned)

    malformed = _Recognizer()
    malformed.get_feat = lambda _image: np.asarray([3.0, 4.0], dtype=np.float32)
    with pytest.raises(RuntimeError, match="one non-empty embedding row"):
        recognition._embed_aligned_face(malformed, aligned)


def test_scan_and_local_review_forward_through_detect_faces() -> None:
    analysis = _FakeAnalysis()
    scan_config = {
        "models": {"detection": {"max_detections": 5}},
        "scan": {
            "passes": [
                {
                    "name": "primary",
                    "angles": [0],
                    "input_size": 128,
                    "confidence_threshold": 0.42,
                    "horizontal_padding_ratio": 0.0,
                    "vertical_padding_ratio": 0.0,
                }
            ],
            "session_sharing": "single_session_serial",
        },
    }
    runner = scan.ScanRunner(scan_config, face_analysis=analysis)
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    prepared: Future[tuple[np.ndarray, int, int]] = Future()
    prepared.set_result((frame, 0, 0))
    try:
        result = runner._run_view(prepared, runner.views[0], frame.shape)
    finally:
        runner.close()

    assert analysis.detect_calls[-1]["input_sizes"] == ((128, 128),)
    assert analysis.detect_calls[-1]["confidence_threshold"] == 0.42
    assert result[0]["detector"] == "detection"

    review_config = {
        "models": {"detection": {"max_detections": 5}},
        "revalidation": {
            "angles": [0],
            "confidence_threshold": 0.5,
            "input_size": 128,
            "crop_expansion": 1.0,
            "match_max_area_ratio": 5.0,
            "match_max_center_distance": 1.0,
            "match_min_iou": 0.1,
            "match_min_containment": 0.1,
            "edge_fallback": {"enabled": False},
        },
    }
    reviewer = revalidation.LocalReviewer(review_config, face_analysis=analysis)
    matched = reviewer.local_match(frame, [2.0, 2.0, 6.0, 6.0])
    scores = reviewer.verify(frame, [[2.0, 2.0, 6.0, 6.0]])

    assert matched["local_match_count"] == 1
    assert scores == [{"face_probability": 0.75}]
    assert analysis.verify_calls == [(frame, [[2.0, 2.0, 6.0, 6.0]])]


def _patch_streaming_init(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
) -> None:
    review_analysis = SimpleNamespace(models={"detection": object()})
    captured["review_analysis"] = review_analysis
    monkeypatch.setattr(
        streaming,
        "make_review_face_analysis",
        lambda _config: review_analysis,
    )
    monkeypatch.setattr(
        streaming,
        "probe_video",
        lambda _source: SimpleNamespace(fps=10.0, frame_count=100),
    )

    class Scanner:
        def __init__(
            self,
            config: Any,
            detector: Any = None,
            *,
            face_analysis: Any = None,
        ) -> None:
            captured["scanner"] = (detector, face_analysis)

        def close(self) -> None:
            captured["scanner_closed"] = True

    class Reviewer:
        def __init__(
            self,
            config: Any,
            detector: Any = None,
            verifier: Any = None,
            *,
            face_analysis: Any = None,
        ) -> None:
            captured["reviewer"] = (detector, verifier, face_analysis)

        def local_match(self, frame: Any, box: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "local_box": None,
                "local_landmarks": None,
                "local_match_count": 0,
                "local_confidence": None,
            }

        def verify(self, frame: Any, boxes: Any) -> list[dict[str, float]]:
            captured["review_verify"] = (frame, boxes)
            return [{"face_probability": 0.8}]

    class Cache:
        def __init__(self, path: Any, source: Any) -> None:
            captured["cache"] = (path, source)

        def close(self) -> None:
            captured["cache_closed"] = True

    monkeypatch.setattr(streaming, "ScanRunner", Scanner)
    monkeypatch.setattr(streaming, "LocalReviewer", Reviewer)
    monkeypatch.setattr(streaming, "EncodedPacketCache", Cache)
    monkeypatch.setattr(streaming, "SceneCutDetector", lambda _settings: object())


def _streaming_config(mode: str) -> dict[str, Any]:
    return {
        "tracking": {},
        "streaming": {
            "max_missed_seconds": 0.5,
            "max_retroactive_seconds": 1.0,
        },
        "scan": {"max_analysis_fps": 30},
        "recognition": {"mode": mode},
        "models": {"detection": {"max_detections": 7}},
        "runtime": {"providers": ["CPUExecutionProvider"]},
    }


@pytest.mark.parametrize("artifacts_level", [None, "final", "audit", "debug"])
def test_streaming_keeps_optional_consensus_audits_out_of_final_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifacts_level: str | None,
) -> None:
    analysis = _FakeAnalysis()
    _patch_streaming_init(monkeypatch, {})
    config = _streaming_config("all")
    config["tracking"]["kalman_optical_flow"] = {
        "bidirectional_fusion": {"max_gap_frames": 1}
    }
    if artifacts_level is not None:
        config["output"] = {"artifacts_level": artifacts_level}
    engine = streaming.StreamingEngine(
        tmp_path / "input.mp4",
        tmp_path,
        config,
        analysis.detector,
        face_analysis=analysis,
    )
    box = np.asarray([10.0, 10.0, 30.0, 30.0])
    pending = [{"frame_idx": 1}]
    state = SimpleNamespace(
        track={"track_id": "face"},
        last_detection_frame=0,
        last_consensus_anchor_box=box,
        last_detection_box=box,
        pending=pending,
    )

    accepted = engine._finish_pending_consensus(
        state,
        anchor_frame=3,
        right_consensus_anchor=box,
        right_geometry_anchor=box,
        right_geometry_source="detector",
    )

    # Optional diagnostics must not alter gap rejection or pending geometry.
    assert accepted is False
    assert engine.bidirectional_skipped_jobs == 1
    assert state.pending == [{"frame_idx": 1}]
    if artifacts_level in {"audit", "debug"}:
        assert len(engine.bidirectional_audits) == 1
        assert engine.bidirectional_audits[0]["reason"] == "gap_frame_limit"
    else:
        assert engine.bidirectional_audits == []


def test_streaming_selective_holds_underlying_get_feat_recognizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    analysis = _FakeAnalysis()
    _patch_streaming_init(monkeypatch, captured)

    def create_engine(
        settings: Any,
        *,
        recognizer: Any,
        gallery_detector: Any,
    ) -> Any:
        captured["recognizer"] = recognizer
        captured["gallery_detector"] = gallery_detector
        return SimpleNamespace(enabled=True, max_frames_per_track=3)

    monkeypatch.setattr(streaming, "create_recognition_engine", create_engine)
    detector = analysis.detector
    engine = streaming.StreamingEngine(
        tmp_path / "input.mp4",
        tmp_path,
        _streaming_config("blur_only"),
        detector,
        face_analysis=analysis,
    )

    assert captured["scanner"] == (detector, analysis)
    assert captured["reviewer"] == (
        None,
        analysis.verifier,
        captured["review_analysis"],
    )
    assert captured["recognizer"] is analysis.recognizer
    assert captured["recognizer"] is not analysis
    assert callable(captured["recognizer"].get_feat)
    assert captured["recognizer"].input_size == (112, 112)
    assert analysis.models["recognition"] is analysis.recognizer

    gallery_image = np.zeros((12, 16, 3), dtype=np.uint8)
    engine._gallery_faces(gallery_image)
    assert analysis.detect_calls[-1]["input_sizes"] == (
        (640, 640),
        (128, 128),
    )

    review_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    review = engine._measure_review(
        review_frame,
        {
            "source": "detector",
            "frame_idx": 0,
            "box": [1.0, 1.0, 5.0, 5.0],
        },
    )
    assert review["verifier_face_probability"] == 0.8
    engine.close()


def test_streaming_mode_all_never_requests_recognizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    analysis = _FakeAnalysis(include_recognizer=False)
    _patch_streaming_init(monkeypatch, captured)

    def create_engine(settings: Any, **kwargs: Any) -> Any:
        assert kwargs == {
            "recognizer": None,
            "gallery_detector": None,
        }
        return SimpleNamespace(enabled=False, max_frames_per_track=3)

    monkeypatch.setattr(streaming, "create_recognition_engine", create_engine)
    streaming.StreamingEngine(
        tmp_path / "input.mp4",
        tmp_path,
        _streaming_config("all"),
        analysis.detector,
        face_analysis=analysis,
    )

    assert "recognition" not in analysis.models


def test_streaming_derives_stride_and_endpoint_coverage_without_mutating_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    analysis = _FakeAnalysis(include_recognizer=False)
    _patch_streaming_init(monkeypatch, captured)
    monkeypatch.setattr(
        streaming,
        "probe_video",
        lambda _source: SimpleNamespace(fps=240.0, frame_count=2400),
    )
    monkeypatch.setattr(
        streaming,
        "create_recognition_engine",
        lambda *_args, **_kwargs: SimpleNamespace(
            enabled=False,
            max_frames_per_track=3,
        ),
    )
    config = _streaming_config("all")
    config["scan"]["max_analysis_fps"] = 15
    config["tracking"].update(
        {
            "endpoint_extension": 8,
            "reliable_endpoint_extension": 45,
        }
    )

    engine = streaming.StreamingEngine(
        tmp_path / "input.mp4",
        tmp_path,
        config,
        analysis.detector,
        face_analysis=analysis,
    )

    assert engine.detector_frame_stride == 16
    assert engine.nominal_regular_analysis_fps == pytest.approx(15.0)
    assert engine.interpolate_tracking is True
    assert engine.settings["endpoint_extension"] == 15
    assert engine.settings["reliable_endpoint_extension"] == 45
    assert config["scan"] == {"max_analysis_fps": 15}
    assert config["tracking"] == {
        "endpoint_extension": 8,
        "reliable_endpoint_extension": 45,
    }
    engine.close()


def test_run_stream_creates_one_analysis_and_preserves_detector_position(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    analysis = _FakeAnalysis()
    detector = analysis.detector

    def make_analysis(config: Any, **kwargs: Any) -> Any:
        captured["make_analysis"] = kwargs
        return analysis

    class Engine:
        def __init__(
            self,
            source: Any,
            workdir: Any,
            config: Any,
            positional_detector: Any,
            *,
            face_analysis: Any,
        ) -> None:
            captured["engine"] = (positional_detector, face_analysis)

        def run(self) -> dict[str, bool]:
            return {"ok": True}

        def _clear_recognition_candidates(self) -> None:
            captured["cleared"] = True

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(streaming, "make_face_analysis", make_analysis)
    monkeypatch.setattr(streaming, "StreamingEngine", Engine)

    result = streaming.run_stream(
        tmp_path / "input.mp4",
        tmp_path,
        _streaming_config("all"),
        detector,
    )

    assert result == {"ok": True}
    assert captured["make_analysis"] == {}
    assert captured["engine"] == (detector, analysis)
    assert captured["cleared"] is True
    assert captured["closed"] is True
