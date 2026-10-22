import json
from types import SimpleNamespace

import numpy as np
import pytest

from insightface.gui.core import face_engine


class _Model:
    def __init__(self, taskname, output_shape=None):
        self.taskname = taskname
        self.output_shape = output_shape


class _Analysis:
    def __init__(self, models, faces=()):
        self.models = models
        self.det_model = models["detection"]
        self.faces = list(faces)
        self.prepare_calls = []
        self.get_calls = []

    def prepare(self, ctx_id, **kwargs):
        self.prepare_calls.append((ctx_id, kwargs))

    def get(self, image, **kwargs):
        self.get_calls.append((image, kwargs))
        return list(self.faces)


def _write_v2_manifest(model_dir):
    model_dir.mkdir(parents=True)
    (model_dir / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 2,
                "model_id": "raccoon_s",
                "license": "MODEL.LICENSE",
                "tasks": {
                    "detection": {"file": "detector.onnx"},
                    "verification": {"file": "verifier.onnx"},
                    "recognition": {"file": "recognizer.onnx"},
                },
            }
        ),
        encoding="utf-8",
    )


def test_v2_package_loads_only_gui_tasks_through_face_analysis(
    monkeypatch,
    tmp_path,
):
    model_dir = tmp_path / "models" / "raccoon_s"
    _write_v2_manifest(model_dir)
    analysis = _Analysis(
        {
            "detection": _Model("detection"),
            "recognition": _Model("recognition", [None, 512]),
        }
    )
    calls = []

    def make_analysis(**kwargs):
        calls.append(kwargs)
        return analysis

    monkeypatch.setattr(face_engine, "FaceAnalysis", make_analysis)
    engine = face_engine.FaceEngine(
        model_name="raccoon_s",
        root=tmp_path,
        providers=["CPUExecutionProvider"],
        det_size=(0, 0),
    )

    engine.load()

    assert engine.is_loaded() is True
    assert engine.face_analysis is analysis
    assert engine.models is analysis.models
    assert engine.det_model is analysis.det_model
    assert engine.embedding_dim == 512
    assert calls[0]["name"] == model_dir.resolve()
    assert calls[0]["allowed_modules"] == ("detection", "recognition")
    assert calls[0]["providers"] == ["CPUExecutionProvider"]
    assert "_coreml_detector_input_size" not in calls[0]
    assert analysis.prepare_calls == [
        (
            -1,
            {
                "det_size": [(128, 128), (640, 640)],
                "det_thresh": 0.5,
            },
        )
    ]


def test_legacy_custom_directory_keeps_all_tasks_and_fixed_coreml_size(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "detector.onnx").write_bytes(b"fake")
    analysis = _Analysis(
        {
            "detection": _Model("detection"),
            "recognition": _Model("recognition", [None, 512]),
            "landmark_3d_68": _Model("landmark_3d_68"),
            "genderage": _Model("genderage"),
        }
    )
    calls = []

    def make_analysis(**kwargs):
        calls.append(kwargs)
        return analysis

    monkeypatch.setattr(face_engine, "FaceAnalysis", make_analysis)
    providers = [
        ("CoreMLExecutionProvider", {"MLComputeUnits": "ALL"}),
        "CPUExecutionProvider",
    ]
    engine = face_engine.FaceEngine(
        model_name="raccoon_s",
        custom_model_dir=tmp_path,
        providers=providers,
        det_size=(320, 320),
    )

    engine.load()

    assert engine.is_loaded() is True
    assert set(engine.models) == {
        "detection",
        "recognition",
        "landmark_3d_68",
        "genderage",
    }
    assert calls[0]["name"] == tmp_path.resolve()
    assert calls[0]["allowed_modules"] is None
    assert calls[0]["providers"] == providers
    assert calls[0]["static_shape_sessions"] is True
    assert calls[0]["_coreml_detector_input_size"] == (320, 320)
    assert analysis.prepare_calls == [(0, {"det_size": (320, 320), "det_thresh": 0.5})]


@pytest.mark.parametrize(
    "manifest_text",
    [
        None,
        "{not-json",
        json.dumps(
            {
                "manifest_version": 2,
                "model_id": "raccoon_l",
                "license": "MODEL.LICENSE",
                "tasks": {
                    "detection": {"file": "detector.onnx"},
                    "recognition": {"file": "recognizer.onnx"},
                },
            }
        ),
        json.dumps(
            {
                "manifest_version": 2,
                "model_id": "raccoon_s",
                "license": "MODEL.LICENSE",
                "tasks": {"detection": {"file": "detector.onnx"}},
            }
        ),
    ],
    ids=("missing", "malformed", "wrong-model-id", "missing-recognition"),
)
def test_official_raccoon_package_requires_valid_v2_manifest(
    monkeypatch,
    tmp_path,
    manifest_text,
):
    model_dir = tmp_path / "models" / "raccoon_s"
    model_dir.mkdir(parents=True)
    (model_dir / "detector.onnx").write_bytes(b"legacy-shaped")
    if manifest_text is not None:
        (model_dir / "manifest.json").write_text(
            manifest_text,
            encoding="utf-8",
        )
    calls = []
    monkeypatch.setattr(
        face_engine,
        "FaceAnalysis",
        lambda **kwargs: calls.append(kwargs),
    )
    engine = face_engine.FaceEngine(
        model_name="raccoon_s",
        root=tmp_path,
        providers=["CPUExecutionProvider"],
    )

    engine.load()

    assert engine.is_loaded() is False
    assert engine.face_analysis is None
    assert calls == []
    assert "requires a valid V2 manifest" in engine.last_error


def test_missing_local_directory_never_constructs_face_analysis(
    monkeypatch,
    tmp_path,
):
    calls = []

    def make_analysis(**kwargs):
        calls.append(kwargs)
        raise AssertionError("FaceAnalysis must not be constructed")

    monkeypatch.setattr(face_engine, "FaceAnalysis", make_analysis)
    engine = face_engine.FaceEngine(
        model_name="not-installed",
        root=tmp_path,
        providers=["CPUExecutionProvider"],
    )

    engine.load()

    assert engine.is_loaded() is False
    assert engine.face_analysis is None
    assert calls == []
    assert "Model directory not found" in engine.last_error


def test_detect_faces_converts_face_analysis_results_to_face_records(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "detector.onnx").write_bytes(b"fake")
    face = SimpleNamespace(
        bbox=np.array([1.0, 2.0, 5.0, 7.0], dtype=np.float32),
        kps=np.array(
            [[1.0, 2.0], [4.0, 2.0], [2.5, 4.0], [1.5, 6.0], [3.5, 6.0]],
            dtype=np.float32,
        ),
        det_score=0.93,
        embedding=np.array([3.0, 4.0], dtype=np.float32),
        normed_embedding=None,
        gender=1,
        age=28,
    )
    analysis = _Analysis({"detection": _Model("detection")}, [face])
    monkeypatch.setattr(face_engine, "FaceAnalysis", lambda **_kwargs: analysis)
    engine = face_engine.FaceEngine(
        custom_model_dir=tmp_path,
        providers=["CPUExecutionProvider"],
    )
    engine.load()
    source = np.zeros((12, 20, 3), dtype=np.uint8)[:, ::2]

    records = engine.detect_faces(source, source_path="portrait.jpg")

    assert len(records) == 1
    record = records[0]
    assert record.face_id == "portrait.jpg:0"
    assert record.bbox == [1.0, 2.0, 5.0, 7.0]
    assert record.kps == face.kps.tolist()
    assert record.det_score == 0.93
    assert np.array_equal(record.embedding, np.array([3.0, 4.0], dtype=np.float32))
    assert np.allclose(record.normed_embedding, np.array([0.6, 0.8]))
    assert record.gender == 1
    assert record.age == 28
    assert record.source_path == "portrait.jpg"
    assert record.crop.shape == (6, 4, 3)
    assert len(analysis.get_calls) == 1
    image, kwargs = analysis.get_calls[0]
    assert image.flags.c_contiguous is True
    assert kwargs == {"max_num": 0, "det_metric": "default"}
