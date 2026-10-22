from pathlib import Path

import numpy as np
import pytest
from insightface.addons.liveness import OUT_OF_BOUNDS_REASON
from insightface.app import face_analysis
from insightface.app.face_analysis import FaceAnalysis

RESULTS = [
    {"status": "ok", "is_live": False, "live_score": 0.1},
    {
        "status": "input_rejected", "is_live": None, "live_score": None,
        "reason": OUT_OF_BOUNDS_REASON,
    },
    {"status": "ok", "is_live": True, "live_score": 0.95},
]


@pytest.fixture
def setup_pipeline(monkeypatch, tmp_path, manifest_package_factory):
    def create(
        mode=None, *, enabled=True, manifest=False, recognition=True, empty=False
    ):
        events = []
        loads = []

        class Model:
            def __init__(self, task):
                self.taskname = task
                self.input_shape = [1, 3, 112, 112]
                self.input_mean = self.input_std = 1

            def prepare(self, ctx_id, **kwargs):
                events.append((self.taskname, "prepare", ctx_id))

            def detect(self, img, max_num=0, metric="default"):
                events.append(("detect", max_num, metric))
                n = 0 if empty else (min(max_num, 3) if max_num else 3)
                return np.array([[i, 0, i + 10, 10, 0.9] for i in range(n)]).reshape(
                    n, 5
                ), None

            def get(self, img, face):
                index = int(face.bbox[0])
                events.append((self.taskname, index))
                if self.taskname == "recognition":
                    face.embedding = np.array([1.0, 0.0])
                else:
                    face.face_probability = 0.99

        class Addon:
            def __init__(self, path, **kwargs):
                loads.append((path, kwargs))

            def prepare(self, ctx_id):
                events.append(("liveness", "prepare", ctx_id))

            def get(self, img, face):
                index = int(face.bbox[0])
                events.append(("liveness", index))
                face.liveness = dict(RESULTS[index])

        models = {
            task: Model(task) for task in ("detection", "recognition", "verification")
        }
        if manifest:
            directory, _ = manifest_package_factory()
        else:
            directory = tmp_path / "base"
            directory.mkdir()
            for task in models:
                (directory / f"{task}.onnx").touch()

        def get_model(path, **kwargs):
            assert not {"addons", "liveness_mode", "liveness_threshold"}.intersection(
                kwargs
            )
            return models[kwargs.get("model_task", Path(path).stem)]

        def ensure_addon(name, root):
            events.append(("download", name))
            return Path(root) / "addons" / f"{name}.onnx"

        monkeypatch.setattr(face_analysis.model_zoo, "get_model", get_model)
        monkeypatch.setattr(face_analysis, "ensure_addon", ensure_addon)
        monkeypatch.setattr(face_analysis, "Liveness", Addon)
        selected = ["detection", "verification"]
        if recognition:
            selected.append("recognition")
        analysis = FaceAnalysis(
            directory,
            root=tmp_path / "custom-root",
            allowed_modules=selected,
            addons=["liveness"] if enabled else None,
            liveness_threshold=0.85,
            providers=["CPUExecutionProvider"],
            provider_options=[{}],
            **({} if mode is None else {"liveness_mode": mode}),
        )
        return analysis, events, loads

    return create


@pytest.mark.parametrize("manifest", [False, True])
@pytest.mark.parametrize("mode", [None, "normal", "observe"])
def test_pipeline_retains_faces_and_gates_only_recognition(
    setup_pipeline, manifest, mode
):
    analysis, events, loads = setup_pipeline(mode, manifest=manifest)
    analysis.prepare(ctx_id=-1)
    faces = analysis.get(np.zeros((80, 80, 3), np.uint8))
    assert len(faces) == 3
    normal = mode in (None, "normal")
    assert [face.embedding is not None for face in faces] == (
        [False, False, True] if normal else [True] * 3
    )
    assert all(face.face_probability == 0.99 for face in faces)
    assert set(analysis.models) == {"detection", "recognition", "verification"}
    assert analysis.liveness_mode == (mode or "normal")
    assert [dict(face.liveness) for face in faces] == RESULTS
    assert faces[0]["liveness"]["is_live"] is False
    assert set(faces[1].liveness) == {"status", "is_live", "live_score", "reason"}
    assert faces[1].liveness.reason == OUT_OF_BOUNDS_REASON
    assert all("reason" not in faces[index].liveness for index in (0, 2))
    for index in range(3):
        if ("recognition", index) in events:
            assert events.index(("liveness", index)) < events.index(
                ("recognition", index)
            )
    assert ("liveness", "prepare", -1) in events
    assert loads[0][0].parts[-3:] == ("custom-root", "addons", "liveness.onnx")
    assert loads[0][1]["threshold"] == 0.85
    assert loads[0][1]["providers"] == ["CPUExecutionProvider"]
    assert loads[0][1]["provider_options"] == [{}]


@pytest.mark.parametrize("manifest", [False, True])
@pytest.mark.parametrize("mode", [None, "normal", "observe"])
def test_unconfigured_addon_preserves_existing_results(setup_pipeline, manifest, mode):
    analysis, events, loads = setup_pipeline(mode, enabled=False, manifest=manifest)
    faces = analysis.get(np.zeros((80, 80, 3), np.uint8))
    assert not loads
    assert analysis.liveness_mode == (mode or "normal")
    assert analysis.addons == {}
    assert not any(event[0] == "download" for event in events)
    assert all(
        "liveness" not in face and face.liveness is None and face.embedding is not None
        for face in faces
    )


def test_detection_only_still_evaluates_liveness(setup_pipeline):
    analysis, events, _ = setup_pipeline(recognition=False)
    faces = analysis.get(np.zeros((80, 80, 3), np.uint8), max_num=2, det_metric="max")
    assert len(faces) == 2
    assert all(face.embedding is None for face in faces)
    assert [dict(face.liveness) for face in faces] == RESULTS[:2]
    assert ("detect", 2, "max") in events


def test_no_faces_does_not_run_liveness(setup_pipeline):
    analysis, events, _ = setup_pipeline(empty=True)
    assert analysis.get(np.zeros((80, 80, 3), np.uint8)) == []
    assert not any(event[0] == "liveness" for event in events)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"addons": "liveness"}, TypeError),
        ({"addons": ["missing"]}, ValueError),
        ({"addons": ["liveness", "liveness"]}, ValueError),
        ({"liveness_mode": "enforce"}, ValueError),
        ({"addons": ["liveness"], "liveness_mode": "enforce"}, ValueError),
        ({"addons": ["liveness"], "liveness_mode": "off"}, ValueError),
        ({"liveness_mode": None}, ValueError),
        ({"liveness_mode": "invalid"}, ValueError),
        ({"liveness_threshold": -0.1}, ValueError),
        ({"liveness_threshold": float("nan")}, ValueError),
        ({"liveness_threshold": True}, TypeError),
    ],
)
def test_invalid_config_fails_before_loading_or_downloading(monkeypatch, kwargs, error):
    def fail(*args, **kwargs):
        pytest.fail("model loading must not start")

    monkeypatch.setattr(face_analysis, "ensure_available", fail)
    with pytest.raises(error):
        FaceAnalysis(**kwargs)


@pytest.mark.parametrize("mode", ["normal", "observe"])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_liveness_runtime_error_never_falls_through_to_recognition(
    setup_pipeline, mode, error_type
):
    analysis, events, _ = setup_pipeline(mode)

    def fail(*args):
        raise error_type("liveness unavailable")

    analysis.addons["liveness"].get = fail
    with pytest.raises(error_type, match="liveness unavailable"):
        analysis.get(np.zeros((80, 80, 3), np.uint8))
    assert not any(event[0] == "recognition" for event in events)
