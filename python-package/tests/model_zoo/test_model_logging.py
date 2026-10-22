import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from insightface.app import face_analysis
from insightface.model_zoo import model_zoo


class _Model:
    def __init__(self, task):
        self.taskname = task
        self.input_shape = [1, 3, 112, 112]
        self.input_mean = 0.0
        self.input_std = 1.0
        self.prepare_calls = []

    def prepare(self, *args, **kwargs):
        self.prepare_calls.append((args, kwargs))


def _assert_debug_only(caplog, capsys, module, enabled, messages):
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    records = [record for record in caplog.records if record.name == module.__name__]
    if enabled:
        assert all(record.levelno == logging.DEBUG for record in records)
        for message in messages:
            assert any(message in record.getMessage() for record in records)
    else:
        assert records == []


@pytest.mark.parametrize("debug_logging", [False, True])
def test_manifest_loading_and_prepare_only_log_at_debug(
    manifest_package_factory, monkeypatch, caplog, capsys, debug_logging
):
    package, _ = manifest_package_factory()
    models = {task: _Model(task) for task in ("detection", "verification", "recognition")}
    monkeypatch.setattr(
        face_analysis.model_zoo,
        "get_model",
        lambda _name, **kwargs: models[kwargs["model_task"]],
    )
    caplog.set_level(
        logging.DEBUG if debug_logging else logging.WARNING,
        logger=face_analysis.__name__,
    )

    analysis = face_analysis.FaceAnalysis(package, providers=["CPUExecutionProvider"])
    analysis.prepare(ctx_id=0, det_size=(320, 320))

    assert analysis.models == models
    assert models["detection"].prepare_calls == [
        ((0,), {"input_size": (320, 320), "det_thresh": 0.5})
    ]
    _assert_debug_only(
        caplog, capsys, face_analysis, debug_logging,
        ["find manifest model:", "set det-size:"],
    )


@pytest.mark.parametrize("debug_logging", [False, True])
def test_legacy_model_selection_only_logs_at_debug(
    tmp_path, monkeypatch, caplog, capsys, debug_logging
):
    models = {
        "01_detector.onnx": _Model("detection"),
        "02_duplicate.onnx": _Model("detection"),
        "03_ignored.onnx": _Model("recognition"),
        "04_unknown.onnx": None,
    }
    for filename in models:
        (tmp_path / filename).write_bytes(b"fake")
    monkeypatch.setattr(
        face_analysis.model_zoo,
        "get_model",
        lambda path, **_kwargs: models[Path(path).name],
    )
    caplog.set_level(
        logging.DEBUG if debug_logging else logging.WARNING,
        logger=face_analysis.__name__,
    )

    analysis = face_analysis.FaceAnalysis(
        tmp_path, allowed_modules=["detection"], providers=["CPUExecutionProvider"]
    )

    assert analysis.models == {"detection": models["01_detector.onnx"]}
    _assert_debug_only(
        caplog, capsys, face_analysis, debug_logging,
        ["find model:", "duplicated model task type, ignore:", "model ignore:", "model not recognized:"],
    )


@pytest.mark.parametrize("debug_logging", [False, True])
def test_model_router_provider_diagnostics_only_log_at_debug(
    monkeypatch, caplog, capsys, debug_logging
):
    session = SimpleNamespace(
        _providers=["CPUExecutionProvider"],
        _provider_options={"CPUExecutionProvider": {}},
        get_inputs=lambda: [SimpleNamespace(shape=[1, 3, 112, 112])],
        get_outputs=lambda: [SimpleNamespace(shape=[1, 512])],
    )
    monkeypatch.setattr(model_zoo, "PickableInferenceSession", lambda *_args, **_kwargs: session)
    recognizer = object()
    monkeypatch.setattr(model_zoo, "ArcFaceONNX", lambda **_kwargs: recognizer)
    caplog.set_level(
        logging.DEBUG if debug_logging else logging.WARNING,
        logger=model_zoo.__name__,
    )

    model = model_zoo.ModelRouter("recognizer.onnx").get_model(
        providers=["CPUExecutionProvider"]
    )

    assert model is recognizer
    _assert_debug_only(
        caplog, capsys, model_zoo, debug_logging,
        ["Applied providers:", "CPUExecutionProvider", "with options:"],
    )
