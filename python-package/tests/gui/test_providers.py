import os

import pytest

from insightface.gui.core import face_engine


def test_provider_runtime_display_resolves_auto_coreml_chain(monkeypatch):
    monkeypatch.setattr(
        face_engine,
        "available_execution_providers",
        lambda: [
            "CoreMLExecutionProvider",
            "AzureExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    primary, tooltip = face_engine.provider_runtime_display("Auto")

    assert primary == "CoreMLExecutionProvider"
    assert "CPUExecutionProvider (fallback)" in tooltip
    assert "AzureExecutionProvider" not in tooltip


def test_provider_runtime_display_resolves_auto_cuda_chain(monkeypatch):
    monkeypatch.setattr(
        face_engine,
        "available_execution_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    primary, tooltip = face_engine.provider_runtime_display("Auto")

    assert primary == "CUDAExecutionProvider"
    assert "CPUExecutionProvider (fallback)" in tooltip


def test_provider_runtime_display_resolves_cpu_choice(monkeypatch):
    monkeypatch.setattr(
        face_engine,
        "available_execution_providers",
        lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )

    primary, tooltip = face_engine.provider_runtime_display("CPU")

    assert primary == "CPUExecutionProvider"
    assert "CoreMLExecutionProvider" not in tooltip
    assert "fallback" not in tooltip.lower()


def test_provider_runtime_display_reports_unavailable_without_providers(monkeypatch):
    monkeypatch.setattr(face_engine, "available_execution_providers", lambda: [])

    primary, tooltip = face_engine.provider_runtime_display("Auto")

    assert primary == "Unavailable"
    assert "no available execution providers" in tooltip.lower()


def test_face_engine_uses_global_default_providers(monkeypatch):
    selected = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.setattr(face_engine, "get_default_providers", lambda: selected)

    engine = face_engine.FaceEngine()

    assert engine.requested_providers == selected


def test_cuda_choice_falls_back_when_provider_is_unavailable(monkeypatch):
    monkeypatch.setattr(face_engine, "available_execution_providers", lambda: ["CPUExecutionProvider"])

    assert face_engine.is_cuda_provider_available() is False
    assert face_engine.providers_from_choice("CUDA") == ["CPUExecutionProvider"]


def test_cuda_choice_is_enabled_when_provider_is_available(monkeypatch):
    monkeypatch.setattr(
        face_engine,
        "available_execution_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    assert face_engine.is_cuda_provider_available() is True
    assert face_engine.providers_from_choice("CUDA") == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert face_engine.providers_from_choice("Auto") == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_cuda_choice_only_adds_an_available_cpu_fallback(monkeypatch):
    monkeypatch.setattr(
        face_engine,
        "available_execution_providers",
        lambda: ["CUDAExecutionProvider"],
    )

    assert face_engine.providers_from_choice("CUDA") == [
        "CUDAExecutionProvider"
    ]


def test_auto_choice_prefers_coreml_over_cuda(monkeypatch):
    monkeypatch.setattr(
        face_engine,
        "available_execution_providers",
        lambda: [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
            "CoreMLExecutionProvider",
        ],
    )

    assert face_engine.providers_from_choice("Auto") == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_coreml_face_engine_does_not_force_cpu(monkeypatch, tmp_path):
    model_file = tmp_path / "detector.onnx"
    model_file.write_bytes(b"fake")
    prepare_calls = []
    constructor_calls = []

    class Detector:
        taskname = "detection"

    detector = Detector()

    class Analysis:
        models = {"detection": detector}
        det_model = detector

        def prepare(self, ctx_id, **kwargs):
            prepare_calls.append((ctx_id, kwargs))

    def make_analysis(**kwargs):
        constructor_calls.append(kwargs)
        return Analysis()

    engine = face_engine.FaceEngine(
        custom_model_dir=tmp_path,
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(face_engine, "FaceAnalysis", make_analysis)

    engine.load()

    assert engine.is_loaded() is True
    assert engine.ctx_id == 0
    assert prepare_calls[0][0] == 0
    assert constructor_calls[0]["static_shape_sessions"] is True
    assert constructor_calls[0]["_coreml_detector_input_size"] == (640, 640)


def test_main_window_and_dashboard_show_resolved_provider(
    monkeypatch,
    tmp_path,
):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import StudioContext, configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.storage import Storage
    from insightface.gui.main_window import MainWindow
    from insightface.gui.pages.dashboard_page import DashboardPage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    # Test provider selection independently of platform font-dependent elision.
    monkeypatch.setattr(MainWindow, "_elide", lambda self, text, width: text)
    available = [
        "CoreMLExecutionProvider",
        "AzureExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setattr(
        face_engine,
        "available_execution_providers",
        lambda: list(available),
    )
    config = AppConfig(
        workspace_path=str(tmp_path),
        model_name="raccoon_s",
        auto_load_model=False,
        safe_mode=True,
        provider="Auto",
        ui_language="en",
    )
    context = StudioContext(
        config,
        True,
        Storage(config.database_path),
        face_engine.FaceEngine(model_name=config.model_name),
        str(tmp_path / "app.log"),
    )
    window = MainWindow(context)
    dashboard = DashboardPage(context)

    assert "CoreMLExecutionProvider" in window.provider_chip.text()
    assert "Auto" not in window.provider_chip.text()
    assert "CoreMLExecutionProvider" in window.status_labels["provider"].text()
    assert "CPUExecutionProvider (fallback)" in window.provider_chip.toolTip()
    assert "AzureExecutionProvider" not in window.provider_chip.toolTip()
    assert dashboard.cards["provider"].value_label.text() == "CoreMLExecutionProvider"

    available[:] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    window.refresh_statusbar()
    dashboard.refresh()

    assert "CUDAExecutionProvider" in window.provider_chip.text()
    assert dashboard.cards["provider"].value_label.text() == "CUDAExecutionProvider"

    # Cheetah uses PersonAnalysis, whose Auto policy supports CUDA/CPU only.
    # It must not inherit the ordinary face workflow's CoreML display.
    import onnxruntime
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"])
    for model_name in ("cheetah_s", "cheetah_l"):
        config.model_name = model_name
        window.refresh_statusbar()
        assert "CPUExecutionProvider" in window.provider_chip.text()
        assert "CoreMLExecutionProvider" not in window.provider_chip.text()
        assert "CoreMLExecutionProvider" not in window.provider_chip.toolTip()
    dashboard.close()
    window.close()
