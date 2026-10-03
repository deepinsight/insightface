"""Cheetah GUI installation uses SDK contracts without model inference/network."""
import os
import urllib.error
import zipfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from insightface.gui.core import model_downloads, model_packages
from insightface.gui.core.model_packages import (
    ensure_person_model, inspect_person_model, person_model_providers,
    person_provider_runtime_display,
)


@pytest.fixture
def write_package(synthetic_person_manifest, synthetic_person_package):
    def write(root, name="cheetah_s"):
        size = 320 if name == "cheetah_s" else 640
        manifest = synthetic_person_manifest(name, size)
        return synthetic_person_package(root, manifest)
    return write


def asset(name="cheetah_s"):
    return next(item for item in model_downloads.fallback_model_assets() if item.stem == name)


@pytest.mark.parametrize("name", model_packages.PERSON_MODEL_PACKAGES)
def test_missing_person_package_is_selectable_without_network(tmp_path, monkeypatch, name):
    monkeypatch.setattr(model_downloads.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    status = inspect_person_model(name, tmp_path)
    assert status.state == "missing" and status.can_start and not status.installed
    assert status.package_path == tmp_path / "models" / name
    assert str(status.package_path) in status.message
    assert not status.package_path.exists()
    assert asset(name).browser_download_url.endswith(f"/model-zoo/{name}.zip")


def test_unsupported_empty_and_partial_person_packages_cannot_start(tmp_path):
    assert inspect_person_model("buffalo_l", tmp_path).state == "unsupported"
    assert inspect_person_model("cheetah_s", "").state == "invalid"
    folder = tmp_path / "models" / "cheetah_s"
    folder.mkdir(parents=True)
    (folder / "face.onnx").write_bytes(b"just one model")
    status = inspect_person_model("cheetah_s", tmp_path)
    assert status.state == "invalid" and not status.can_start
    assert not model_downloads.is_model_package_installed("cheetah_s", tmp_path)
    assert not model_downloads.is_model_asset_installed(asset(), tmp_path)
    assert "manifest" in model_downloads.local_model_status(asset(), tmp_path)


@pytest.mark.parametrize("name", model_packages.PERSON_MODEL_PACKAGES)
def test_valid_local_package_never_downloads_or_creates_inference_session(tmp_path, monkeypatch, name, write_package):
    import onnxruntime
    folder = write_package(tmp_path, name)
    monkeypatch.setattr(model_downloads, "_download_with_retries", lambda *a, **k: pytest.fail("download"))
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **k: pytest.fail("inference"))
    assert inspect_person_model(name, tmp_path).installed
    assert ensure_person_model(name, tmp_path) == folder
    assert model_downloads.download_model_asset(asset(name), tmp_path, tmp_path / "cache") == folder
    assert not (tmp_path / "gui").exists()


def test_local_validation_detects_changed_artifacts_after_cached_success(tmp_path, write_package):
    folder = write_package(tmp_path)
    assert inspect_person_model("cheetah_s", tmp_path).installed
    (folder / "person_reid.onnx").write_bytes(b"bad weights")
    status = inspect_person_model("cheetah_s", tmp_path)
    assert status.state == "invalid" and "SHA-256 mismatch" in status.message


def test_invalid_local_package_is_never_downloaded_over(tmp_path, monkeypatch, write_package):
    folder = write_package(tmp_path)
    original = b"damaged but user-owned"
    (folder / "person_reid.onnx").write_bytes(original)
    monkeypatch.setattr(model_downloads, "_download_with_retries", lambda *a, **k: pytest.fail("download"))
    with pytest.raises(RuntimeError, match="will not be overwritten"):
        ensure_person_model("cheetah_s", tmp_path)
    assert (folder / "person_reid.onnx").read_bytes() == original


def test_missing_remote_release_asset_is_not_retried_and_reports_install_path(tmp_path, monkeypatch):
    requests = []

    def unavailable(request, **kwargs):
        requests.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(model_downloads.urllib.request, "urlopen", unavailable)
    monkeypatch.setattr(model_downloads.time, "sleep", lambda *args: pytest.fail("404 retry"))
    with pytest.raises(RuntimeError, match="HTTP 404") as error:
        ensure_person_model("cheetah_s", tmp_path)
    assert requests == [asset().browser_download_url]
    assert "may not have been published" in str(error.value)
    assert str(tmp_path / "models" / "cheetah_s") in str(error.value)
    assert not (tmp_path / "models" / "cheetah_s").exists()


def stage_download(monkeypatch, folder, *, wrapped=False, after_download=None):
    def download(url, destination, asset_name, **kwargs):
        assert url.endswith("/model-zoo/cheetah_s.zip")
        with zipfile.ZipFile(destination, "w") as archive:
            for path in folder.iterdir():
                archive.write(path, f"cheetah_s/{path.name}" if wrapped else path.name)
        if after_download is not None:
            after_download()
    monkeypatch.setattr(model_downloads, "_download_with_retries", download)


@pytest.mark.parametrize("wrapped", [False, True])
def test_download_validates_four_tasks_before_atomic_install(tmp_path, monkeypatch, wrapped, write_package):
    source = write_package(tmp_path / "source")
    stage_download(monkeypatch, source, wrapped=wrapped)
    root = tmp_path / "installed"
    result = ensure_person_model("cheetah_s", root, tmp_path / "cache")
    assert result == root / "models" / "cheetah_s"
    assert inspect_person_model("cheetah_s", root).installed
    assert not list(result.parent.glob(".cheetah_s-install-*"))


def test_invalid_remote_archive_never_becomes_installed(tmp_path, monkeypatch, write_package):
    source = write_package(tmp_path / "source")
    (source / "person_reid.onnx").write_bytes(b"corrupted in archive")
    stage_download(monkeypatch, source)
    root = tmp_path / "installed"
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        ensure_person_model("cheetah_s", root)
    assert not (root / "models" / "cheetah_s").exists()
    assert not list((root / "models").glob(".cheetah_s-install-*"))


def test_local_package_created_during_download_is_preserved(tmp_path, monkeypatch, write_package):
    source = write_package(tmp_path / "source")
    root = tmp_path / "installed"
    target = root / "models" / "cheetah_s"
    def create_local():
        target.mkdir()
        (target / "user.txt").write_text("keep")
    stage_download(monkeypatch, source, after_download=create_local)
    with pytest.raises(RuntimeError, match="was not replaced"):
        ensure_person_model("cheetah_s", root)
    assert (target / "user.txt").read_text() == "keep"
    assert not (target / "manifest.json").exists()


def test_cancelled_download_never_installs_package(tmp_path, monkeypatch, write_package):
    source = write_package(tmp_path / "source")
    cancelled = []
    stage_download(monkeypatch, source, after_download=lambda: cancelled.append(True))
    root = tmp_path / "installed"
    with pytest.raises(RuntimeError, match="cancelled"):
        ensure_person_model("cheetah_s", root, is_cancelled=lambda: bool(cancelled))
    assert not (root / "models" / "cheetah_s").exists()


@pytest.mark.parametrize("choice,available,expected", [
    ("Auto", ["CoreMLExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]),
    ("Auto", ["CUDAExecutionProvider", "CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    ("CPU", ["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]),
])
def test_person_provider_policy_and_display_match(choice, available, expected, monkeypatch):
    import onnxruntime
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: available)
    assert person_model_providers(choice) == expected
    label, detail = person_provider_runtime_display(choice)
    assert label == expected[0]
    assert "CoreML" not in detail


def test_explicit_cuda_never_silently_falls_back(monkeypatch):
    import onnxruntime
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="unavailable"):
        person_model_providers("CUDA")
    assert person_provider_runtime_display("CUDA")[0] == "Unavailable"
    with pytest.raises(ValueError, match="Auto, CPU, or CUDA"):
        person_model_providers("CoreML")


@pytest.mark.parametrize("name", model_packages.PERSON_MODEL_PACKAGES)
def test_face_engine_never_scans_or_loads_selected_person_package(tmp_path, monkeypatch, name):
    from insightface.gui.core import face_engine
    monkeypatch.setattr(face_engine, "FaceAnalysis", lambda *a, **k: pytest.fail("FaceAnalysis"))
    engine = face_engine.FaceEngine(name, root=tmp_path, providers=["CPUExecutionProvider"])
    monkeypatch.setattr(engine, "resolve_model_dir", lambda: pytest.fail("ONNX scan"))
    engine.load()
    assert not engine.is_loaded()
    assert "Use Person Analysis or select a face model in Models" in engine.last_error


@pytest.fixture
def settings_page(tmp_path):
    from PySide6.QtWidgets import QApplication
    from types import SimpleNamespace
    from insightface.gui.app import configure_qt_plugin_paths, create_face_engine
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages.model_settings_page import ModelSettingsPage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(workspace_path=str(tmp_path / "workspace"),
                       model_root=str(tmp_path / "models-root"), provider="CPU",
                       model_name="cheetah_s", auto_load_model=False)
    context = SimpleNamespace(config=config, engine=create_face_engine(config))
    page = ModelSettingsPage(context)
    yield page
    page.close()


@pytest.mark.parametrize("prepare_error", [False, True])
def test_person_test_load_prepares_and_closes_in_background_task(settings_page, monkeypatch, prepare_error):
    import insightface.app.person_analysis as person_analysis
    from insightface.gui.pages import model_settings_page

    calls = []
    captured = []

    class FakeAnalysis:
        def __init__(self, **kwargs):
            calls.append(("construct", kwargs))
        def prepare(self):
            calls.append("prepare")
            if prepare_error:
                raise RuntimeError("model node validation failed")
        def close(self):
            calls.append("close")

    monkeypatch.setattr(person_analysis, "PersonAnalysis", FakeAnalysis)
    monkeypatch.setattr(model_settings_page, "ensure_person_model", lambda *args: calls.append("ensure"))
    monkeypatch.setattr(model_settings_page, "FaceEngine", lambda *a, **kw: pytest.fail("FaceEngine"))
    monkeypatch.setattr(settings_page, "run_task", lambda *args, **kwargs: captured.append((args, kwargs)))
    original_engine = settings_page.context.engine
    settings_page.test_load()
    assert calls == []  # Clicking only schedules work; no prepare on the GUI thread.
    assert settings_page.context.person_analysis_jobs_in_progress == 1
    args, kwargs = captured[0]
    try:
        if prepare_error:
            with pytest.raises(RuntimeError, match="node validation"):
                args[1]()
        else:
            result = args[1]()
            args[2](result)
            assert result["providers"] == ["CPUExecutionProvider"]
            assert "closed successfully" in settings_page.status_label.text()
    finally:
        kwargs["on_finished"]()
    assert calls[0] == "ensure"
    assert calls[-2:] == ["prepare", "close"]
    assert settings_page.context.engine is original_engine
    assert settings_page.context.person_analysis_jobs_in_progress == 0


def test_person_input_size_and_provider_controls(settings_page):
    assert not settings_page.det_combo.isEnabled()
    assert "otherwise CPU" in settings_page.provider_combo.toolTip()
    settings_page.model_combo.setCurrentIndex(settings_page.model_combo.findData("buffalo_l"))
    assert settings_page.det_combo.isEnabled()
    assert "CoreML" in settings_page.provider_combo.toolTip()


def test_person_activity_blocks_model_download_and_selection(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from types import SimpleNamespace
    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.pages.model_download_page import ModelDownloadPage

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    config = AppConfig(workspace_path=str(tmp_path / "workspace"), model_root=str(tmp_path / "root"))
    context = SimpleNamespace(config=config, person_analysis_jobs_in_progress=1)
    page = ModelDownloadPage(context)
    page.assets = [asset()]
    page.populate()
    page.table.selectRow(0)
    errors = []
    monkeypatch.setattr(page, "show_error", errors.append)
    monkeypatch.setattr(page, "run_task", lambda *a, **kw: pytest.fail("background download"))
    assert not page.download_selected_button.isEnabled()
    assert not page.use_selected_button.isEnabled()
    page.download_selected()
    page.use_selected_model()
    assert len(errors) == 2
    assert config.model_name == "cheetah_s"
    page.close()
