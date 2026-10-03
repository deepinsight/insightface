from insightface.gui.__main__ import main
from insightface.gui.app import create_context
from insightface.gui.core.config import AppConfig, save_config


def test_cli_import_and_version(capsys):
    import insightface
    import insightface.gui

    assert insightface.__version__ == "2.1"
    assert insightface.gui.__version__ == "2.1"
    assert main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "InsightFace Evaluation Studio 2.1" in out
    assert "insightface 2.1" in out


def test_insightface_cli_import_does_not_require_mxnet():
    from insightface.commands import insightface_cli

    assert callable(insightface_cli.main)


def test_safe_mode_is_runtime_only(tmp_path):
    workspace = tmp_path / "workspace"
    cfg = AppConfig(workspace_path=str(workspace))
    cfg.safe_mode = True
    cfg.auto_load_model = False
    save_config(cfg)

    args = type(
        "Args",
        (),
        {"workspace": str(workspace), "model": None, "provider": None, "safe_mode": True},
    )
    context = create_context(args())

    assert context.runtime_safe_mode is True
    assert context.config.safe_mode is False
    assert context.config.auto_load_model is True


def test_bad_person_json_shows_path_and_field_after_qt_initialization(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from PySide6.QtWidgets import QApplication, QMessageBox
    from insightface.gui import app as module

    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    path = tmp_path / 'config.json'
    original = json.dumps({'workspace_path': str(tmp_path), 'model_name': 'cheetah_l',
                           'person_config': {'face_interval_ms': -1}})
    path.write_text(original, encoding='utf-8')
    messages = []

    def critical(parent, title, message):
        assert QApplication.instance() is not None
        messages.append((title, message))
        return 0

    monkeypatch.setattr(QMessageBox, 'critical', critical)
    assert module.run_app(SimpleNamespace(workspace=str(tmp_path))) == 2
    assert len(messages) == 1
    assert str(path) in messages[0][1] and 'face_interval_ms' in messages[0][1]
    assert path.read_text(encoding='utf-8') == original
    assert not (tmp_path / 'insightface_gui.db').exists()


def test_startup_does_not_swallow_unrelated_errors(monkeypatch):
    import pytest
    from PySide6.QtWidgets import QMessageBox
    from insightface.gui import app as module

    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    def fail(_args):
        raise RuntimeError('unrelated startup failure')
    monkeypatch.setattr(module, 'create_context', fail)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: pytest.fail('Wrong error handler'))
    with pytest.raises(RuntimeError, match='unrelated startup failure'):
        module.run_app()
