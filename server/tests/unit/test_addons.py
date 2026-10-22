from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from insightface.addons import catalog
from insightface_server import models_cli
from insightface_server.addons import install_addon, require_installed_addon
from insightface_server.config import Settings, load_server_config
from insightface_server.inference.onnx_engine import OnnxInsightFaceEngine

SHIPPED_CONFIG = Path(__file__).resolve().parents[2] / "config" / "server.toml"


def test_shipped_config_disables_liveness_installation_and_runtime(monkeypatch):
    config = load_server_config(SHIPPED_CONFIG)
    assert config.addons == config.auto_download_addons == ()
    assert config.liveness_mode == "normal"
    assert config.liveness_on_registration is False
    monkeypatch.setenv("INSIGHTFACE_CONFIG_FILE", str(SHIPPED_CONFIG))
    settings = Settings.from_env()
    assert settings.addons == settings.auto_download_addons == ()
    assert settings.liveness_on_registration is False


@pytest.mark.parametrize("package", ["buffalo_l", "raccoon_s", "raccoon_l"])
def test_shipped_config_does_not_install_liveness_with_cached_base_package(
    tmp_path, monkeypatch, package
):
    monkeypatch.setenv("INSIGHTFACE_CONFIG_FILE", str(SHIPPED_CONFIG))
    monkeypatch.setattr(models_cli, "installed_package_name", lambda root: package)
    calls = []
    monkeypatch.setattr(
        models_cli, "install_package",
        lambda selected, root: calls.append((selected.name, root)) or "already_installed",
    )
    monkeypatch.setattr(
        models_cli, "install_addon",
        lambda name, root: calls.append((name, root)) or root / "addons/liveness.onnx",
    )
    assert models_cli.main(["--models-dir", str(tmp_path), "install", package]) == 0
    assert calls == [(package, tmp_path)]


def test_old_config_disables_addons_and_new_settings_are_independent(tmp_path):
    path = tmp_path / "server.toml"
    path.write_text('[inference]\nmax_concurrency="auto"\n')
    old = load_server_config(path)
    assert old.addons == old.auto_download_addons == ()
    assert old.liveness_mode == "normal"
    assert old.liveness_on_registration is False
    path.write_text(
        '[inference]\naddons=["liveness"]\nliveness_mode="observe"\nliveness_threshold=0.9\nliveness_compare_scope="target"\n[addons]\nauto_download=[]\n'
    )
    config = load_server_config(path)
    assert config.addons == ("liveness",)
    assert config.auto_download_addons == ()
    assert config.liveness_mode == "observe"
    assert config.liveness_threshold == 0.9
    assert config.liveness_compare_scope == "target"
    assert config.liveness_on_registration is False


@pytest.mark.parametrize("value,expected", [(None, False), ("false", False), ("true", True)])
def test_registration_liveness_defaults_and_config_propagation(
    tmp_path, monkeypatch, value, expected
):
    path = tmp_path / "server.toml"
    content = '[inference]\naddons=["liveness"]\n'
    if value is not None:
        content += f"liveness_on_registration={value}\n"
    path.write_text(content)
    monkeypatch.setenv("INSIGHTFACE_CONFIG_FILE", str(path))
    assert load_server_config(path).liveness_on_registration is expected
    assert Settings.from_env().liveness_on_registration is expected


@pytest.mark.parametrize("value", ['"false"', '"true"', "0", "1", "[]"])
def test_registration_liveness_requires_a_boolean(tmp_path, value):
    path = tmp_path / "server.toml"
    path.write_text(f"[inference]\nliveness_on_registration={value}\n")
    with pytest.raises(ValueError, match="liveness_on_registration must be a boolean"):
        load_server_config(path)


@pytest.mark.parametrize(
    "setting",
    [
        '[inference]\naddons="liveness"',
        '[inference]\naddons=["missing"]',
        '[inference]\naddons=["liveness","liveness"]',
        '[inference]\nliveness_mode="off"',
        '[inference]\nliveness_mode="enforce"',
        "[inference]\nliveness_threshold=nan",
        '[inference]\nliveness_compare_scope="none"',
        "[addons]\nauto_download=true",
        '[addons]\nauto_download=["missing"]',
    ],
)
def test_invalid_addon_config_is_rejected(tmp_path, setting):
    path = tmp_path / "server.toml"
    path.write_text(setting)
    with pytest.raises(ValueError):
        load_server_config(path)


@pytest.mark.parametrize("package", list(models_cli.PACKAGES))
@pytest.mark.parametrize("status", ["installed", "already_installed"])
def test_every_base_install_including_cached_packages_installs_configured_addons(
    tmp_path, monkeypatch, package, status
):
    config = tmp_path / "server.toml"
    config.write_text('[addons]\nauto_download=["liveness"]\n')
    calls = []
    monkeypatch.setattr(models_cli, "installed_package_name", lambda root: package)
    monkeypatch.setattr(models_cli, "install_package", lambda *args: calls.append("base") or status)
    monkeypatch.setattr(
        models_cli,
        "install_addon",
        lambda name, root: calls.append(name) or root / "addons/liveness.onnx",
    )
    assert (
        models_cli.main(
            ["--config-file", str(config), "--models-dir", str(tmp_path), "install", package]
        )
        == 0
    )
    assert calls == ["base", "liveness"]


def test_addon_failure_returns_nonzero_and_reports_base_was_kept(tmp_path, monkeypatch, capsys):
    config = tmp_path / "server.toml"
    config.write_text('[addons]\nauto_download=["liveness"]\n')
    monkeypatch.setattr(models_cli, "installed_package_name", lambda root: "buffalo_l")
    monkeypatch.setattr(models_cli, "install_package", lambda *args: "already_installed")

    def fail(*args):
        raise OSError("download interrupted")

    monkeypatch.setattr(models_cli, "install_addon", fail)
    assert (
        models_cli.main(
            ["--config-file", str(config), "--models-dir", str(tmp_path), "install", "buffalo_l"]
        )
        == 2
    )
    message = capsys.readouterr().err
    assert "Base model package buffalo_l is installed" in message
    assert "addon liveness installation failed" in message
    assert "Rerun the same install command" in message


def test_missing_addon_stops_startup_before_any_download_or_session(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("startup must not download models")

    monkeypatch.setattr(catalog.requests, "get", forbidden)
    with pytest.raises(RuntimeError, match="addon_model_missing") as error:
        OnnxInsightFaceEngine({"models_dir": tmp_path, "addons": ["liveness"]})
    assert str(tmp_path / "addons/liveness.onnx") in str(error.value)
    assert "models addons install liveness" in str(error.value)
    assert not (tmp_path / "addons").exists()


def test_corrupt_addon_error_is_distinct_and_never_silently_replaced(tmp_path):
    path = tmp_path / "addons/liveness.onnx"
    path.parent.mkdir()
    path.write_bytes(b"invalid")
    with pytest.raises(RuntimeError, match="addon_model_invalid") as error:
        require_installed_addon("liveness", tmp_path)
    assert "SHA256 mismatch" in str(error.value)
    assert path.read_bytes() == b"invalid"


def test_installed_addon_is_readable_by_runtime_uid(tmp_path, monkeypatch):
    content = b"verified fixture"
    spec = catalog.AddonArtifact(
        "liveness.onnx", "unused", hashlib.sha256(content).hexdigest(), len(content)
    )
    monkeypatch.setattr(catalog, "ADDON_CATALOG", {"liveness": spec})
    path = tmp_path / "addons/liveness.onnx"
    path.parent.mkdir()
    path.write_bytes(content)
    path.chmod(0o600)
    assert install_addon("liveness", tmp_path) == path
    assert path.stat().st_mode & 0o777 == 0o644
    assert require_installed_addon("liveness", tmp_path) == path
