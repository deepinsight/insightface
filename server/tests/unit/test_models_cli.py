from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import pytest
from insightface_server import models_cli
from insightface_server.licensing import ModelLicense


class _NonInteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return False


def test_install_requires_explicit_noninteractive_acceptance(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(models_cli.sys, "stdin", _NonInteractiveInput())
    result = models_cli.main(
        ["--models-dir", str(tmp_path), "install", "buffalo_l"]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert "non-commercial research use only" in captured.out
    assert "add --accept-license" in captured.err


def test_successful_install_prints_license_after_result(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(models_cli, "install_package", lambda *_args, **_kwargs: "installed")
    result = models_cli.main(
        [
            "--models-dir",
            str(tmp_path),
            "install",
            "buffalo_l",
            "--accept-license",
        ]
    )
    captured = capsys.readouterr()
    assert result == 0
    assert "is installed and verified" in captured.out
    assert captured.out.rfind("LICENSE NOTICE") > captured.out.find("is installed and verified")
    assert "Commercial use requires a separate license" in captured.out
    assert "https://www.insightface.ai" in captured.out


def test_info_reports_pinned_source_and_hashes(capsys) -> None:
    assert models_cli.main(["info", "buffalo_l"]) == 0
    output = capsys.readouterr().out
    assert "/releases/download/model-zoo/buffalo_l.zip" in output
    assert "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f" in output
    assert "det_10g.onnx SHA-256" in output
    assert "w600k_r50.onnx SHA-256" in output


def test_list_reports_every_supported_package(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    calls = 0

    def installed_package(_models_dir: Path) -> str | None:
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(models_cli, "installed_package_name", installed_package)
    assert models_cli.main(["--models-dir", str(tmp_path), "list"]) == 0
    assert calls == 1
    output = capsys.readouterr().out
    assert output.splitlines()[0] == "NAME\tSTATUS"
    for name in models_cli.PACKAGES:
        assert f"{name}\tnot installed" in output
    assert "v0.7" not in output



@pytest.mark.parametrize(
    ("name", "detector", "recognizer"),
    (
        ("buffalo_m", "det_2.5g.onnx", "w600k_r50.onnx"),
        ("buffalo_s", "det_500m.onnx", "w600k_mbf.onnx"),
        ("buffalo_sc", "det_500m.onnx", "w600k_mbf.onnx"),
        ("antelopev2", "scrfd_10g_bnkps.onnx", "glintr100.onnx"),
        ("raccoon_s", "det_10g_wo.onnx", "w600k_mbf.onnx"),
        ("raccoon_l", "det_10g_wo.onnx", "w600k_r50.onnx"),
    ),
)
def test_info_reports_new_catalog_packages(
    name: str, detector: str, recognizer: str, capsys
) -> None:
    assert models_cli.main(["info", name]) == 0
    output = capsys.readouterr().out
    assert f"/releases/download/model-zoo/{name}.zip" in output
    assert f"{detector} SHA-256" in output
    assert f"{recognizer} SHA-256" in output
    assert "Commercial use requires a separate license" in output


def test_verify_prints_installed_package_license(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        models_cli,
        "verify_installed",
        lambda _models_dir: (
            "buffalo_l",
            (
                {"file": "det_10g.onnx", "sha256": "a" * 64},
                {"file": "w600k_r50.onnx", "sha256": "b" * 64},
            ),
            ModelLicense(
                license_id="buffalo_l-public-v1",
                issuer="InsightFace",
                model_id="buffalo_l",
                grant="non-commercial",
                valid_from=datetime(2026, 7, 22, tzinfo=UTC),
                valid_until=None,
            ),
        ),
    )
    assert models_cli.main(["verify", "buffalo_l"]) == 0
    output = capsys.readouterr().out
    assert "Installed package: buffalo_l" in output
    assert "LICENSE VERIFIED" in output
    assert "Issuer: InsightFace" in output
    assert "Model ID: buffalo_l" in output
    assert "Signature: VALID" in output
    assert "Commercial use: NOT PERMITTED" in output


@pytest.fixture
def liveness_install(tmp_path, monkeypatch):
    import hashlib

    from insightface.addons import catalog

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = config_dir / "server.toml"
    config.write_text(
        '# Keep this operator comment\n[inference]\naddons=[] # runtime addons\n'
        'liveness_mode="observe"\nliveness_threshold=0.91\n'
        '[addons]\nauto_download=[] # installer addons\n'
        '[detection]\nthreshold=0.63\n'
    )
    root = tmp_path / "models"
    events = []
    model_bytes = b"test liveness artifact"
    artifact = catalog.AddonArtifact(
        "liveness.onnx", "https://example.invalid/liveness.onnx",
        hashlib.sha256(model_bytes).hexdigest(), len(model_bytes),
    )
    monkeypatch.setattr(catalog, "ADDON_CATALOG", {"liveness": artifact})
    monkeypatch.delenv("INSIGHTFACE_CONFIG_FILE", raising=False)
    monkeypatch.setattr(
        models_cli, "installed_package_name",
        lambda path: "buffalo_l" if (path / "base-cache").exists() else None,
    )

    def install_base(package, path):
        events.append("base")
        path.mkdir(exist_ok=True)
        cached = (path / "base-cache").exists()
        (path / "base-cache").touch()
        return "already_installed" if cached else "installed"

    monkeypatch.setattr(models_cli, "install_package", install_base)

    class Download:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield model_bytes

    def get(*args, **kwargs):
        events.append("download")
        return Download()

    monkeypatch.setattr(catalog.requests, "get", get)
    argv = ["--config-file", str(config), "--models-dir", str(root), "install", "buffalo_l", "--accept-license"]
    return config, root, events, argv


def test_enable_liveness_installs_verifies_preserves_config_and_reuses_cache(liveness_install, capsys):
    from insightface_server.config import load_server_config

    config, root, events, argv = liveness_install
    assert models_cli.main([*argv, "--enable-liveness"]) == 0
    result = capsys.readouterr()
    assert "Liveness is installed and configured for the next Server start." in result.out
    assert "Restart a running Server" in result.out
    assert "this command does not restart it" in result.out
    assert events == ["base", "download"]
    saved = load_server_config(config)
    assert saved.addons == saved.auto_download_addons == ("liveness",)
    assert saved.liveness_mode == "observe" and saved.liveness_threshold == 0.91
    assert saved.detection.threshold == 0.63
    assert "# Keep this operator comment" in config.read_text()
    assert "# runtime addons" in config.read_text() and "# installer addons" in config.read_text()
    before = config.read_bytes()
    assert (root / "addons/liveness.onnx").is_file()
    assert models_cli.main([*argv, "--enable-liveness"]) == 0
    assert events == ["base", "download", "base"]
    assert config.read_bytes() == before


def test_enable_liveness_also_saves_when_base_and_addon_are_already_cached(liveness_install):
    from insightface_server.config import load_server_config

    config, root, events, argv = liveness_install
    assert models_cli.main(argv) == 0
    models_cli.install_addon("liveness", root)
    assert load_server_config(config).addons == ()
    assert models_cli.main([*argv, "--enable-liveness"]) == 0
    assert events == ["base", "download", "base"]
    assert load_server_config(config).addons == ("liveness",)


def test_enable_liveness_deduplicates_configured_addon(liveness_install, monkeypatch):
    config, _root, events, argv = liveness_install
    config.write_text('[inference]\naddons=[]\n[addons]\nauto_download=["liveness"]\n')
    installs = []
    original = models_cli.install_addon

    def install(name, root):
        installs.append(name)
        return original(name, root)

    monkeypatch.setattr(models_cli, "install_addon", install)
    assert models_cli.main([*argv, "--enable-liveness"]) == 0
    assert installs == ["liveness"] and events == ["base", "download"]


def test_install_without_flag_does_not_write_config_or_download_unrequested_addon(liveness_install, monkeypatch):
    config, root, events, argv = liveness_install
    original = config.read_bytes()
    config.chmod(0o444)

    def forbidden(*args, **kwargs):
        pytest.fail("an ordinary install must not write configuration")

    monkeypatch.setattr(models_cli, "write_enabled_addons", forbidden)
    try:
        assert models_cli.main(argv) == 0
        assert events == ["base"]
        assert config.read_bytes() == original
        assert not (root / "addons").exists()
    finally:
        config.chmod(0o644)


@pytest.mark.parametrize("invalid", ["unset", "missing", "symlink", "file_readonly", "directory_readonly", "file_mount", "invalid_toml"])
def test_enable_liveness_preflights_before_any_model_install(liveness_install, monkeypatch, invalid, capsys):
    from insightface_server import addon_config

    config, root, events, argv = liveness_install
    if invalid == "unset":
        argv = argv[2:]
    elif invalid == "missing":
        config.unlink()
    elif invalid == "symlink":
        target = config.with_name("actual.toml")
        config.rename(target)
        config.symlink_to(target)
    elif invalid == "file_readonly":
        config.chmod(0o444)
    elif invalid == "directory_readonly":
        config.parent.chmod(0o555)
    elif invalid == "file_mount":
        monkeypatch.setattr(addon_config, "file_mount", lambda path: path == config)
    else:
        config.write_text("[inference\n")
    try:
        assert models_cli.main([*argv, "--enable-liveness"]) == 2
        assert events == [] and not root.exists()
        assert "configured for the next Server start" not in capsys.readouterr().out
    finally:
        config.parent.chmod(0o755)
        if config.exists():
            config.chmod(0o644)


@pytest.mark.parametrize("command", [["verify"], ["info", "buffalo_l"], ["addons", "install", "liveness"], ["list"]])
def test_enable_liveness_flag_is_only_valid_for_base_install(command):
    with pytest.raises(SystemExit) as exc:
        models_cli.main([*command, "--enable-liveness"])
    assert exc.value.code == 2


def test_enable_liveness_addon_failure_preserves_config_and_retry_reuses_base(liveness_install, monkeypatch, capsys):
    config, root, events, argv = liveness_install
    original = config.read_bytes()
    install = models_cli.install_addon

    def fail(*args):
        raise OSError("download failed")

    monkeypatch.setattr(models_cli, "install_addon", fail)
    assert models_cli.main([*argv, "--enable-liveness"]) == 2
    captured = capsys.readouterr()
    assert "Configuration was not changed by this command" in captured.err
    assert "configured for the next Server start" not in captured.out
    assert config.read_bytes() == original and (root / "base-cache").exists()
    monkeypatch.setattr(models_cli, "install_addon", install)
    assert models_cli.main([*argv, "--enable-liveness"]) == 0
    assert events == ["base", "base", "download"]


def test_enable_liveness_failed_final_verification_cannot_save_config(liveness_install, monkeypatch):
    config, _root, _events, argv = liveness_install
    original = config.read_bytes()

    def fail(*args):
        raise RuntimeError("SHA256 mismatch")

    monkeypatch.setattr(models_cli, "require_installed_addon", fail)
    assert models_cli.main([*argv, "--enable-liveness"]) == 2
    assert config.read_bytes() == original


def test_enable_liveness_config_failure_is_nonzero_and_retry_reuses_models(liveness_install, monkeypatch, capsys):
    config, _root, events, argv = liveness_install
    original = config.read_bytes()
    save = models_cli.write_enabled_addons

    def fail(*args):
        raise OSError("atomic replacement denied")

    monkeypatch.setattr(models_cli, "write_enabled_addons", fail)
    assert models_cli.main([*argv, "--enable-liveness"]) == 2
    captured = capsys.readouterr()
    assert "Could not save the liveness configuration" in captured.err
    assert "can be reused" in captured.err
    assert "configured for the next Server start" not in captured.out
    assert config.read_bytes() == original
    monkeypatch.setattr(models_cli, "write_enabled_addons", save)
    assert models_cli.main([*argv, "--enable-liveness"]) == 0
    assert events == ["base", "download", "base"]


def test_enable_liveness_busy_config_lock_stops_before_download(liveness_install, capsys):
    from insightface_server.addon_config import liveness_config_lock

    config, root, events, argv = liveness_install
    with liveness_config_lock(config):
        assert models_cli.main([*argv, "--enable-liveness"]) == 2
    assert "wait and retry" in capsys.readouterr().err
    assert events == [] and not root.exists()


def test_enable_liveness_preserves_operator_edit_during_download(liveness_install, monkeypatch):
    from insightface_server.config import load_server_config

    config, _root, _events, argv = liveness_install
    install = models_cli.install_addon

    def edited(*args):
        config.write_text(config.read_text().replace("0.91", "0.95"))
        return install(*args)

    monkeypatch.setattr(models_cli, "install_addon", edited)
    assert models_cli.main([*argv, "--enable-liveness"]) == 0
    saved = load_server_config(config)
    assert saved.liveness_threshold == 0.95 and saved.addons == ("liveness",)
