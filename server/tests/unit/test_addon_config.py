from __future__ import annotations

import stat

import pytest
from insightface_server import addon_config
from insightface_server.config import load_server_config


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "server.toml"
    path.write_text(
        '# operator settings\n[inference]\naddons=[] # retained comment\n'
        'liveness_threshold=0.92\n[addons]\nauto_download=[]\n'
    )
    path.chmod(0o640)
    return path


def test_atomic_enable_preserves_other_settings_comments_mode_and_is_idempotent(config_file, monkeypatch):
    with addon_config.liveness_config_lock(config_file):
        addon_config.write_enabled_addons(config_file, ("liveness",))
    saved = load_server_config(config_file)
    assert saved.addons == saved.auto_download_addons == ("liveness",)
    assert saved.liveness_threshold == 0.92
    assert "# operator settings" in config_file.read_text()
    assert "# retained comment" in config_file.read_text()
    assert stat.S_IMODE(config_file.stat().st_mode) == 0o640
    previous = config_file.read_bytes()

    def unexpected_replace(*args):
        pytest.fail("already-enabled config must not be replaced")

    monkeypatch.setattr(addon_config.os, "replace", unexpected_replace)
    with addon_config.liveness_config_lock(config_file):
        addon_config.write_enabled_addons(config_file, ("liveness",))
    assert config_file.read_bytes() == previous


@pytest.mark.parametrize(
    "initial",
    [
        '# empty configuration\n',
        'inference = { addons = [] }\naddons = { auto_download = [] }\n',
        'inference.addons = []\naddons.auto_download = []\n',
    ],
)
def test_enable_accepts_legacy_inline_and_dotted_tables(config_file, initial):
    config_file.write_text(initial)
    with addon_config.liveness_config_lock(config_file):
        addon_config.write_enabled_addons(config_file, ("liveness",))
    saved = load_server_config(config_file)
    assert saved.addons == saved.auto_download_addons == ("liveness",)


def test_shared_lock_is_nonblocking_and_released_after_exception(config_file):
    with pytest.raises(RuntimeError, match="abort job"):
        with addon_config.liveness_config_lock(config_file):
            with pytest.raises(addon_config.AddonConfigError) as exc:
                with addon_config.liveness_config_lock(config_file):
                    pytest.fail("second writer entered the same configuration job")
            assert exc.value.code == "addon_job_in_progress"
            raise RuntimeError("abort job")
    with addon_config.liveness_config_lock(config_file):
        addon_config.write_enabled_addons(config_file, ("liveness",))
    assert load_server_config(config_file).addons == ("liveness",)


def test_concurrent_manual_edit_during_save_is_not_overwritten(config_file, monkeypatch):
    validate = addon_config.load_server_config
    edited = False

    def concurrent_edit(path):
        nonlocal edited
        result = validate(path)
        if path.name.startswith(".server-config-") and not edited:
            edited = True
            config_file.write_text(config_file.read_text().replace("0.92", "0.96"))
        return result

    monkeypatch.setattr(addon_config, "load_server_config", concurrent_edit)
    with addon_config.liveness_config_lock(config_file):
        with pytest.raises(RuntimeError, match="Configuration changed"):
            addon_config.write_enabled_addons(config_file, ("liveness",))
    saved = load_server_config(config_file)
    assert saved.liveness_threshold == 0.96 and saved.addons == ()
    assert not list(config_file.parent.glob(".server-config-*.toml"))


def test_replacement_failure_preserves_old_file_and_cleans_temporary(config_file, monkeypatch):
    previous = config_file.read_bytes()

    def denied(*args):
        raise PermissionError("read-only destination")

    monkeypatch.setattr(addon_config.os, "replace", denied)
    with addon_config.liveness_config_lock(config_file):
        with pytest.raises(PermissionError):
            addon_config.write_enabled_addons(config_file, ("liveness",))
    assert config_file.read_bytes() == previous
    assert not list(config_file.parent.glob(".server-config-*.toml"))


def test_temporary_config_must_validate_before_publication(config_file, monkeypatch):
    previous = config_file.read_bytes()
    validate = addon_config.load_server_config

    def reject_temporary(path):
        if path.name.startswith(".server-config-"):
            raise ValueError("invalid final configuration")
        return validate(path)

    monkeypatch.setattr(addon_config, "load_server_config", reject_temporary)
    with addon_config.liveness_config_lock(config_file):
        with pytest.raises(ValueError, match="invalid final configuration"):
            addon_config.write_enabled_addons(config_file, ("liveness",))
    assert config_file.read_bytes() == previous
    assert not list(config_file.parent.glob(".server-config-*.toml"))


def test_cancelled_job_cannot_publish_after_download(config_file):
    previous = config_file.read_bytes()
    calls = 0

    def stopping():
        nonlocal calls
        calls += 1
        return calls >= 2

    with addon_config.liveness_config_lock(config_file):
        with pytest.raises(RuntimeError, match="shutdown interrupted"):
            addon_config.write_enabled_addons(config_file, ("liveness",), cancelled=stopping)
    assert config_file.read_bytes() == previous
    assert not list(config_file.parent.glob(".server-config-*.toml"))


def test_writer_rechecks_permissions_after_download(config_file):
    previous = config_file.read_bytes()
    config_file.chmod(0o444)
    try:
        with pytest.raises(addon_config.AddonConfigError) as exc:
            addon_config.write_enabled_addons(config_file, ("liveness",))
        assert exc.value.code == "config_not_writable"
        assert config_file.read_bytes() == previous
    finally:
        config_file.chmod(0o640)
