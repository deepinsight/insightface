from __future__ import annotations

import fcntl
import hashlib
import threading
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from insightface.addons import catalog
from insightface_server import addon_config, addon_management, models_cli
from insightface_server.app import create_app
from insightface_server.config import Settings, load_server_config

MODEL = b"synthetic verified addon fixture"


@pytest.fixture
def preparation(make_settings, tmp_path, monkeypatch):
    config = tmp_path / "configuration" / "server.toml"
    config.parent.mkdir()
    config.write_text(
        '# Keep this deployment comment\n[inference]\naddons = [] # selected addons\n'
        'liveness_mode = "observe"\nliveness_threshold = 0.91\n'
        'liveness_on_registration = false\n[addons]\nauto_download = []\n'
        '[detection]\nthreshold = 0.63\n',
        encoding="utf-8",
    )
    config.chmod(0o664)
    settings = make_settings(config_file=config)
    settings.models_dir.mkdir(parents=True)
    spec = catalog.AddonArtifact(
        "liveness.onnx", "https://example.invalid/liveness.onnx",
        hashlib.sha256(MODEL).hexdigest(), len(MODEL),
    )
    monkeypatch.setattr(catalog, "ADDON_CATALOG", {"liveness": spec})
    calls = []

    class Download:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield MODEL

    def get(*args, **kwargs):
        calls.append(args[0])
        return Download()

    monkeypatch.setattr(catalog.requests, "get", get)
    return settings, calls


def completed(client, *, headers=None):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get("/v1/addons/liveness", headers=headers)
        assert response.status_code == 200, response.text
        status = response.json()
        if status["state"] != "downloading":
            return status
        time.sleep(0.01)
    pytest.fail("Addon preparation did not complete")


def test_prepare_verified_download_saves_only_settings_and_requires_restart(preparation, monkeypatch):
    settings, calls = preparation
    with TestClient(create_app(settings)) as client:
        initial = client.get("/v1/addons/liveness").json()
        assert initial["enabled"] is initial["configured_enabled"] is initial["installed"] is False
        assert initial["can_enable"] is True
        assert initial["unavailable_code"] is initial["unavailable_reason"] is None
        assert calls == []
        response = client.post("/v1/addons/liveness/enable", json={})
        assert response.status_code == 202, response.text
        status = completed(client)
        assert status["state"] == "ready"
        assert status["installed"] is status["configured_enabled"] is status["restart_required"] is True
        assert status["enabled"] is False
        assert status["unavailable_code"] is None
        assert client.get("/v1/system").json()["safe_config"]["addons"] == []
    saved = load_server_config(settings.config_file)
    assert saved.addons == saved.auto_download_addons == ("liveness",)
    assert saved.liveness_mode == "observe"
    assert saved.liveness_threshold == 0.91
    assert saved.liveness_on_registration is False
    assert saved.detection.threshold == 0.63
    assert "# Keep this deployment comment" in settings.config_file.read_text()
    assert "# selected addons" in settings.config_file.read_text()
    assert settings.config_file.stat().st_mode & 0o777 == 0o664
    assert len(calls) == 1
    monkeypatch.setenv("INSIGHTFACE_CONFIG_FILE", str(settings.config_file))
    assert Settings.from_env().addons == ("liveness",)
    # A fresh manager reconstructs pending config even if the prior process
    # exited, and clears the restart flag once the new startup uses it.
    assert addon_management.LivenessManager(settings, enabled=False).status()["restart_required"]
    assert not addon_management.LivenessManager(settings, enabled=True).status()["restart_required"]


def test_cached_verified_model_is_reused_without_network(preparation):
    settings, calls = preparation
    path = settings.models_dir / "addons/liveness.onnx"
    path.parent.mkdir(mode=0o775)
    path.write_bytes(MODEL)
    path.chmod(0o444)
    with TestClient(create_app(settings)) as client:
        assert client.get("/v1/addons/liveness").json()["installed"]
        assert client.post("/v1/addons/liveness/enable", json={}).status_code == 202
        assert completed(client)["restart_required"]
    assert calls == []
    assert path.stat().st_mode & 0o777 == 0o444


def test_download_failure_keeps_configuration_and_hides_proxy_credentials(preparation, monkeypatch):
    settings, _ = preparation
    original = settings.config_file.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("Proxy http://private-user:private-password@proxy.invalid failed")

    monkeypatch.setattr(catalog.requests, "get", fail)
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/addons/liveness/enable", json={}).status_code == 202
        status = completed(client)
    assert status["error"]["code"] == "addon_download_failed"
    assert "private-password" not in str(status)
    assert status["configured_enabled"] is False
    assert settings.config_file.read_bytes() == original


def test_config_save_failure_preserves_disabled_config_and_download_for_retry(preparation, monkeypatch):
    settings, calls = preparation
    original = settings.config_file.read_bytes()
    real_replace = addon_config.os.replace

    def fail(source, target):
        if target == settings.config_file:
            raise PermissionError("read-only bind mount")
        return real_replace(source, target)

    monkeypatch.setattr(addon_config.os, "replace", fail)
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/addons/liveness/enable", json={}).status_code == 202
        status = completed(client)
        assert status["error"]["code"] == "addon_config_save_failed"
        assert status["installed"] is True
        assert settings.config_file.read_bytes() == original
        monkeypatch.setattr(addon_config.os, "replace", real_replace)
        assert client.post("/v1/addons/liveness/enable", json={}).status_code == 202
        assert completed(client)["restart_required"]
    assert len(calls) == 1
    assert not list(settings.config_file.parent.glob(".server-config-*.toml"))


def test_duplicate_post_shares_job_and_latest_unrelated_config_edits_survive(preparation, monkeypatch):
    settings, calls = preparation
    started, release = threading.Event(), threading.Event()
    original = addon_management.install_addon

    def paused(*args):
        started.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(addon_management, "install_addon", paused)
    with TestClient(create_app(settings)) as client:
        try:
            assert client.post("/v1/addons/liveness/enable", json={}).status_code == 202
            assert started.wait(2)
            response = client.post("/v1/addons/liveness/enable", json={})
            assert response.status_code == 202
            assert response.json()["state"] == "downloading"
            settings.config_file.write_text(settings.config_file.read_text().replace("0.91", "0.95"))
        finally:
            release.set()
        assert completed(client)["state"] == "ready"
    assert len(calls) == 1
    assert load_server_config(settings.config_file).liveness_threshold == 0.95


@pytest.mark.parametrize("restriction", ["file", "directory", "models", "missing", "single_bind"])
def test_readonly_or_missing_management_paths_report_actionable_status(preparation, monkeypatch, restriction):
    settings, calls = preparation
    if restriction == "file":
        settings.config_file.chmod(0o444)
    elif restriction == "directory":
        settings.config_file.parent.chmod(0o555)
    elif restriction == "models":
        settings.models_dir.chmod(0o555)
    elif restriction == "missing":
        settings = replace(settings, config_file=None)
    else:
        monkeypatch.setattr(addon_config.os.path, "ismount", lambda path: path == settings.config_file)
    try:
        with TestClient(create_app(settings)) as client:
            status = client.get("/v1/addons/liveness").json()
            assert status["can_enable"] is False
            assert status["unavailable_reason"]
            assert status["unavailable_code"] == {
                "file": "config_not_writable", "directory": "config_not_writable",
                "models": "addon_directory_not_writable", "missing": "config_file_missing",
                "single_bind": "config_file_mount",
            }[restriction]
            assert client.post("/v1/addons/liveness/enable", json={}).status_code == 409
        assert calls == []
    finally:
        if settings.config_file:
            settings.config_file.parent.chmod(0o755)
            settings.config_file.chmod(0o644)
        settings.models_dir.chmod(0o755)


def test_corrupt_existing_model_never_overwritten(preparation):
    settings, calls = preparation
    path = settings.models_dir / "addons/liveness.onnx"
    path.parent.mkdir()
    path.write_bytes(b"corrupt")
    with TestClient(create_app(settings)) as client:
        status = client.get("/v1/addons/liveness").json()
        assert status["error"]["code"] == "addon_model_invalid"
        assert status["unavailable_code"] == "addon_model_invalid"
        assert status["can_enable"] is False
        assert client.post("/v1/addons/liveness/enable", json={}).status_code == 409
    assert path.read_bytes() == b"corrupt"
    assert calls == []


def test_unavailable_codes_cover_invalid_configuration_and_shutdown(preparation):
    settings, calls = preparation
    manager = addon_management.LivenessManager(settings, enabled=False)
    settings.config_file.write_text("[inference\n", encoding="utf-8")
    assert manager.status()["unavailable_code"] == "addon_config_invalid"
    settings.config_file.write_text("[inference]\naddons = []\n", encoding="utf-8")
    target = settings.config_file.with_name("actual.toml")
    settings.config_file.rename(target)
    settings.config_file.symlink_to(target)
    assert manager.status()["unavailable_code"] == "config_file_not_regular"
    settings.config_file.unlink()
    target.rename(settings.config_file)
    manager._stopping.set()
    status = manager.status()
    assert status["unavailable_code"] == "server_stopping"
    assert status["unavailable_reason"]
    assert not status["can_enable"]
    assert calls == []


def test_model_root_readonly_with_writable_addon_mount_is_supported(preparation):
    settings, _ = preparation
    (settings.models_dir / "addons").mkdir(mode=0o775)
    settings.models_dir.chmod(0o555)
    try:
        with TestClient(create_app(settings)) as client:
            assert client.get("/v1/addons/liveness").json()["can_enable"]
            assert client.post("/v1/addons/liveness/enable", json={}).status_code == 202
            assert completed(client)["restart_required"]
    finally:
        settings.models_dir.chmod(0o755)


def test_other_process_config_lock_prevents_a_second_preparation(preparation):
    settings, calls = preparation
    original = settings.config_file.read_bytes()
    lock_path = settings.config_file.parent / ".liveness-management.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with TestClient(create_app(settings)) as client:
            assert client.post("/v1/addons/liveness/enable", json={}).status_code == 202
            status = completed(client)
            assert status["error"]["code"] == "addon_job_in_progress"
    assert calls == []
    assert settings.config_file.read_bytes() == original


def test_shutdown_during_download_cannot_enable_config(preparation, monkeypatch):
    settings, calls = preparation
    original_config = settings.config_file.read_bytes()
    started, release = threading.Event(), threading.Event()
    install = addon_management.install_addon

    def paused(*args):
        started.set()
        assert release.wait(5)
        return install(*args)

    monkeypatch.setattr(addon_management, "install_addon", paused)
    timer = None
    try:
        with TestClient(create_app(settings)) as client:
            assert client.post("/v1/addons/liveness/enable", json={}).status_code == 202
            assert started.wait(2)
            manager = client.app.state.liveness_manager
            # Release the download only after lifespan shutdown has begun.
            def finish_during_shutdown():
                assert manager._stopping.wait(3)
                release.set()

            timer = threading.Thread(target=finish_during_shutdown)
            timer.start()
        assert manager._task.done()
    finally:
        release.set()
        if timer is not None:
            timer.join(timeout=5)
    assert len(calls) == 1
    assert (settings.models_dir / "addons/liveness.onnx").read_bytes() == MODEL
    assert settings.config_file.read_bytes() == original_config


def test_management_uses_existing_auth_and_rejects_cross_origin_json(preparation):
    settings, calls = preparation
    settings = replace(settings, auth_enabled=True, startup_api_key="test-management-key")
    token = {"Authorization": "Bearer test-management-key"}
    with TestClient(create_app(settings)) as client:
        assert client.get("/v1/addons/liveness").status_code == 401
        assert client.post("/v1/addons/liveness/enable", json={}).status_code == 401
        assert client.get("/v1/addons/liveness", headers=token).status_code == 200
        assert client.post("/v1/addons/liveness/enable", headers={**token, "Origin": "https://hostile.invalid"}, json={}).status_code == 403
        assert client.post("/v1/addons/liveness/enable", headers={**token, "Origin": "http://["}, json={}).status_code == 403
        assert client.post("/v1/addons/liveness/enable", headers=token, content="{}").status_code == 415
        assert client.post("/v1/addons/liveness/enable", headers=token, json={"url": "https://hostile.invalid"}).status_code == 400
        assert calls == []
        assert client.post("/v1/addons/liveness/enable", headers={**token, "Origin": "http://testserver"}, json={}).status_code == 202
        assert completed(client, headers=token)["restart_required"]


def test_explicit_cors_origin_is_allowed_and_openapi_documents_management(preparation):
    settings, _ = preparation
    settings = replace(settings, cors_origins=("https://console.example",))
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/addons/liveness/enable", headers={"Origin": "https://console.example"}, json={})
        assert response.status_code == 202
        assert completed(client)["restart_required"]
        spec = client.get("/openapi.json").json()
        operation = spec["paths"]["/v1/addons/liveness/enable"]["post"]
        assert "202" in operation["responses"]
        assert operation["requestBody"]["required"]
        assert "disabled by default" in spec["info"]["description"]


def test_cli_enable_job_and_web_share_the_same_config_lock(preparation, monkeypatch):
    settings, calls = preparation
    started, release = threading.Event(), threading.Event()
    install = models_cli.install_addon
    monkeypatch.setattr(models_cli, "installed_package_name", lambda root: "buffalo_l")
    monkeypatch.setattr(models_cli, "install_package", lambda *args: "already_installed")

    def paused(*args):
        started.set()
        assert release.wait(5)
        return install(*args)

    monkeypatch.setattr(models_cli, "install_addon", paused)
    result = []
    worker = threading.Thread(target=lambda: result.append(models_cli.main([
        "--config-file", str(settings.config_file),
        "--models-dir", str(settings.models_dir),
        "install", "buffalo_l", "--enable-liveness",
    ])))
    try:
        with TestClient(create_app(settings)) as client:
            worker.start()
            assert started.wait(2)
            assert client.post("/v1/addons/liveness/enable", json={}).status_code == 202
            status = completed(client)
            assert status["error"]["code"] == "addon_job_in_progress"
            assert calls == []
            release.set()
            worker.join(timeout=5)
            assert not worker.is_alive() and result == [0]
            status = client.get("/v1/addons/liveness").json()
            assert status["enabled"] is False
            assert status["configured_enabled"] is status["restart_required"] is True
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(timeout=5)
    assert len(calls) == 1
