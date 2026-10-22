from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

SERVER_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_DIR = SERVER_DIR.parent


def _read(relative: str) -> str:
    return (REPOSITORY_DIR / relative).read_text(encoding="utf-8")


def test_dockerfiles_are_pinned_root_and_model_free() -> None:
    cpu = _read("server/docker/Dockerfile.cpu")
    cuda = _read("server/docker/Dockerfile.cuda12")
    cpu_lock = _read("server/requirements.cpu.lock")
    cuda_lock = _read("server/requirements.cuda12.lock")

    assert re.search(r"^FROM python:3\.11-[^\s]+@sha256:[0-9a-f]{64}$", cpu, re.MULTILINE)
    assert re.search(
        r"^FROM nvcr\.io/nvidia/cuda:12\.9\.1-runtime-ubuntu22\.04"
        r"@sha256:[0-9a-f]{64}$",
        cuda,
        re.MULTILINE,
    )
    assert "onnxruntime==1.27.0" in cpu_lock
    assert "onnxruntime-gpu" in cuda_lock
    assert "onnxruntime_gpu-1.27.0-cp311" in cuda_lock
    assert "sha256=" in cuda_lock
    assert "libcudnn9-cuda-12=9.24.0.43-1" in cuda

    for dockerfile in (cpu, cuda):
        assert "USER 0:0" in dockerfile
        assert 'VOLUME ["/data"]' in dockerfile
        assert "HEALTHCHECK" in dockerfile
        assert "/v1/health" in dockerfile
        assert "INSIGHTFACE_CONFIG_FILE=/etc/insightface/server.toml" in dockerfile
        assert "COPY server/config/server.toml /etc/insightface/server.toml" in dockerfile
        assert (
            "COPY server/LICENSING.md /opt/insightface/server/LICENSING.md"
            in dockerfile
        )
        assert "server/LICENSE " not in dockerfile
        assert "LICENSE-NOTICE.md" not in dockerfile
        assert "COPY server/README" not in dockerfile
        assert "COPY server/docs/api*.md /opt/insightface/server/docs/" in dockerfile
        assert "COPY server/docs/user-guide*.md /opt/insightface/server/docs/" in dockerfile
        assert (
            "COPY server/docs/maintainer-guide.md "
            "/opt/insightface/server/docs/maintainer-guide.md"
        ) in dockerfile
        assert "COPY server/docs/images /opt/insightface/server/docs/images" in dockerfile
        assert "COPY models" not in dockerfile
        assert "latest" not in dockerfile.lower()


def test_offline_license_public_key_is_packaged_but_private_issuer_is_excluded() -> None:
    public_key = (
        REPOSITORY_DIR
        / "python-package"
        / "insightface"
        / "model_zoo"
        / "trusted_keys"
        / "insightface-model-license-public-ed25519.pem"
    )
    assert public_key.read_text(encoding="ascii").startswith("-----BEGIN PUBLIC KEY-----")
    for relative in ("server/docker/Dockerfile.cpu", "server/docker/Dockerfile.cuda12"):
        dockerfile = _read(relative)
        assert "COPY server/backend /opt/insightface/server/backend" in dockerfile
        assert "COPY python-package/insightface/model_zoo " in dockerfile
    dockerignore = _read(".dockerignore")
    assert ".private/" in dockerignore
    assert "**/*.pem" in dockerignore
    assert (
        "!python-package/insightface/model_zoo/trusted_keys/*.pem"
        in dockerignore
    )
    assert "!server/backend/insightface_server/licensing/trusted_keys/*.pem" not in dockerignore
    assert ".private/" in _read(".gitignore")


def test_compose_uses_one_writable_model_mount_and_persists_data() -> None:
    variants = {
        "server/deploy/compose.cpu.yml": ("0.3.1-cpu", "18097"),
        "server/deploy/compose.cuda12.yml": ("0.3.1-cuda12", "18098"),
    }
    for relative, (image_tag, host_port) in variants.items():
        compose = _read(relative)
        assert (
            f"image: ghcr.io/deepinsight/insightface-server:{image_tag}" in compose
        )
        assert f'      - "{host_port}:8080"' in compose
        assert "x-models-path: &models-path ../.models" in compose
        assert "source: *models-path" in compose
        assert "INSIGHTFACE_CPU_IMAGE" not in compose
        assert "INSIGHTFACE_CUDA_IMAGE" not in compose
        assert compose.count("INSIGHTFACE_MODELS_DIR: /models") == 1
        assert "INSIGHTFACE_PORT" not in compose
        assert "target: /data" in compose
        assert "target: /models" in compose
        assert compose.count("target: /etc/insightface\n") == 2
        assert "target: /etc/insightface/server.toml" not in compose
        assert "source: ../config/server.toml" not in compose
        assert "read_only: true" in compose
        assert compose.count('user: "0:0"') == 2
        assert "  models:\n" in compose
        assert "profiles: [tools]" in compose
        assert 'entrypoint: ["python", "-m", "insightface_server.models_cli"]' in compose
        assert "INSIGHTFACE_MODELS_UID" not in compose
        assert "INSIGHTFACE_MODELS_GID" not in compose
        assert "group_add:" not in compose
        assert "cap_drop:" not in compose
        assert "privileged:" not in compose
        assert compose.count("no-new-privileges:true") == 2
        assert "x-addons-path" not in compose
        assert "target: /models/addons" not in compose
        assert "healthcheck:\n      disable: true" in compose
        server_service = compose.split("\n  models:\n", 1)[0]
        assert "source: ../config\n        target: /etc/insightface" in server_service
        assert (
            "target: /models\n        bind:\n          create_host_path: true"
            in server_service
        )
        assert "source: ../config\n        target: /etc/insightface\n        bind:\n          create_host_path: false" in server_service
        assert "HTTPS_PROXY:" in server_service
        models_service = compose.split("\n  models:\n", 1)[1]
        assert "target: /models\n        bind:\n          create_host_path: true" in models_service
        assert "source: ../config\n        target: /etc/insightface\n        bind:\n          create_host_path: false" in models_service
        assert "gpus: all" not in models_service
        for variable in (
            "INSIGHTFACE_DEFAULT_THRESHOLD",
            "INSIGHTFACE_COLLECTION_DEFAULT_SEARCH_PROFILE",
            "INSIGHTFACE_COLLECTION_DEFAULT_CAPACITY_ROWS",
            "INSIGHTFACE_COLLECTION_MAX_CAPACITY_ROWS",
            "INSIGHTFACE_COLLECTION_DEFAULT_MAX_FACES_PER_PERSON",
            "INSIGHTFACE_COLLECTION_DEFAULT_LOAD_POLICY",
            "INSIGHTFACE_SEARCH_DEVICE_ID",
            "INSIGHTFACE_SEARCH_TOPK_MODE",
            "INSIGHTFACE_SEARCH_BUILD_BATCH_ROWS",
        ):
            assert variable in compose


def test_build_context_excludes_unrelated_repository_data() -> None:
    dockerignore = _read(".dockerignore")
    assert dockerignore.splitlines()[0].startswith("#")
    assert "**" in dockerignore.splitlines()
    assert "!server/**" in dockerignore
    for required in (
        "!python-package/insightface/app/**",
        "!python-package/insightface/addons/**",
        "!python-package/insightface/model_zoo/**",
        "!python-package/insightface/utils/**",
        "!python-package/insightface/data/__init__.py",
        "!python-package/insightface/data/image.py",
        "!python-package/insightface/data/pickle_object.py",
        "!python-package/insightface/thirdparty/__init__.py",
    ):
        assert required in dockerignore
    assert "!python-package/insightface/**" not in dockerignore
    assert "!python-package/insightface/data/images" not in dockerignore
    assert "**/*.onnx" in dockerignore
    assert "**/__pycache__/" in dockerignore


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _request(url: str, *, method: str = "GET", body: dict[str, object] | None = None):
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method=method,
        headers={"content-type": "application/json"} if payload is not None else {},
    )
    # Never send local CI traffic through a developer or runner proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def _wait_ready(base_url: str, *, container: str) -> None:
    deadline = time.monotonic() + 60
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status, body = _request(f"{base_url}/v1/health")
            if status == 200 and body.get("status") == "ready":
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(0.5)
    logs = _docker("logs", container, check=False).stdout
    pytest.fail(f"mock container did not become ready: {last_error}\n{logs}")


@pytest.mark.docker
def test_mock_container_restart_preserves_unique_named_volume() -> None:
    if os.getenv("INSIGHTFACE_RUN_DOCKER_TESTS") != "1":
        pytest.skip(
            "Docker integration is opt-in; set INSIGHTFACE_RUN_DOCKER_TESTS=1"
        )
    image = os.getenv("INSIGHTFACE_TEST_CPU_IMAGE")
    if not image:
        pytest.fail("INSIGHTFACE_TEST_CPU_IMAGE is required for Docker integration")
    if _docker("image", "inspect", image, check=False).returncode != 0:
        pytest.fail(f"Docker image does not exist: {image}")

    unique = uuid.uuid4().hex
    container = f"insightface-simple-ci-{unique}"
    volume = f"insightface-simple-ci-data-{unique}"
    created_container = False
    created_volume = False
    try:
        _docker("volume", "create", volume)
        created_volume = True
        result = _docker(
            "run",
            "--detach",
            "--name",
            container,
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev",
            "--mount",
            f"source={volume},target=/data",
            "--env",
            "INSIGHTFACE_INFERENCE_MODE=mock",
            "--env",
            "INSIGHTFACE_EXECUTION_PROVIDER=CPUExecutionProvider",
            "--env",
            "INSIGHTFACE_AUTH_ENABLED=false",
            "--publish",
            "127.0.0.1::8080",
            "--entrypoint",
            "python",
            image,
            "-m",
            "uvicorn",
            "insightface_server.app:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--no-access-log",
        )
        created_container = result.returncode == 0
        port_output = _docker("port", container, "8080/tcp").stdout.strip()
        port = int(port_output.rsplit(":", 1)[1])
        base_url = f"http://127.0.0.1:{port}"
        _wait_ready(base_url, container=container)

        status, created = _request(
            f"{base_url}/v1/collections",
            method="POST",
            body={
                "id": f"persistent-{unique[:12]}",
                "name": "CI persistence",
                "threshold": 0.68,
            },
        )
        assert status == 201
        collection_id = created["collection"]["id"]

        _docker("restart", container)
        # Docker may allocate a new ephemeral host port when restarting.
        port_output = _docker("port", container, "8080/tcp").stdout.strip()
        port = int(port_output.rsplit(":", 1)[1])
        base_url = f"http://127.0.0.1:{port}"
        _wait_ready(base_url, container=container)
        status, persisted = _request(f"{base_url}/v1/collections/{collection_id}")
        assert status == 200
        assert persisted["collection"]["id"] == collection_id
        assert persisted["collection"]["person_count"] == 0
        assert persisted["collection"]["face_count"] == 0
    finally:
        # Names are generated by this test. Never enumerate or prune shared state.
        if created_container:
            _docker("rm", "--force", container, check=False)
        if created_volume:
            _docker("volume", "rm", volume, check=False)
