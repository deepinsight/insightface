"""Exercise shipped health probes and documented setup without Docker or sudo."""

from __future__ import annotations

import json
import os
import re
import shlex
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from email.message import Message
from io import BytesIO
from pathlib import Path
from urllib.response import addinfourl

import pytest

SERVER_DIR = Path(__file__).resolve().parents[2]
SETUP_GUIDES = sorted(SERVER_DIR.glob("README*.md")) + sorted(
    (SERVER_DIR / "docs").glob("user-guide*.md")
)


@pytest.mark.parametrize("variant", ["cpu", "cuda12"])
@pytest.mark.parametrize("outcome", [200, 503, "connection_error"])
def test_image_healthcheck_bypasses_proxies_and_preserves_failures(
    monkeypatch: pytest.MonkeyPatch, variant: str, outcome: int | str,
) -> None:
    dockerfile = (SERVER_DIR / "docker" / f"Dockerfile.{variant}").read_text()
    healthcheck = re.search(r"HEALTHCHECK[^\n]+\\\n\s+CMD (\[[^\n]+\])", dockerfile)
    assert healthcheck is not None
    command = json.loads(healthcheck[1])
    assert command[:2] == ["python", "-c"]
    assert len(command) == 3

    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(variable, "http://unreachable-proxy.invalid:3128")
    for variable in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(variable, "")
    monkeypatch.delenv("REQUEST_METHOD", raising=False)
    # Do not let macOS system proxy exclusions hide a regression on Linux.
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: False)
    monkeypatch.setattr(urllib.request, "_opener", None)

    requests = []

    def offline_http_open(self, request):
        requests.append(request)
        assert request.host == "127.0.0.1:8080"
        assert request.selector == "/v1/health"
        assert not request.has_proxy()
        assert request.timeout == 3
        if outcome == "connection_error":
            raise urllib.error.URLError("offline simulated connection refusal")
        response = addinfourl(BytesIO(b'{"status":"ready"}'), Message(), request.full_url, outcome)
        response.msg = "OK" if outcome == 200 else "Service Unavailable"
        return response

    def no_network(*args, **kwargs):
        raise AssertionError("healthcheck regression tests must stay offline")

    # Keep urllib's real proxy and HTTP status processing; replace only transport.
    monkeypatch.setattr(urllib.request.HTTPHandler, "http_open", offline_http_open)
    monkeypatch.setattr(socket, "create_connection", no_network)
    if outcome == 200:
        exec(command[2], {})
    else:
        expected = urllib.error.URLError if outcome == "connection_error" else urllib.error.HTTPError
        with pytest.raises(expected):
            exec(command[2], {})
    assert len(requests) == 1


@pytest.mark.parametrize("guide", SETUP_GUIDES, ids=lambda path: path.name)
@pytest.mark.parametrize("existing_models", [False, True], ids=["empty-checkout", "existing-directory"])
def test_initial_setup_does_not_require_host_permission_changes(
    tmp_path: Path, guide: Path, existing_models: bool,
) -> None:
    markdown = guide.read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", markdown, re.DOTALL)
    setup = next(block for block in blocks if "docker compose" in block)
    directory = tmp_path / "server" / ".models"
    if existing_models:
        directory.mkdir(parents=True)
        directory.chmod(0o700)

    # Execute the documented shell block, replacing Docker with a recorder.
    # Installation now provisions storage through Compose; no host chmod,
    # chown, supplementary group, or exported user identity may be required.
    probe = tmp_path / "record_compose.py"
    probe.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "assert sys.argv[1] == 'compose'\n"
        "assert 'INSIGHTFACE_MODELS_UID' not in os.environ\n"
        "assert 'INSIGHTFACE_MODELS_GID' not in os.environ\n"
        "assert not Path('server/.models/addons').exists()\n"
        "with Path('compose-calls.jsonl').open('a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    script = "umask 077\n"
    for command in ("sudo", "chmod", "chown", "chgrp", "mkdir", "id"):
        script += f"{command}() {{ echo 'unexpected host preparation: {command}' >&2; return 1; }}\n"
    script += "curl() { :; }\n"
    script += f"docker() {{ {shlex.quote(sys.executable)} {shlex.quote(str(probe))} \"$@\"; }}\n"
    script += setup
    environment = os.environ.copy()
    environment.pop("INSIGHTFACE_MODELS_UID", None)
    environment.pop("INSIGHTFACE_MODELS_GID", None)
    result = subprocess.run(
        ["bash", "-euc", script], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, f"{guide.name}: {result.stderr}"
    calls = [json.loads(line) for line in (tmp_path / "compose-calls.jsonl").read_text().splitlines()]
    assert any("pull" in call for call in calls)
    assert any("models" in call and "install" in call for call in calls)
    assert directory.exists() is existing_models
    if existing_models:
        assert directory.stat().st_mode & 0o777 == 0o700
