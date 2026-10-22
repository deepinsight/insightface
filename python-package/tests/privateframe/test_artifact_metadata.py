import subprocess
from types import SimpleNamespace

import pytest

from insightface.app.privateframe import artifacts


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("git"), PermissionError("git"), subprocess.TimeoutExpired("git", 5)],
)
def test_optional_git_metadata_does_not_require_git(monkeypatch, tmp_path, error):
    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(artifacts.subprocess, "run", unavailable)
    assert artifacts.git_version(tmp_path) == {"commit": None, "dirty": None}


def test_git_metadata_preserves_available_revision(monkeypatch, tmp_path):
    def available(command, **kwargs):
        value = "abc123\n" if command[1] == "rev-parse" else " M changed.py\n"
        return SimpleNamespace(returncode=0, stdout=value)

    monkeypatch.setattr(artifacts.subprocess, "run", available)
    assert artifacts.git_version(tmp_path) == {"commit": "abc123", "dirty": True}
