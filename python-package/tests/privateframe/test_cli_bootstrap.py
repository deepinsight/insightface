from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import builtins
from pathlib import Path

import insightface_privateframe_bootstrap as bootstrap


PYTHON_PACKAGE = Path(__file__).resolve().parents[2]


def _run_with_blocked_import(
    module: str, argv: list[str]
) -> subprocess.CompletedProcess[str]:
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import json

        class Blocked(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == {module!r}:
                    raise ModuleNotFoundError(
                        "No module named {module}",
                        name={module!r},
                    )
                return None

        import sys
        sys.meta_path.insert(0, Blocked())
        from insightface_privateframe_bootstrap import main
        raise SystemExit(main({argv!r}))
        """
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PYTHON_PACKAGE)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def test_describe_still_works_when_optional_pyav_is_not_installed() -> None:
    completed = _run_with_blocked_import("av", ["describe"])

    assert completed.returncode == 0
    assert completed.stdout.count("\n") == 1
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["command"] == "describe"


def test_doctor_reports_missing_pyav_instead_of_crashing_at_startup() -> None:
    completed = _run_with_blocked_import("av", ["doctor"])

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["ready"] is False
    pyav_check = next(
        check for check in payload["checks"] if check["name"] == "runtime.pyav"
    )
    assert pyav_check["ok"] is False


def test_bootstrap_structures_a_missing_configuration_dependency() -> None:
    completed = _run_with_blocked_import("yaml", ["doctor"])

    assert completed.returncode == 1
    assert completed.stdout.count("\n") == 1
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert payload["command"] == "doctor"
    assert payload["error"]["stage"] == "startup"
    assert payload["error"]["code"] == "missing_dependency"
    assert payload["error"]["dependency"] == "yaml"
    assert "insightface[privateframe]" in payload["error"]["hints"][0]


def test_bootstrap_structures_interrupt_during_dependency_import(
    monkeypatch,
    capsys,
) -> None:
    original_import = builtins.__import__

    def interrupted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "insightface.app.privateframe.cli":
            raise KeyboardInterrupt
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", interrupted_import)

    exit_code = bootstrap.main(["describe"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 130
    assert captured.out.count("\n") == 1
    assert payload["command"] == "describe"
    assert payload["error"]["code"] == "cancelled"
    assert payload["error"]["stage"] == "startup"
