from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


PYTHON_PACKAGE = Path(__file__).resolve().parents[1]


def _run_without_site_packages(
    script: str, telemetry_value: str | None = None
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PYTHON_PACKAGE)
    environment.pop("ORT_DISABLE_TELEMETRY", None)
    if telemetry_value is not None:
        environment["ORT_DISABLE_TELEMETRY"] = telemetry_value
    return subprocess.run(
        [sys.executable, "-S", "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        env=environment,
    )


@pytest.mark.parametrize("telemetry_value", [None, "0"], ids=["unset", "enabled"])
@pytest.mark.parametrize(
    ("entrypoint", "watched_import"),
    [
        ("import insightface", "onnxruntime"),
        ("import insightface.model_zoo", "onnxruntime"),
        (
            'runpy.run_module("insightface.app.privateframe", run_name="__main__")',
            "onnxruntime",
        ),
        (
            'from insightface_privateframe_bootstrap import main; main(["describe"])',
            "insightface",
        ),
        (
            'from insightface_privateframe_bootstrap import main; main(["--version"])',
            "insightface",
        ),
    ],
    ids=["package", "submodule", "module_cli", "console_cli", "console_version"],
)
def test_telemetry_is_disabled_before_inference_import(
    entrypoint: str, watched_import: str, telemetry_value: str | None
) -> None:
    # Stop at the dependency boundary: no native runtime is loaded, and the
    # console bootstrap must establish its policy before InsightFace can help.
    completed = _run_without_site_packages(
        f"""
        import importlib.abc
        import json
        import os
        import runpy
        import sys

        class ImportObserved(BaseException):
            pass

        class ObserveImport(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == {watched_import!r}:
                    raise ImportObserved({{
                        "module": fullname,
                        "telemetry": os.environ.get("ORT_DISABLE_TELEMETRY"),
                    }})
                return None

        sys.meta_path.insert(0, ObserveImport())
        try:
            {entrypoint}
        except ImportObserved as exc:
            print(json.dumps(exc.args[0]))
        else:
            raise AssertionError("The expected inference import never occurred")
        """,
        telemetry_value,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "module": watched_import,
        "telemetry": "1",
    }


def test_missing_runtime_still_has_actionable_import_error() -> None:
    completed = _run_without_site_packages(
        """
        import importlib.abc
        import json
        import os
        import sys

        class MissingRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "onnxruntime":
                    raise ModuleNotFoundError(
                        "No module named 'onnxruntime'", name="onnxruntime"
                    )
                return None

        sys.meta_path.insert(0, MissingRuntime())
        try:
            import insightface
        except ImportError as exc:
            print(json.dumps({
                "message": str(exc),
                "missing_module": exc.__cause__.name,
                "telemetry": os.environ.get("ORT_DISABLE_TELEMETRY"),
            }))
        else:
            raise AssertionError("The missing runtime was not reported")
        """
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert "python -m pip install onnxruntime" in payload["message"]
    assert payload["missing_module"] == "onnxruntime"
    assert payload["telemetry"] == "1"
