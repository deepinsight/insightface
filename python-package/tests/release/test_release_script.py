from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


BASH_PATH = shutil.which("bash")
pytestmark = pytest.mark.skipif(
    os.name == "nt" or BASH_PATH is None,
    reason="Release script integration requires POSIX Bash and executable shebang stubs",
)


FAKE_PYTHON = r'''
import hashlib, io, json, os, pathlib, sys, tarfile, zipfile

args = sys.argv[1:]
artifacts = [pathlib.Path(arg) for arg in args if arg.endswith((".whl", ".tar.gz"))]
event = {"args": args, "artifacts": {
    str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts
}}
with open(os.environ["RELEASE_TEST_LOG"], "a") as log:
    log.write(json.dumps(event) + "\n")
if args == ["-c", "import sys; print(sys.executable)"]:
    print(sys.executable)
elif args[0] == "-":
    source = sys.stdin.read()
    if 'required = ["build", "twine"]' in source:
        pass
    elif "name_match =" in source or 'destination = Path("dist")' in source:
        sys.argv = args
        exec(compile(source, "<release script>", "exec"))
    else:
        raise SystemExit("Unexpected inline script; no network operations permitted")
elif args[:2] == ["-m", "pytest"]:
    pass
elif args[:2] == ["-m", "build"]:
    mode = os.environ["RELEASE_TEST_BUILD_MODE"]
    if mode == "fail":
        raise SystemExit(17)
    destination = pathlib.Path(args[args.index("--outdir") + 1])
    destination.mkdir(parents=True)
    version = "1.9" if mode == "bad_metadata" else "2.0"
    metadata = (
        "Metadata-Version: 2.1\nName: insightface\n"
        f"Version: {version}\nRequires-Python: >=3.10\n\n"
    ).encode()
    with zipfile.ZipFile(destination / "insightface-2.0-py3-none-any.whl", "w") as archive:
        archive.writestr("insightface-2.0.dist-info/METADATA", metadata)
    with tarfile.open(destination / "insightface-2.0.tar.gz", "w:gz") as archive:
        entry = tarfile.TarInfo("insightface-2.0/PKG-INFO")
        entry.size = len(metadata)
        archive.addfile(entry, io.BytesIO(metadata))
elif args[0] == "packaging/pypi/release_artifacts.py":
    os.execv(sys.executable, [sys.executable, *args])
elif args[0] == "packaging/pypi/artifact_smoke.py" and args[1] == "inspect":
    pass
elif args[:2] == ["-m", "twine"] and args[2] in ("check", "upload"):
    pass  # Record the intended action only; never invoke Twine.
else:
    raise SystemExit(f"Unexpected Python invocation: {args!r}")
'''


@pytest.fixture
def release_run(tmp_path):
    package = tmp_path / "package"
    helpers = package / "packaging" / "pypi"
    helpers.mkdir(parents=True)
    source = Path(__file__).resolve().parents[2] / "packaging" / "pypi"
    for filename in ("build_upload_pypi.sh", "release_artifacts.py"):
        shutil.copyfile(source / filename, helpers / filename)
    (package / "setup.py").write_text("name='insightface'\n")
    (package / "insightface").mkdir()
    (package / "insightface" / "__init__.py").write_text("__version__ = '2.0'\n")
    dist = package / "dist"
    dist.mkdir()
    for filename in ("foreign-9.0-py3-none-any.whl", "insightface-1.9.tar.gz"):
        (dist / filename).write_bytes(b"stale unrelated output")
    fake_python = tmp_path / "fake python"
    fake_python.write_text(f"#!{sys.executable}\n" + FAKE_PYTHON)
    fake_python.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    work = tmp_path / "work"
    work.mkdir()

    def run(*, dry_run=False, build_mode="valid"):
        command = [
            BASH_PATH, str(helpers / "build_upload_pypi.sh"),
            "--python", str(fake_python), "--allow-dirty", "--skip-existing-check",
            "--no-clean",
        ]
        if dry_run:
            command.append("--dry-run")
        result = subprocess.run(
            command, cwd=package, input="upload insightface 2.0 to pypi\n",
            text=True, capture_output=True, timeout=20, check=False,
            env={**os.environ, "TMPDIR": str(work), "RELEASE_TEST_LOG": str(log),
                 "RELEASE_TEST_BUILD_MODE": build_mode},
        )
        events = [json.loads(line) for line in log.read_text().splitlines()]
        return result, events, dist

    return run


def test_no_clean_checks_and_uploads_only_fresh_artifacts(release_run):
    result, events, dist = release_run()

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"==> Python: {sys.executable}\n" in result.stdout
    assert result.stderr == ""
    twine = [event for event in events if event["args"][:2] == ["-m", "twine"]]
    assert [event["args"][2] for event in twine] == ["check", "upload"]
    expected = {"insightface-2.0-py3-none-any.whl", "insightface-2.0.tar.gz"}
    assert twine[0]["artifacts"] == twine[1]["artifacts"]
    for event in twine:
        assert {Path(path).name for path in event["artifacts"]} == expected
        assert all(Path(path).parent != dist for path in event["artifacts"])
    for path, digest in twine[1]["artifacts"].items():
        assert hashlib.sha256((dist / Path(path).name).read_bytes()).hexdigest() == digest
    assert (dist / "foreign-9.0-py3-none-any.whl").read_bytes() == b"stale unrelated output"
    assert (dist / "insightface-1.9.tar.gz").read_bytes() == b"stale unrelated output"


@pytest.mark.parametrize("build_mode", ["fail", "bad_metadata"])
def test_failed_or_invalid_build_aborts_before_upload(release_run, build_mode):
    result, events, dist = release_run(build_mode=build_mode)

    assert result.returncode != 0
    assert not any(event["args"][:2] == ["-m", "twine"] for event in events)
    assert not (dist / "insightface-2.0-py3-none-any.whl").exists()


def test_dry_run_executes_full_suite_and_never_uploads(release_run):
    result, events, dist = release_run(dry_run=True)

    assert result.returncode == 0, result.stdout + result.stderr
    commands = [event["args"] for event in events]
    assert ["-m", "pytest", "-q", "tests"] in commands
    assert any(args[:2] == ["-m", "build"] for args in commands)
    assert any(args[:3] == ["-m", "twine", "check"] for args in commands)
    assert not any(args[:3] == ["-m", "twine", "upload"] for args in commands)
    assert (dist / "insightface-2.0-py3-none-any.whl").is_file()
