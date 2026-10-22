from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import release_preflight
from scripts.release_preflight import Preflight


def write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        (
            "git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
            "-c", "tag.gpgsign=false", *args,
        ),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def fixture_repository(
    tmp_path: Path, *, initialize_git: bool = True, commit: bool = True,
) -> Path:
    version = "1.2.3"
    write(
        tmp_path,
        "server/pyproject.toml",
        f'[project]\nname="server"\nversion="{version}"\n',
    )
    write(
        tmp_path,
        "server/sdk/python/pyproject.toml",
        f'[project]\nname="sdk"\nversion="{version}"\nclassifiers=[]\n',
    )
    write(
        tmp_path,
        "server/backend/insightface_server/__init__.py",
        f'__version__ = "{version}"\n',
    )
    write(
        tmp_path,
        "server/sdk/python/src/insightface_server/__init__.py",
        f'__version__ = "{version}"\n',
    )
    write(
        tmp_path,
        "server/sdk/python/src/insightface_server/client.py",
        f'insightface-server-python/{version}\n',
    )
    write(
        tmp_path,
        "server/backend/insightface_server/models/packages.py",
        f"InsightFace-Server-Model-Installer/{version}\n",
    )
    write(tmp_path, "server/Makefile", f"SERVER_VERSION ?= {version}\n")
    for name in ("Dockerfile.cpu", "Dockerfile.cuda12"):
        write(
            tmp_path,
            f"server/docker/{name}",
            "\n".join(
                (
                    'LABEL org.opencontainers.image.source="https://github.com/deepinsight/insightface"',
                    f"ARG INSIGHTFACE_SERVER_VERSION={version}",
                )
            )
            + "\n",
        )
    for variant in ("cpu", "cuda12"):
        image = f"ghcr.io/deepinsight/insightface-server:{version}-{variant}"
        write(
            tmp_path,
            f"server/deploy/compose.{variant}.yml",
            f"server:\n  image: {image}\nmodels:\n  image: {image}\n",
        )
    for name in (
        "README.md",
        "README.zh-CN.md",
        "README.ja.md",
        "README.de.md",
        "README.es.md",
        "README.fr.md",
        "README.ru.md",
        "README.pt.md",
        "README.ko.md",
    ):
        write(
            tmp_path,
            f"server/{name}",
            f"{version} {version}-cpu {version}-cuda12\n",
        )
    write(
        tmp_path,
        "server/docs/openapi.snapshot.json",
        f'{{"info": {{"version": "{version}"}}}}\n',
    )
    if initialize_git:
        git(tmp_path, "init")
        git(tmp_path, "config", "user.email", "test@example.invalid")
        git(tmp_path, "config", "user.name", "Release Test")
        if commit:
            git(tmp_path, "add", ".")
            git(tmp_path, "commit", "-m", "fixture")
    return tmp_path


def checked_report(root: Path, *, relaxed: bool = False) -> dict:
    preflight = Preflight(root, "1.2.3", relaxed=relaxed)
    preflight.run()
    report = preflight.report()
    assert report["schema_version"] == 2
    assert report["mode"] == ("precheck" if relaxed else "release")
    assert report["summary"]["checks"] == 39
    assert all(not check["name"].startswith("git-") for check in report["checks"])
    assert set(report["git"]) == {"head_revision", "dirty"}
    return report


@pytest.mark.parametrize("relaxed", [False, True])
def test_consistent_clean_source_reports_exact_revision(tmp_path: Path, relaxed: bool) -> None:
    root = fixture_repository(tmp_path)
    revision = git(root, "rev-parse", "HEAD")
    report = checked_report(root, relaxed=relaxed)

    assert {check["status"] for check in report["checks"]} == {"pass"}
    assert report["git"] == {"head_revision": revision, "dirty": False}
    assert report["source_revision"] == revision


@pytest.mark.parametrize("relaxed", [False, True])
@pytest.mark.parametrize("change", ["tracked", "untracked"])
def test_dirty_source_passes_without_claiming_exact_revision(
    tmp_path: Path, relaxed: bool, change: str,
) -> None:
    root = fixture_repository(tmp_path)
    revision = git(root, "rev-parse", "HEAD")
    if change == "tracked":
        path = root / "server/README.md"
        path.write_text(path.read_text() + "Local source changes.\n", encoding="utf-8")
    else:
        write(root, "scratch.txt", "untracked source\n")

    report = checked_report(root, relaxed=relaxed)

    assert {check["status"] for check in report["checks"]} == {"pass"}
    assert report["git"] == {"head_revision": revision, "dirty": True}
    assert report["source_revision"] is None


@pytest.mark.parametrize("relaxed", [False, True])
def test_source_archive_without_git_metadata_passes(tmp_path: Path, relaxed: bool) -> None:
    root = fixture_repository(tmp_path, initialize_git=False)
    report = checked_report(root, relaxed=relaxed)

    assert report["summary"]["failures"] == 0
    assert report["git"] == {"head_revision": None, "dirty": None}
    assert report["source_revision"] is None


def test_source_archive_does_not_borrow_parent_repository_revision(tmp_path: Path) -> None:
    parent = fixture_repository(tmp_path)
    root = fixture_repository(parent / "unpacked-source", initialize_git=False)
    # Git itself can discover the parent, but that HEAD does not identify this
    # unpacked source tree and must not be reported as its provenance.
    assert git(root, "rev-parse", "--show-toplevel") == str(parent.resolve())
    report = checked_report(root)

    assert report["summary"]["failures"] == 0
    assert report["git"] == {"head_revision": None, "dirty": None}
    assert report["source_revision"] is None


@pytest.mark.parametrize("problem", ["missing_program", "permission", "timeout"])
def test_git_execution_failure_does_not_block_structural_checks(
    tmp_path: Path, monkeypatch, problem: str,
) -> None:
    root = fixture_repository(tmp_path)

    def unavailable(*args, **kwargs):
        if problem == "missing_program":
            raise FileNotFoundError("git not installed")
        if problem == "permission":
            raise PermissionError("git execution denied")
        raise subprocess.TimeoutExpired("git", 1)

    monkeypatch.setattr(release_preflight.subprocess, "run", unavailable)
    report = checked_report(root)

    assert report["summary"]["failures"] == 0
    assert report["git"] == {"head_revision": None, "dirty": None}
    assert report["source_revision"] is None


@pytest.mark.parametrize("tag_kind", ["lightweight", "another_head"])
def test_existing_release_tags_do_not_gate_container_preflight(
    tmp_path: Path, tag_kind: str,
) -> None:
    root = fixture_repository(tmp_path)
    if tag_kind == "lightweight":
        git(root, "tag", "server-v1.2.3")
    else:
        git(root, "tag", "-a", "server-v1.2.3", "-m", "earlier source")
        write(root, "next.txt", "new source\n")
        git(root, "add", "next.txt")
        git(root, "commit", "-m", "next source")
        assert git(root, "rev-list", "-n", "1", "server-v1.2.3") != git(root, "rev-parse", "HEAD")

    report = checked_report(root)

    assert report["summary"]["failures"] == 0
    assert report["source_revision"] == git(root, "rev-parse", "HEAD")


def test_repository_without_head_does_not_claim_a_revision(tmp_path: Path) -> None:
    root = fixture_repository(tmp_path, commit=False)
    report = checked_report(root)

    assert report["summary"]["failures"] == 0
    assert report["git"]["head_revision"] is None
    assert report["source_revision"] is None


def test_unavailable_status_does_not_claim_a_clean_revision(tmp_path: Path, monkeypatch) -> None:
    root = fixture_repository(tmp_path)
    revision = git(root, "rev-parse", "HEAD")
    original_run = subprocess.run

    def status_fails(command, *args, **kwargs):
        if "status" in command:
            return subprocess.CompletedProcess(command, 128, stdout="", stderr="status unavailable")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(release_preflight.subprocess, "run", status_fails)
    report = checked_report(root)

    assert report["summary"]["failures"] == 0
    assert report["git"] == {"head_revision": revision, "dirty": None}
    assert report["source_revision"] is None


@pytest.mark.parametrize("initialize_git", [False, True])
@pytest.mark.parametrize("relaxed", [False, True])
def test_version_and_structure_errors_still_fail_without_git_gates(
    tmp_path: Path, initialize_git: bool, relaxed: bool,
) -> None:
    root = fixture_repository(tmp_path, initialize_git=initialize_git)
    write(root, "server/sdk/python/src/insightface_server/__init__.py", '__version__ = "9.9.9"\n')
    write(root, "server/deploy/compose.cpu.yml", "server:\n  image: unversioned\n")

    report = checked_report(root, relaxed=relaxed)
    failed = {check["name"] for check in report["checks"] if check["status"] == "fail"}

    assert failed == {"sdk-runtime-version", "compose-cpu-version"}
    assert report["summary"]["failures"] == 2


def test_report_keeps_run_time_git_snapshot(tmp_path: Path, monkeypatch) -> None:
    root = fixture_repository(tmp_path)
    revision = git(root, "rev-parse", "HEAD")
    preflight = Preflight(root, "1.2.3", relaxed=False)
    preflight.run()
    write(root, "later.txt", "changed after preflight\n")

    def unexpected_git(*args, **kwargs):
        pytest.fail("report() must not invoke Git after run() captured the source state")

    monkeypatch.setattr(release_preflight.subprocess, "run", unexpected_git)
    report = preflight.report()

    assert report["git"] == {"head_revision": revision, "dirty": False}
    assert report["source_revision"] == revision
    assert preflight.report()["git"] == report["git"]
