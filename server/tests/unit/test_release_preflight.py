from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.release_preflight import Preflight


def write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def fixture_repository(tmp_path: Path) -> Path:
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
    subprocess.run(("git", "init"), cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"),
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Release Test"),
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(("git", "add", "."), cwd=tmp_path, check=True)
    subprocess.run(
        ("git", "commit", "-m", "fixture"),
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return tmp_path


def statuses(preflight: Preflight) -> dict[str, str]:
    return {check.name: check.status for check in preflight.run()}


def test_formal_preflight_accepts_consistent_clean_release(tmp_path: Path) -> None:
    root = fixture_repository(tmp_path)
    result = statuses(Preflight(root, "1.2.3", relaxed=False))

    assert set(result.values()) == {"pass"}


def test_relaxed_precheck_warns_for_dirty_tree(tmp_path: Path) -> None:
    root = fixture_repository(tmp_path)
    write(root, "scratch.txt", "dirty\n")

    result = statuses(Preflight(root, "1.2.3", relaxed=True))

    assert result["git-clean"] == "warning"
    assert "fail" not in result.values()


def test_formal_preflight_rejects_version_drift(tmp_path: Path) -> None:
    root = fixture_repository(tmp_path)
    write(
        root,
        "server/sdk/python/src/insightface_server/__init__.py",
        '__version__ = "9.9.9"\n',
    )

    result = statuses(Preflight(root, "1.2.3", relaxed=False))

    assert result["sdk-runtime-version"] == "fail"
    assert result["git-clean"] == "fail"


def test_formal_preflight_accepts_same_source_release_tag_for_resume(
    tmp_path: Path,
) -> None:
    root = fixture_repository(tmp_path)
    subprocess.run(
        ("git", "tag", "-a", "server-v1.2.3", "-m", "release"),
        cwd=root,
        check=True,
    )

    result = statuses(Preflight(root, "1.2.3", relaxed=False))

    assert result["git-tag-idempotent"] == "pass"
    assert result["git-tag-annotated"] == "pass"
    assert "fail" not in result.values()


def test_formal_preflight_rejects_lightweight_release_tag(tmp_path: Path) -> None:
    root = fixture_repository(tmp_path)
    subprocess.run(
        ("git", "tag", "server-v1.2.3"),
        cwd=root,
        check=True,
    )

    result = statuses(Preflight(root, "1.2.3", relaxed=False))

    assert result["git-tag-idempotent"] == "pass"
    assert result["git-tag-annotated"] == "fail"


def test_formal_preflight_rejects_release_tag_from_another_source(
    tmp_path: Path,
) -> None:
    root = fixture_repository(tmp_path)
    subprocess.run(
        ("git", "tag", "-a", "server-v1.2.3", "-m", "release"),
        cwd=root,
        check=True,
    )
    write(root, "next.txt", "new source\n")
    subprocess.run(("git", "add", "next.txt"), cwd=root, check=True)
    subprocess.run(
        ("git", "commit", "-m", "next source"),
        cwd=root,
        check=True,
        capture_output=True,
    )

    result = statuses(Preflight(root, "1.2.3", relaxed=False))

    assert result["git-tag-idempotent"] == "fail"
