#!/usr/bin/env python3
"""Validate that every public InsightFace Server release surface agrees.

The normal mode is intentionally strict and is intended for the owner-operated
manual release process. ``--relaxed`` keeps all structural checks enabled while
turning the dirty-worktree condition expected during local development into a
warning.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
README_NAMES = (
    "README.md",
    "README.zh-CN.md",
    "README.ja.md",
    "README.de.md",
    "README.es.md",
    "README.fr.md",
    "README.ru.md",
    "README.pt.md",
    "README.ko.md",
)


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


class Preflight:
    def __init__(self, root: Path, version: str, *, relaxed: bool) -> None:
        self.root = root.resolve()
        self.server = self.root / "server"
        self.version = version
        self.relaxed = relaxed
        self.checks: list[Check] = []

    def pass_(self, name: str, detail: str) -> None:
        self.checks.append(Check(name, "pass", detail))

    def warn(self, name: str, detail: str) -> None:
        self.checks.append(Check(name, "warning", detail))

    def fail(self, name: str, detail: str) -> None:
        self.checks.append(Check(name, "fail", detail))

    def require(self, name: str, condition: bool, detail: str) -> None:
        if condition:
            self.pass_(name, detail)
        else:
            self.fail(name, detail)

    def _text(self, relative: str) -> str:
        return (self.root / relative).read_text(encoding="utf-8")

    def _toml(self, relative: str) -> dict[str, Any]:
        with (self.root / relative).open("rb") as handle:
            return tomllib.load(handle)

    def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", *args),
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
        )

    def _check_exact_version(self, name: str, actual: object) -> None:
        self.require(
            name,
            actual == self.version,
            f"expected {self.version!r}; found {actual!r}",
        )

    def run(self) -> list[Check]:
        self.require(
            "version-format",
            bool(SEMVER.fullmatch(self.version)),
            f"release version is {self.version!r}",
        )
        self._check_package_versions()
        self._check_container_versions()
        self._check_document_versions()
        self._check_license_metadata()
        self._check_git_state()
        return self.checks

    def _check_package_versions(self) -> None:
        server_project = self._toml("server/pyproject.toml")["project"]
        sdk_project = self._toml("server/sdk/python/pyproject.toml")["project"]
        self._check_exact_version("server-pyproject-version", server_project["version"])
        self._check_exact_version("sdk-pyproject-version", sdk_project["version"])

        backend_init = self._text("server/backend/insightface_server/__init__.py")
        sdk_init = self._text(
            "server/sdk/python/src/insightface_server/__init__.py"
        )
        client = self._text(
            "server/sdk/python/src/insightface_server/client.py"
        )
        model_packages = self._text(
            "server/backend/insightface_server/models/packages.py"
        )
        self.require(
            "backend-runtime-version",
            f'__version__ = "{self.version}"' in backend_init,
            "backend __version__ matches",
        )
        self.require(
            "sdk-runtime-version",
            f'__version__ = "{self.version}"' in sdk_init,
            "SDK __version__ matches",
        )
        self.require(
            "sdk-user-agent-version",
            f"insightface-server-python/{self.version}" in client,
            "SDK User-Agent matches",
        )
        self.require(
            "model-installer-user-agent-version",
            f"InsightFace-Server-Model-Installer/{self.version}" in model_packages,
            "model installer User-Agent matches",
        )

    def _check_container_versions(self) -> None:
        makefile = self._text("server/Makefile")
        self.require(
            "make-version",
            re.search(
                rf"^SERVER_VERSION\s*\?=\s*{re.escape(self.version)}$",
                makefile,
                re.MULTILINE,
            )
            is not None,
            "Makefile SERVER_VERSION matches",
        )
        for variant, dockerfile in (
            ("cpu", "server/docker/Dockerfile.cpu"),
            ("cuda12", "server/docker/Dockerfile.cuda12"),
        ):
            text = self._text(dockerfile)
            self.require(
                f"docker-{variant}-version",
                f"ARG INSIGHTFACE_SERVER_VERSION={self.version}" in text,
                f"{dockerfile} default version matches",
            )
            self.require(
                f"docker-{variant}-source-label",
                'org.opencontainers.image.source="https://github.com/deepinsight/insightface"'
                in text,
                f"{dockerfile} links the source repository",
            )

        for variant, compose in (
            ("cpu", "server/deploy/compose.cpu.yml"),
            ("cuda12", "server/deploy/compose.cuda12.yml"),
        ):
            expected = (
                f"ghcr.io/deepinsight/insightface-server:"
                f"{self.version}-{variant}"
            )
            text = self._text(compose)
            self.require(
                f"compose-{variant}-version",
                text.count(expected) == 2,
                f"{compose} has exactly two {expected} references",
            )
            self.require(
                f"compose-{variant}-no-latest",
                "insightface-server:latest" not in text,
                f"{compose} does not use latest",
            )

    def _check_document_versions(self) -> None:
        for name in README_NAMES:
            text = self._text(f"server/{name}")
            self.require(
                f"{name}-version",
                self.version in text,
                f"{name} names {self.version}",
            )
            self.require(
                f"{name}-variant-tags",
                f"{self.version}-cpu" in text and f"{self.version}-cuda12" in text,
                f"{name} names both immutable image variants",
            )

        snapshot = json.loads(
            self._text("server/docs/openapi.snapshot.json")
        )
        self._check_exact_version(
            "openapi-version", snapshot.get("info", {}).get("version")
        )

    def _check_license_metadata(self) -> None:
        for relative in (
            "server/pyproject.toml",
            "server/sdk/python/pyproject.toml",
        ):
            project = self._toml(relative)["project"]
            self.require(
                f"{relative}-legacy-license-field",
                "license" not in project,
                f"{relative} has no legacy project.license declaration",
            )
            classifiers = project.get("classifiers", [])
            self.require(
                f"{relative}-mit-classifier",
                "License :: OSI Approved :: MIT License" not in classifiers,
                f"{relative} has no MIT classifier",
            )

    def _check_git_state(self) -> None:
        revision = self._git("rev-parse", "--verify", "HEAD")
        sha = revision.stdout.strip()
        self.require(
            "git-revision",
            revision.returncode == 0 and bool(re.fullmatch(r"[0-9a-f]{40}", sha)),
            f"source revision is {sha or 'unavailable'}",
        )

        status = self._git("status", "--porcelain=v1", "--untracked-files=all")
        dirty_lines = [line for line in status.stdout.splitlines() if line.strip()]
        if dirty_lines and self.relaxed:
            self.warn(
                "git-clean",
                f"relaxed precheck accepts {len(dirty_lines)} dirty paths; "
                "release rejects them",
            )
        else:
            self.require(
                "git-clean",
                status.returncode == 0 and not dirty_lines,
                f"worktree has {len(dirty_lines)} dirty paths",
            )

        tag = f"server-v{self.version}"
        existing_tag = self._git("show-ref", "--verify", "--quiet", f"refs/tags/{tag}")
        if existing_tag.returncode != 0:
            self.pass_("git-tag-idempotent", f"release tag {tag} is absent")
            self.pass_("git-tag-annotated", f"release tag {tag} is not created yet")
        else:
            tag_revision = self._git("rev-list", "-n", "1", tag).stdout.strip()
            tag_type = self._git("cat-file", "-t", tag).stdout.strip()
            self.require(
                "git-tag-idempotent",
                tag_revision == sha,
                f"existing release tag points to {tag_revision or 'unavailable'}",
            )
            self.require(
                "git-tag-annotated",
                tag_type == "tag",
                f"existing release tag object type is {tag_type or 'unavailable'}",
            )

    def report(self) -> dict[str, Any]:
        failures = sum(check.status == "fail" for check in self.checks)
        warnings = sum(check.status == "warning" for check in self.checks)
        revision = self._git("rev-parse", "--verify", "HEAD").stdout.strip()
        return {
            "schema_version": 1,
            "version": self.version,
            "mode": "precheck" if self.relaxed else "release",
            "source_revision": revision,
            "generated_at": datetime.now(UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "summary": {
                "checks": len(self.checks),
                "failures": failures,
                "warnings": warnings,
            },
            "checks": [asdict(check) for check in self.checks],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (defaults to the root containing server/)",
    )
    parser.add_argument(
        "--relaxed",
        action="store_true",
        help="warn instead of failing for a dirty tree",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preflight = Preflight(args.root, args.version, relaxed=args.relaxed)
    preflight.run()
    report = preflight.report()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if report["summary"]["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
