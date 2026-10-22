from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import warnings
import zipfile

import pytest


PACKAGE_NAME = "insightface"
PACKAGE_VERSION = "2.0"
HELPER_PATH = (
    Path(__file__).resolve().parents[2]
    / "packaging"
    / "pypi"
    / "release_artifacts.py"
)


@pytest.fixture(scope="module")
def release_artifacts():
    spec = importlib.util.spec_from_file_location(
        "insightface_release_artifacts_test", HELPER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata(
    *,
    name: str = PACKAGE_NAME,
    version: str = PACKAGE_VERSION,
    requires_python: str | None = ">=3.10",
) -> bytes:
    headers = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    if requires_python is not None:
        headers.append(f"Requires-Python: {requires_python}")
    return ("\n".join(headers) + "\n\n").encode()


def _write_artifact(
    directory: Path,
    kind: str,
    *,
    filename: str | None = None,
    metadata: bytes | None = None,
    metadata_copies: int = 1,
) -> Path:
    payload = _metadata() if metadata is None else metadata
    if kind == "wheel":
        path = directory / (filename or "insightface-2.0-py3-none-any.whl")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("insightface/__init__.py", "__version__ = '2.0'\n")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Duplicate name:")
                for _ in range(metadata_copies):
                    archive.writestr("insightface-2.0.dist-info/METADATA", payload)
    else:
        path = directory / (filename or "insightface-2.0.tar.gz")
        with tarfile.open(path, "w:gz") as archive:
            for _ in range(metadata_copies):
                entry = tarfile.TarInfo("insightface-2.0/PKG-INFO")
                entry.size = len(payload)
                archive.addfile(entry, io.BytesIO(payload))
    return path


@pytest.fixture
def artifact_pair(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    return dist_dir, _write_artifact(dist_dir, "wheel"), _write_artifact(dist_dir, "sdist")


def _select(release_artifacts, dist_dir: Path):
    return release_artifacts.select_release_artifacts(
        dist_dir, PACKAGE_NAME, PACKAGE_VERSION
    )


def test_selects_only_valid_pair_in_wheel_then_sdist_order(
    release_artifacts, artifact_pair
):
    dist_dir, wheel, sdist = artifact_pair

    assert _select(release_artifacts, dist_dir) == (wheel, sdist)


def test_accepts_canonical_name_and_equivalent_version(
    release_artifacts, artifact_pair
):
    dist_dir, wheel, sdist = artifact_pair
    for kind in ("wheel", "sdist"):
        _write_artifact(
            dist_dir, kind, metadata=_metadata(name="InsightFace", version="2.0.0")
        )

    assert release_artifacts.select_release_artifacts(
        dist_dir, "InsightFace", "2.0.0"
    ) == (wheel, sdist)


@pytest.mark.parametrize(
    ("kind", "filename"),
    [
        ("wheel", "foreign-2.0-py3-none-any.whl"),
        ("wheel", "insightface-1.9-py3-none-any.whl"),
        ("sdist", "foreign-2.0.tar.gz"),
        ("sdist", "insightface-1.9.tar.gz"),
        ("wheel", "insightface.whl"),
        ("sdist", "insightface.tar.gz"),
    ],
)
def test_rejects_foreign_stale_or_malformed_artifact_filenames(
    release_artifacts, artifact_pair, kind, filename
):
    dist_dir, wheel, sdist = artifact_pair
    (wheel if kind == "wheel" else sdist).rename(dist_dir / filename)

    with pytest.raises(ValueError):
        _select(release_artifacts, dist_dir)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("headers", [{"name": "foreign"}, {"version": "1.9"}])
def test_rejects_mismatched_embedded_identity(
    release_artifacts, artifact_pair, kind, headers
):
    dist_dir, _, _ = artifact_pair
    _write_artifact(dist_dir, kind, metadata=_metadata(**headers))

    with pytest.raises(ValueError):
        _select(release_artifacts, dist_dir)


def test_rejects_multiple_wheels_even_when_both_match_the_release(
    release_artifacts, artifact_pair
):
    dist_dir, _, _ = artifact_pair
    _write_artifact(
        dist_dir, "wheel", filename="insightface-2.0-1-py3-none-any.whl"
    )

    with pytest.raises(ValueError):
        _select(release_artifacts, dist_dir)


@pytest.mark.parametrize("missing", ["wheel", "sdist", "both"])
def test_rejects_incomplete_release(release_artifacts, artifact_pair, missing):
    dist_dir, wheel, sdist = artifact_pair
    if missing in ("wheel", "both"):
        wheel.unlink()
    if missing in ("sdist", "both"):
        sdist.unlink()

    with pytest.raises(ValueError):
        _select(release_artifacts, dist_dir)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "requires_python", [None, ">=3.9", ">=3.9.1", ">=3.11", "invalid"]
)
def test_rejects_missing_or_incompatible_python_requirement(
    release_artifacts, artifact_pair, kind, requires_python
):
    dist_dir, _, _ = artifact_pair
    _write_artifact(
        dist_dir, kind, metadata=_metadata(requires_python=requires_python)
    )

    with pytest.raises(ValueError):
        _select(release_artifacts, dist_dir)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_rejects_malformed_archives(release_artifacts, artifact_pair, kind):
    dist_dir, wheel, sdist = artifact_pair
    (wheel if kind == "wheel" else sdist).write_bytes(b"not an archive\n")

    with pytest.raises(ValueError):
        _select(release_artifacts, dist_dir)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("metadata_copies", [0, 2])
def test_rejects_missing_or_duplicate_archive_metadata(
    release_artifacts, artifact_pair, kind, metadata_copies
):
    dist_dir, _, _ = artifact_pair
    _write_artifact(dist_dir, kind, metadata_copies=metadata_copies)

    with pytest.raises(ValueError):
        _select(release_artifacts, dist_dir)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_rejects_unexpected_dist_entries(release_artifacts, artifact_pair, kind):
    dist_dir, _, _ = artifact_pair
    unexpected = dist_dir / "leftover"
    if kind == "file":
        unexpected.write_text("stale build output\n")
    else:
        unexpected.mkdir()

    with pytest.raises(ValueError):
        _select(release_artifacts, dist_dir)


def test_rejects_artifact_symlink(release_artifacts, artifact_pair):
    dist_dir, wheel, _ = artifact_pair
    outside_wheel = dist_dir.parent / wheel.name
    wheel.rename(outside_wheel)
    try:
        wheel.symlink_to(outside_wheel)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlink creation is unavailable or not permitted: {exc}")

    with pytest.raises(ValueError):
        _select(release_artifacts, dist_dir)


def _run_cli(dist_dir: Path):
    return subprocess.run(
        [
            sys.executable,
            str(HELPER_PATH),
            "--dist-dir",
            str(dist_dir),
            "--name",
            PACKAGE_NAME,
            "--version",
            PACKAGE_VERSION,
        ],
        capture_output=True,
        check=False,
    )


def test_cli_emits_only_explicit_absolute_artifact_paths(artifact_pair):
    dist_dir, wheel, sdist = artifact_pair

    result = _run_cli(dist_dir)

    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == f"{wheel.resolve()}\0{sdist.resolve()}\0".encode()


def test_cli_failure_does_not_emit_upload_paths(artifact_pair):
    dist_dir, wheel, _ = artifact_pair
    wheel.unlink()

    result = _run_cli(dist_dir)

    assert result.returncode != 0
    assert result.stdout == b""
    assert result.stderr
