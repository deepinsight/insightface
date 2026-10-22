"""Validate the wheel/sdist pair produced by one InsightFace release build."""

from __future__ import annotations

import argparse
from email import policy
from email.parser import BytesParser
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import zipfile

from packaging.utils import (
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import Version


def _read_metadata(path: Path) -> bytes:
    try:
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                members = [
                    name for name in archive.namelist()
                    if len(PurePosixPath(name).parts) == 2
                    and name.endswith(".dist-info/METADATA")
                ]
                if len(members) != 1:
                    raise ValueError(f"Expected one wheel METADATA file: {path.name}")
                return archive.read(members[0])
        with tarfile.open(path, "r:gz") as archive:
            members = [
                member for member in archive.getmembers()
                if len(PurePosixPath(member.name).parts) == 2
                and PurePosixPath(member.name).name == "PKG-INFO"
            ]
            if len(members) != 1 or not members[0].isfile():
                raise ValueError(f"Expected one regular sdist PKG-INFO file: {path.name}")
            stream = archive.extractfile(members[0])
            if stream is None:
                raise ValueError(f"Cannot read sdist PKG-INFO: {path.name}")
            with stream:
                return stream.read()
    except (OSError, zipfile.BadZipFile, tarfile.TarError, KeyError) as exc:
        raise ValueError(f"Cannot read release artifact {path.name}: {exc}") from exc


def _validate_artifact(path: Path, name: str, version: Version) -> None:
    parsed = (
        parse_wheel_filename(path.name)
        if path.suffix == ".whl"
        else parse_sdist_filename(path.name)
    )
    if parsed[0] != name or parsed[1] != version:
        raise ValueError(f"Artifact filename does not match {name} {version}: {path.name}")

    metadata = BytesParser(policy=policy.default).parsebytes(_read_metadata(path))
    for field in ("Name", "Version", "Requires-Python"):
        values = metadata.get_all(field, [])
        if len(values) != 1 or not values[0].strip():
            raise ValueError(f"Expected one {field} field in {path.name}")
    if canonicalize_name(metadata["Name"]) != name or Version(metadata["Version"]) != version:
        raise ValueError(f"Artifact metadata does not match {name} {version}: {path.name}")
    if metadata["Requires-Python"] != ">=3.10":
        raise ValueError(f"Requires-Python must be >=3.10: {path.name}")


def select_release_artifacts(
    dist_dir: Path, name: str, version: str
) -> tuple[Path, Path]:
    """Return only a validated wheel/sdist pair from a fresh build directory."""
    dist_dir = Path(dist_dir).resolve()
    if not dist_dir.is_dir():
        raise ValueError(f"Release output directory does not exist: {dist_dir}")
    files = sorted(dist_dir.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError("Release output must contain only regular artifact files")
    wheels = [path for path in files if path.suffix == ".whl"]
    sdists = [path for path in files if path.name.endswith(".tar.gz")]
    if len(files) != 2 or len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one .tar.gz source distribution")
    expected_name = canonicalize_name(name)
    expected_version = Version(version)
    for path in (wheels[0], sdists[0]):
        _validate_artifact(path, expected_name, expected_version)
    return wheels[0], sdists[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    try:
        paths = select_release_artifacts(args.dist_dir, args.name, args.version)
    except ValueError as exc:
        print(f"Release artifact validation failed: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        sys.stdout.buffer.write(os.fsencode(path) + b"\0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
