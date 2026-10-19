#!/usr/bin/env python3
"""Validate and collect one universal Python wheel for each release platform."""

import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile


PLATFORMS = {
    "manylinux2014-wheels": ("manylinux2014_x86_64", "linux/x64/libInspireFace.so"),
    "manylinux2014-aarch64-wheels": ("manylinux2014_aarch64", "linux/arm64/libInspireFace.so"),
    "macos-arm64-wheels": ("macosx_11_0_arm64", "darwin/arm64/libInspireFace.dylib"),
    "macos-x86_64-wheels": ("macosx_12_0_x86_64", "darwin/x64/libInspireFace.dylib"),
    "windows-x64-wheels": ("win_amd64", "windows/x64/libInspireFace.dll"),
}


def project_version():
    root = Path(__file__).resolve().parents[2]
    cmake = (root / "CMakeLists.txt").read_text()
    parts = []
    for part in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(r"set\(INSPIRE_FACE_VERSION_" + part + r"\s+(\d+)\)", cmake)
        if not match:
            raise ValueError(f"Cannot read {part} version from CMakeLists.txt")
        parts.append(match.group(1))
    return ".".join(parts) + (root / "python/post").read_text().strip()


def validate_wheel(path, version, platform, library):
    expected = f"inspireface-{version}-py3-none-{platform}.whl"
    if path.name != expected:
        raise ValueError(f"Expected {expected}, found {path.name}")
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate entries in wheel ZIP")
        bad_entry = wheel.testzip()
        if bad_entry:
            raise ValueError(f"ZIP CRC check failed: {bad_entry}")
        info = f"inspireface-{version}.dist-info/"
        metadata = BytesParser().parsebytes(wheel.read(info + "METADATA"))
        if metadata["Name"] != "inspireface" or metadata["Version"] != version:
            raise ValueError("Package metadata does not match the expected name/version")
        wheel_info = BytesParser().parsebytes(wheel.read(info + "WHEEL"))
        if wheel_info.get_all("Tag") != [f"py3-none-{platform}"]:
            raise ValueError("Wheel metadata does not match the expected py3-none platform tag")
        if wheel_info["Root-Is-Purelib"] != "false":
            raise ValueError("Native library wheel must set Root-Is-Purelib: false")
        # setuptools may put ctypes packages in .data/purelib even though the
        # wheel itself is platform-specific. Both layouts install to site-packages.
        relative = f"inspireface/modules/core/libs/{library}"
        candidates = [relative, f"inspireface-{version}.data/purelib/{relative}"]
        libraries = [name for name in candidates if name in names]
        if len(libraries) != 1 or wheel.getinfo(libraries[0]).file_size == 0:
            raise ValueError(f"Missing, empty or duplicated native library: {library}")
        native_libraries = [name for name in names if re.search(r"\.(dll|dylib|so(?:\.[0-9]+)*)$", name, re.I)]
        if native_libraries != libraries:
            raise ValueError(f"Unexpected native libraries for {platform}: {native_libraries}")


def collect_wheels(artifacts, output, version, legacy_macos_python=None):
    if not re.fullmatch(r"[0-9][A-Za-z0-9.!+]*", version):
        raise ValueError(f"Invalid package version: {version!r}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    selected = []
    for artifact, (platform, library) in PLATFORMS.items():
        directories = [artifacts / artifact]
        if legacy_macos_python and artifact.startswith("macos-"):
            directories.append(artifacts / f"{artifact}-py{legacy_macos_python}")
        found = [directory for directory in directories if directory.is_dir()]
        if len(found) != 1:
            raise ValueError(f"Expected exactly one artifact for {platform}, found {found}")
        wheels = sorted(found[0].rglob("*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"Expected one wheel in {found[0]}, found {len(wheels)}")
        path = wheels[0]
        try:
            validate_wheel(path, version, platform, library)
        except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
            raise ValueError(f"Invalid wheel {path}: {error}") from error
        selected.append(path)

    # Validate the entire release before writing any publishable files. Do not
    # merge or overwrite independently built wheels with identical filenames.
    output.mkdir(parents=True, exist_ok=True)
    for path in selected:
        shutil.copyfile(path, output / path.name)
        digest = hashlib.sha256((output / path.name).read_bytes()).hexdigest()
        print(f"{path.parent.name} -> {path.name}  sha256:{digest}")
    return selected


def write_release_manifest(output, version, manifest_path):
    """Record the validated files so a later PyPI install can verify their identity."""
    manifest = {
        "version": version,
        "wheels": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.glob("*.whl"))
        },
    }
    if len(manifest["wheels"]) != len(PLATFORMS):
        raise ValueError("Release manifest requires all five validated wheels")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", help="Defaults to CMakeLists.txt plus python/post")
    parser.add_argument("--manifest", type=Path,
                        help="Write validated wheel versions and SHA256 hashes for post-publish checks")
    parser.add_argument("--legacy-macos-python", choices=["3.12"],
                        help="Select only py3.12 artifacts from the old macOS matrix")
    args = parser.parse_args()
    try:
        version = args.version or project_version()
        selected = collect_wheels(args.artifacts, args.output, version, args.legacy_macos_python)
        if args.manifest:
            write_release_manifest(args.output, version, args.manifest)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Validated {len(selected)} wheels for inspireface {version}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as outputs:
            outputs.write(f"version={version}\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(f"### Validated inspireface {version}\n\n")
            for path in selected:
                summary.write(f"- `{path.name}` from `{path.parent.name}`\n")


if __name__ == "__main__":
    main()
