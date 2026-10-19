#!/usr/bin/env python3
"""Validate and archive the installed Windows x64 shared CPU SDK."""

import argparse
import hashlib
from pathlib import Path
import re
import struct
import zipfile

from package_wheel import project_version, validate_dll


DLL = "InspireFace/lib/libInspireFace.dll"
IMPLIB = "InspireFace/lib/InspireFace.lib"
CONFIG = "InspireFace/lib/cmake/InspireFace/InspireFaceConfig.cmake"
CONFIG_VERSION = "InspireFace/lib/cmake/InspireFace/InspireFaceConfigVersion.cmake"
REQUIRED = {
    DLL, IMPLIB, CONFIG, CONFIG_VERSION, "version.txt",
    "InspireFace/include/inspireface.h", "InspireFace/include/intypedef.h",
    "InspireFace/include/herror.h", "InspireFace/include/inspireface/inspireface.hpp",
    "InspireFace/include/inspirecv/inspirecv.h",
}


def validate_import_library(data):
    """Check the MSVC import archive's machine and referenced DLL."""
    if not data.startswith(b"!<arch>\n"):
        raise ValueError("Expected an MSVC import library archive")
    cursor = 8
    imports = 0
    while cursor < len(data):
        header = data[cursor:cursor + 60]
        if len(header) != 60 or header[58:] != b"`\n":
            raise ValueError("Invalid import library archive header")
        size = int(header[48:58].strip())
        start = cursor + 60
        end = start + size
        if size < 0 or end > len(data):
            raise ValueError("Truncated import library archive member")
        name = header[:16].strip()
        member = data[start:end]
        if name not in (b"/", b"//"):
            if len(member) < 20:
                raise ValueError("Invalid import library object")
            if member[:4] == b"\0\0\xff\xff":
                machine = struct.unpack_from("<H", member, 6)[0]
                payload_size = struct.unpack_from("<I", member, 12)[0]
                payload = member[20:20 + payload_size].split(b"\0")
                if len(member) != 20 + payload_size or len(payload) < 3 or payload[1] != b"libInspireFace.dll":
                    raise ValueError("Import library must reference libInspireFace.dll")
                imports += 1
            else:
                machine = struct.unpack_from("<H", member)[0]
            if machine != 0x8664:
                raise ValueError("Import library contains a non-AMD64 object")
        cursor = end + (size % 2)
    if cursor != len(data) or not imports:
        raise ValueError("Import library has no valid x64 DLL imports")


def check_sdk(repo, sdk, version=""):
    native_version, _ = project_version(repo)
    version = version.removeprefix("v") if version else native_version
    release = re.fullmatch(r"(\d+\.\d+\.\d+)(?:[-.][A-Za-z0-9][A-Za-z0-9.-]*)?", version)
    if not release or release.group(1) != native_version:
        raise ValueError("Release version must match the CMake native version " + native_version)
    if sdk.is_symlink() or not sdk.is_dir():
        raise ValueError("SDK must be an installed directory, not a symlink")
    files = []
    folded = set()
    for path in sorted(sdk.rglob("*")):
        relative = path.relative_to(sdk)
        name = relative.as_posix()
        if path.is_symlink() or any(part.startswith(".") for part in relative.parts):
            raise ValueError("Unexpected symlink or hidden SDK content: " + name)
        if path.is_dir():
            continue
        header = name.startswith("InspireFace/include/") and path.suffix in {".h", ".hpp", ".inl"}
        if not path.is_file() or not (name in REQUIRED or header):
            raise ValueError("Unexpected SDK content: " + name)
        if name.casefold() in folded:
            raise ValueError("SDK contains a case-insensitive filename collision: " + name)
        folded.add(name.casefold())
        files.append(path)
    names = {path.relative_to(sdk).as_posix() for path in files}
    if not REQUIRED <= names:
        raise ValueError("SDK is missing required files: " + ", ".join(sorted(REQUIRED - names)))
    if (sdk / "version.txt").read_text(encoding="utf-8").strip() != "InspireFace Version: " + native_version:
        raise ValueError("SDK version.txt does not match the CMake native version")
    cmake_version = (sdk / CONFIG_VERSION).read_text(encoding="utf-8")
    match = re.search(r'set\(PACKAGE_VERSION\s+"([^"]+)"\)', cmake_version)
    if not match or match.group(1) != native_version:
        raise ValueError("SDK CMake package version does not match the native version")
    config = (sdk / CONFIG).read_text(encoding="utf-8")
    if not re.search(r"add_library\(InspireFace::InspireFace\s+SHARED\s+IMPORTED\)", config):
        raise ValueError("SDK CMake package must expose the shared InspireFace target")
    for token in ("libInspireFace.dll", "InspireFace.lib", "IMPORTED_IMPLIB", "INSPIRECV_API=__declspec(dllimport)"):
        if token not in config:
            raise ValueError("SDK CMake package is missing " + token)
    validate_dll((sdk / DLL).read_bytes())
    validate_import_library((sdk / IMPLIB).read_bytes())
    return version, files


def package_sdk(repo, sdk, output, version=""):
    version, files = check_sdk(repo, sdk, version)
    root_name = "inspireface-windows-x64-" + version
    output = output.resolve()
    if output == sdk.resolve() or sdk.resolve() in output.parents:
        raise ValueError("Archive output must be outside the SDK install directory")
    destination = output / (root_name + ".zip")
    checksum = destination.with_suffix(".zip.sha256")
    if destination.exists() or checksum.exists():
        raise FileExistsError("SDK archive already exists: " + str(destination))
    output.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.partial")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                archive.write(path, root_name + "/" + path.relative_to(sdk).as_posix())
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("SDK archive checksum validation failed")
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        temporary.replace(destination)
        checksum.write_text(digest + "  " + destination.name + "\n", encoding="ascii")
    finally:
        temporary.unlink(missing_ok=True)
    print("SDK=" + str(destination))
    print("SHA256=" + digest)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--sdk", type=Path, required=True, help="Install root containing InspireFace/ and version.txt")
    parser.add_argument("--version", default="", help="Native release version or v-prefixed tag")
    parser.add_argument("--output-dir", type=Path, default=Path("build/release"))
    args = parser.parse_args()
    try:
        package_sdk(args.repo.resolve(), args.sdk, args.output_dir, args.version)
    except (OSError, ValueError, struct.error, zipfile.BadZipFile) as error:
        parser.exit(1, "SDK packaging failed: {}\n".format(error))


if __name__ == "__main__":
    main()
