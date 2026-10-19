#!/usr/bin/env python3
"""Stage and validate a Windows x64 ctypes wheel without changing python/libs."""

import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile


LIBRARY = "inspireface/modules/core/libs/windows/x64/libInspireFace.dll"
TAG = "py3-none-win_amd64"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_pe_imports(data):
    """Read PE32+ import names using only stdlib, including on SDK-only PCs."""
    if len(data) < 64 or data[:2] != b"MZ":
        raise ValueError("Native library is not a PE DLL")
    pe = struct.unpack_from("<I", data, 60)[0]
    if pe + 24 > len(data) or data[pe:pe + 4] != b"PE\0\0":
        raise ValueError("Native library has an invalid PE header")
    machine, section_count = struct.unpack_from("<HH", data, pe + 4)
    optional_size, characteristics = struct.unpack_from("<HH", data, pe + 20)
    optional = pe + 24
    if machine != 0x8664 or not characteristics & 0x2000:
        raise ValueError("Expected an AMD64/x64 DLL (x86 and ARM64 are unsupported)")
    if optional_size < 128 or optional + optional_size > len(data):
        raise ValueError("Native library has an invalid optional PE header")
    if struct.unpack_from("<H", data, optional)[0] != 0x20B:
        raise ValueError("Expected a PE32+ x64 DLL")
    sections = []
    table = optional + optional_size
    if table + section_count * 40 > len(data):
        raise ValueError("Native library has an invalid PE section table")
    for index in range(section_count):
        virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", data, table + index * 40 + 8
        )
        sections.append((virtual_address, max(virtual_size, raw_size), raw_offset, raw_size))

    def file_offset(rva, size):
        for address, extent, offset, raw_size in sections:
            delta = rva - address
            if 0 <= delta < extent and delta + size <= raw_size and offset + delta + size <= len(data):
                return offset + delta
        raise ValueError("Native library has an invalid PE import address")

    import_rva, import_size = struct.unpack_from("<II", data, optional + 120)
    if not import_rva or import_size < 20:
        raise ValueError("Native library has no import table")
    imports = []
    for index in range(import_size // 20):
        offset = file_offset(import_rva + index * 20, 20)
        descriptor = struct.unpack_from("<IIIII", data, offset)
        if not any(descriptor):
            break
        name_offset = file_offset(descriptor[3], 1)
        end = data.find(b"\0", name_offset, min(len(data), name_offset + 512))
        if end < 0:
            raise ValueError("Native library has an unterminated import name")
        imports.append(data[name_offset:end].decode("ascii"))
    else:
        raise ValueError("Native library has an unterminated import table")
    return imports


def validate_dll(data):
    imports = read_pe_imports(data)
    # The supported CPU/MNN SDK embeds MNN statically. Do not silently create
    # a broken wheel for a SDK needing additional OpenCV/MNN/GPU DLLs.
    windows_dlls = {"kernel32.dll", "advapi32.dll", "user32.dll", "ntdll.dll"}
    release_crt = {"msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "ucrtbase.dll"}
    for name in imports:
        lower = name.lower()
        if re.fullmatch(r"(?:msvcp\d+d|vcruntime\d+(?:_\d+)?d|ucrtbased)\.dll", lower):
            raise ValueError("Debug CRT dependency is not distributable in a release wheel: " + name)
        if lower not in windows_dlls | release_crt and not lower.startswith(("api-ms-win-", "ext-ms-win-")):
            raise ValueError("Unsupported DLL dependency; use the CPU SDK with static MNN: " + name)
    return imports


def project_version(repo):
    cmake = (repo / "CMakeLists.txt").read_text(encoding="utf-8")
    parts = []
    for part in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(r"set\(INSPIRE_FACE_VERSION_" + part + r"\s+(\d+)\)", cmake)
        if not match:
            raise ValueError("Cannot read native " + part + " version from CMakeLists.txt")
        parts.append(match.group(1))
    base_version = ".".join(parts)
    post_file = repo / "python/post"
    post = post_file.read_text(encoding="utf-8").strip() if post_file.exists() else ""
    return base_version, base_version + post


def validate_wheel(path, version, native_data):
    if path.name != "inspireface-{}-{}.whl".format(version, TAG):
        raise ValueError("Unexpected wheel filename: " + path.name)
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        if len(names) != len(set(names)) or wheel.testzip():
            raise ValueError("Invalid wheel ZIP entries or CRC")
        info = "inspireface-{}.dist-info/".format(version)
        metadata = BytesParser().parsebytes(wheel.read(info + "METADATA"))
        wheel_info = BytesParser().parsebytes(wheel.read(info + "WHEEL"))
        if metadata["Name"] != "inspireface" or metadata["Version"] != version:
            raise ValueError("Wheel name/version metadata mismatch")
        if wheel_info.get_all("Tag") != [TAG] or wheel_info["Root-Is-Purelib"] != "false":
            raise ValueError("Expected a non-pure py3-none-win_amd64 wheel")
        candidates = [LIBRARY, "inspireface-{}.data/purelib/{}".format(version, LIBRARY)]
        bundled = [name for name in names if name in candidates]
        native = [name for name in names if "/modules/core/libs/" in name or
                  name.lower().endswith((".dll", ".so", ".dylib", ".pyd", ".lib", ".a"))]
        if len(bundled) != 1 or native != bundled:
            raise ValueError("Wheel must contain exactly one native file in windows/x64")
        if wheel.read(bundled[0]) != native_data:
            raise ValueError("Bundled DLL differs from the validated SDK DLL")


def build_wheel(repo, sdk, output, no_build_isolation=False, overwrite=False):
    base_version, version = project_version(repo)
    dll = sdk / "InspireFace/lib/libInspireFace.dll"
    native_data = dll.read_bytes()
    imports = validate_dll(native_data)
    sdk_version = (sdk / "version.txt").read_text(encoding="utf-8").strip()
    if sdk_version != "InspireFace Version: " + base_version:
        raise ValueError("SDK version.txt does not match the source native version " + base_version)
    filename = "inspireface-{}-{}.whl".format(version, TAG)
    target = output / filename
    if target.exists() and not overwrite:
        raise FileExistsError("Wheel already exists; use -Force to replace only this file: " + str(target))

    with tempfile.TemporaryDirectory(prefix="inspireface-wheel-") as temporary:
        stage = Path(temporary) / "python"
        stage.mkdir()
        for name in ("setup.py", "pyproject.toml", "README.md"):
            shutil.copy2(repo / "python" / name, stage / name)
        (stage / "version.txt").write_text(base_version + "\n", encoding="utf-8")
        if (repo / "python/post").exists():
            shutil.copy2(repo / "python/post", stage / "post")
        shutil.copytree(repo / "python/inspireface", stage / "inspireface",
                        ignore=shutil.ignore_patterns("libs", "__pycache__", "*.pyc", "*.pyo", ".DS_Store"))
        staged_dll = stage / LIBRARY
        staged_dll.parent.mkdir(parents=True)
        staged_dll.write_bytes(native_data)
        dist = Path(temporary) / "dist"
        env = os.environ.copy()
        env.update(INSPIRE_FACE_TARGET_PLATFORM="windows", INSPIRE_FACE_TARGET_ARCH="x64",
                   INSPIRE_FACE_TARGET_AARCH_MAPPING="win_amd64")
        command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-cache-dir",
                   "--wheel-dir", str(dist)]
        if no_build_isolation:
            command.append("--no-build-isolation")
        subprocess.run(command + [str(stage)], env=env, check=True)
        wheels = list(dist.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("Expected exactly one built wheel")
        validate_wheel(wheels[0], version, native_data)
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wheels[0], target)

    manifest = {
        "wheel": target.name, "wheel_sha256": sha256(target.read_bytes()),
        "tag": TAG, "version": version, "native_library": LIBRARY,
        "native_sha256": sha256(native_data), "native_imports": imports,
    }
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("WHEEL=" + str(target))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--sdk", required=True, type=Path, help="Install root containing InspireFace/ and version.txt")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-build-isolation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        build_wheel(args.repo.resolve(), args.sdk.resolve(), args.output.resolve(),
                    args.no_build_isolation, args.overwrite)
    except (ValueError, OSError, struct.error, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        parser.exit(1, "Wheel packaging failed: {}\n".format(error))


if __name__ == "__main__":
    main()
