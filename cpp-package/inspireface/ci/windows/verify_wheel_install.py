"""Verify an installed Windows wheel without downloading a model.

Run with the isolated interpreter of a fresh virtual environment:
    python -I ci/windows/verify_wheel_install.py --wheel path/to/inspireface.whl
"""
import argparse
import base64
import ctypes
from email.parser import BytesParser
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import sys
import zipfile


LIBRARY = "inspireface/modules/core/libs/windows/x64/libInspireFace.dll"


def resource_counts(native):
    result = []
    for counter in (native.HFDeBugGetUnreleasedSessionsCount, native.HFDeBugGetUnreleasedStreamsCount):
        count = native.HInt32()
        assert counter(ctypes.byref(count)) == 0
        result.append(count.value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    if sys.flags.optimize or not sys.flags.isolated:
        raise RuntimeError("Use python -I without -O for installed wheel verification")
    assert sys.platform == "win32" and ctypes.sizeof(ctypes.c_void_p) == 8
    assert sys.prefix != sys.base_prefix, "Install the wheel in a fresh virtual environment"
    for name in ("PYTHONPATH", "INSPIREFACE_LIBRARY_PATH", "INSPIREFACE_TEST_NATIVE_OVERRIDE"):
        assert not os.environ.get(name), "Installed wheel verification must not use " + name

    import numpy as np
    import inspireface as isf
    from inspireface.modules.core import native
    from inspireface.modules.core._library_path import get_lib_path

    package = distribution("inspireface")
    package_path = Path(isf.__file__).resolve()
    assert package_path == Path(package.locate_file("inspireface/__init__.py")).resolve()
    assert package_path.is_relative_to(Path(sys.prefix).resolve()), "Imported package is outside the virtual environment"
    assert isf.__version__ == package.version
    library_path = Path(get_lib_path()).resolve()
    assert library_path == Path(package.locate_file(LIBRARY)).resolve()
    assert Path(native._LIBRARY_FILENAME).resolve() == library_path

    records = {str(item).replace("\\", "/"): item for item in package.files or ()}
    assert LIBRARY in records and "inspireface/__init__.py" in records
    verified = 0
    for name, entry in records.items():
        if not name.startswith("inspireface/") or not name.endswith((".py", ".dll", ".so", ".dylib")):
            continue
        assert entry.hash is not None, "Missing RECORD hash: " + name
        data = Path(package.locate_file(entry)).read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.new(entry.hash.mode, data).digest()).decode("ascii").rstrip("=")
        assert digest == entry.hash.value and len(data) == entry.size, "RECORD mismatch: " + name
        verified += 1

    info = "inspireface-" + package.version + ".dist-info/"
    with zipfile.ZipFile(args.wheel) as wheel:
        metadata = BytesParser().parsebytes(wheel.read(info + "METADATA"))
        assert metadata["Name"] == "inspireface" and metadata["Version"] == package.version
        wheel_info = BytesParser().parsebytes(wheel.read(info + "WHEEL"))
        assert wheel_info.get_all("Tag") == ["py3-none-win_amd64"]
        assert wheel_info["Root-Is-Purelib"] == "false"
        candidates = [LIBRARY, "inspireface-" + package.version + ".data/purelib/" + LIBRARY]
        entries = [name for name in candidates if name in wheel.namelist()]
        assert len(entries) == 1, "Expected exactly one Windows DLL in the wheel"
        assert wheel.read(entries[0]) == library_path.read_bytes(), "Installed DLL differs from the built wheel"

    # Confirm the Windows loader actually opened the packaged DLL.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    module_filename = kernel32.GetModuleFileNameW
    module_filename.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    module_filename.restype = ctypes.c_uint32
    buffer = ctypes.create_unicode_buffer(32768)
    module = native._libs[native._LIBRARY_FILENAME].access["cdecl"]
    length = module_filename(module._handle, buffer, len(buffer))
    assert 0 < length < len(buffer)
    assert Path(buffer.value).resolve() == library_path

    # Windows x64 uses LLP64: C long remains 32 bits, identifiers are 64 bits.
    assert ctypes.sizeof(native.HResult) == ctypes.sizeof(ctypes.c_long) == 4
    assert ctypes.sizeof(native.HFaceId) == 8
    assert ctypes.sizeof(native.HFSessionConfigV2) == 64
    assert native.HFSessionConfigV2.featureMask.offset == 8
    assert isf.c_api_level() == 2
    assert resource_counts(native) == [0, 0]
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    with isf.ImageStream.load_from_cv_image(pixels) as stream:
        assert resource_counts(native) == [0, 1]
    stream.close()
    assert stream.closed and resource_counts(native) == [0, 0]
    print(json.dumps({"windows_wheel": "passed", "version": package.version,
                      "native_version": isf.__native_version__, "package": str(package_path),
                      "library": str(library_path), "record_files_verified": verified,
                      "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest()}, ensure_ascii=True))


if __name__ == "__main__":
    main()
