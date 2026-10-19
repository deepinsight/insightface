"""Smoke-test the real Python package, including Unicode paths and persistence.

Use --require-installed-wheel in an isolated environment after pip install.
It rejects source/override imports and verifies wheel RECORD hashes, including
the DLL actually loaded by Windows. Without that flag, source-tree and explicit
native-library diagnostics remain supported.
"""
import argparse
import base64
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np


def verify_installed_wheel(isf, native, library_path):
    try:
        from importlib.metadata import distribution
    except ImportError:  # Python 3.7
        from importlib_metadata import distribution

    package = distribution('inspireface')
    records = {str(item).replace('\\', '/'): item for item in package.files or ()}
    package_entry = 'inspireface/__init__.py'
    assert package_entry in records, 'Installed distribution has no package RECORD entry'
    assert Path(isf.__file__).resolve() == Path(package.locate_file(package_entry)).resolve(), (
        'Imported Python package does not belong to the installed distribution'
    )
    assert isf.__version__ == package.version
    library_relative = library_path.relative_to(Path(isf.__file__).resolve().parent)
    library_entry = 'inspireface/' + library_relative.as_posix()
    assert library_entry in records, 'Loaded native library is not recorded in the wheel'

    checked = 0
    for name, entry in records.items():
        if not name.startswith('inspireface/') or not name.endswith(('.py', '.dll', '.so', '.dylib')):
            continue
        assert entry.hash is not None, 'Missing wheel RECORD hash: ' + name
        data = Path(package.locate_file(entry)).read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.new(entry.hash.mode, data).digest()).decode('ascii').rstrip('=')
        assert digest == entry.hash.value, 'Installed wheel file differs from RECORD: ' + name
        assert entry.size == len(data), 'Installed wheel file size differs from RECORD: ' + name
        checked += 1

    wheel_metadata = package.read_text('WHEEL') or ''
    assert 'Root-Is-Purelib: false' in wheel_metadata
    tags = [line[5:] for line in wheel_metadata.splitlines() if line.startswith('Tag: ')]
    if sys.platform == 'win32':
        assert 'py3-none-win_amd64' in tags
        assert library_relative.as_posix() == 'modules/core/libs/windows/x64/libInspireFace.dll'
        assert ctypes.sizeof(ctypes.c_void_p) == 8
        # Confirm the loaded module, not only the path selected by our loader.
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        module_filename = kernel32.GetModuleFileNameW
        module_filename.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
        module_filename.restype = ctypes.c_uint32
        buffer = ctypes.create_unicode_buffer(32768)
        module = native._libs[native._LIBRARY_FILENAME].access['cdecl']
        length = module_filename(module._handle, buffer, len(buffer))
        assert 0 < length < len(buffer), 'Could not resolve the loaded Windows DLL'
        assert Path(buffer.value).resolve() == library_path
    return {'version': package.version, 'tags': tags, 'record_files_verified': checked}


def resource_counts(native):
    counts = []
    for counter in (native.HFDeBugGetUnreleasedSessionsCount, native.HFDeBugGetUnreleasedStreamsCount):
        value = native.HInt32()
        assert counter(ctypes.byref(value)) == 0
        counts.append(value.value)
    return counts


def main():
    if sys.flags.optimize:
        raise RuntimeError('Run this validation without -O so its assertions remain active')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--require-installed-wheel', action='store_true')
    parser.add_argument('--expected-dll-sha256')
    args = parser.parse_args()
    if args.require_installed_wheel:
        for name in ('PYTHONPATH', 'INSPIREFACE_LIBRARY_PATH', 'INSPIREFACE_TEST_NATIVE_OVERRIDE'):
            assert not os.environ.get(name), 'Wheel validation must not use ' + name

    import inspireface as isf
    from inspireface.modules.core import native
    from inspireface.modules.core._library_path import get_lib_path

    library = Path(get_lib_path()).resolve()
    assert Path(native._LIBRARY_FILENAME).resolve() == library
    provenance = verify_installed_wheel(isf, native, library) if args.require_installed_wheel else None
    dll_sha256 = hashlib.sha256(library.read_bytes()).hexdigest()
    if args.expected_dll_sha256:
        assert dll_sha256 == args.expected_dll_sha256.lower()
    assert ctypes.sizeof(native.HResult) == ctypes.sizeof(ctypes.c_long)
    assert ctypes.sizeof(native.HFaceId) == 8
    if sys.platform == 'win32':
        assert ctypes.sizeof(native.HResult) == 4  # Windows uses LLP64, including x64.
    assert ctypes.sizeof(native.HFSessionConfigV2) == 64
    assert native.HFSessionConfigV2.featureMask.offset == 8
    assert isf.c_api_level() == 2
    assert resource_counts(native) == [0, 0]

    args.work_dir.mkdir(parents=True, exist_ok=True)
    # Use a non-ASCII symbol to exercise UTF-8 paths while keeping the source ASCII.
    with tempfile.TemporaryDirectory(prefix='Python Unicode \u2603 spaces ', dir=args.work_dir) as folder:
        root = Path(folder)
        model = root / 'model Pikachu \u2603'
        shutil.copy2(args.model, model)
        assert isf.launch(resource_path=str(model))
        try:
            pixels = np.zeros((192, 192, 3), dtype=np.uint8)
            with isf.InspireFaceSession(0, auto_launch=False) as session:
                assert session.face_detection(pixels) == []
            session.close()  # Closing twice must not release an already-freed native handle.
            assert session.closed
            output = root / 'image output \u2603.png'
            with isf.ImageStream.load_from_cv_image(pixels) as stream:
                stream.write_to_file(str(output))
            stream.close()
            assert stream.closed
            assert output.is_file() and output.stat().st_size > 0
            assert resource_counts(native) == [0, 0]
            for directory_mode in (False, True):
                database = root / ('database directory \u2603' if directory_mode else 'database file \u2603.db')
                if directory_mode:
                    database.mkdir()
                key_mode = isf.HF_PK_MANUAL_INPUT if directory_mode else isf.HF_PK_AUTO_INCREMENT
                requested_id = (1 << 40) + 17 if directory_mode else -1
                configuration = isf.FeatureHubConfiguration(key_mode, True, str(database), 0.48, 1)
                assert isf.feature_hub_enable(configuration)
                try:
                    feature = np.zeros(512, dtype=np.float32)
                    feature[0] = 1.0
                    inserted, identifier = isf.feature_hub_face_insert(isf.FaceIdentity(feature, requested_id))
                    assert inserted and identifier >= 0
                    if directory_mode:
                        assert identifier == requested_id  # Catch accidental 32-bit C long IDs.
                    assert isf.feature_hub_get_face_count() == 1
                finally:
                    isf.feature_hub_disable()
                assert isf.feature_hub_enable(configuration)
                try:
                    assert isf.feature_hub_get_face_count() == 1
                    saved = isf.feature_hub_get_face_identity(identifier)
                    assert saved.id == identifier
                    np.testing.assert_array_equal(saved.feature, feature)
                finally:
                    isf.feature_hub_disable()
        finally:
            isf.terminate()
        assert not isf.query_launch_status()
        assert resource_counts(native) == [0, 0]
    print(json.dumps({'python_api': 'passed', 'version': isf.__native_version__,
                      'python': sys.version, 'package': isf.__file__,
                      'library': str(library), 'dll_sha256': dll_sha256,
                      'installed_wheel': provenance}, ensure_ascii=True))


if __name__ == '__main__':
    main()
