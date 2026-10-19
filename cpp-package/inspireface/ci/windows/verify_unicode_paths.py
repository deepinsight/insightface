"""Exercise UTF-8 model, image and persistent database paths through the C ABI."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import shutil
import tempfile


class HubConfiguration(ct.Structure):
    _fields_ = [('primary_key_mode', ct.c_int32), ('enable_persistence', ct.c_int32),
                ('path', ct.c_char_p), ('threshold', ct.c_float), ('search_mode', ct.c_int32)]


class Feature(ct.Structure):
    _fields_ = [('size', ct.c_int32), ('data', ct.POINTER(ct.c_float))]


class Identity(ct.Structure):
    _fields_ = [('id', ct.c_int64), ('feature', ct.POINTER(Feature))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    args = parser.parse_args()
    library = ct.CDLL(str(args.library.resolve()))
    signatures = {
        'HFLaunchInspireFace': [ct.c_char_p], 'HFTerminateInspireFace': [],
        'HFCreateImageBitmapFromFilePath': [ct.c_char_p, ct.c_int32, ct.POINTER(ct.c_void_p)],
        'HFImageBitmapWriteToFile': [ct.c_void_p, ct.c_char_p], 'HFReleaseImageBitmap': [ct.c_void_p],
        'HFFeatureHubDataEnable': [HubConfiguration], 'HFFeatureHubDataDisable': [],
        'HFFeatureHubInsertFeature': [Identity, ct.POINTER(ct.c_int64)],
        'HFFeatureHubGetFaceCount': [ct.POINTER(ct.c_int32)],
    }
    for name, types in signatures.items():
        function = getattr(library, name)
        function.argtypes = types
        function.restype = ct.c_int32
    failures = []

    def check(name, status):
        if status != 0:
            failures.append('{} returned {}'.format(name, status))
            return False
        return True

    def encoded(path):
        return str(path.resolve()).encode('utf-8')

    args.work_dir.mkdir(parents=True, exist_ok=True)
    # Use a non-ASCII symbol to exercise UTF-8 paths while keeping the source ASCII.
    with tempfile.TemporaryDirectory(prefix='InspireFace Unicode \u2603 spaces ', dir=args.work_dir) as folder:
        root = Path(folder)
        model = root / 'model Pikachu \u2603'
        picture = root / ('face input \u2603' + args.image.suffix)
        output = root / 'image output \u2603.png'
        shutil.copy2(args.model, model)
        shutil.copy2(args.image, picture)
        check('launch UTF-8 model', library.HFLaunchInspireFace(encoded(model)))
        library.HFTerminateInspireFace()
        # Keep the remaining checks independent of a model-path failure.
        check('launch original model', library.HFLaunchInspireFace(encoded(args.model)))
        try:
            bitmap = ct.c_void_p()
            if check('read UTF-8 image', library.HFCreateImageBitmapFromFilePath(encoded(picture), 3, ct.byref(bitmap))):
                try:
                    if check('write UTF-8 image', library.HFImageBitmapWriteToFile(bitmap, encoded(output))):
                        if not output.is_file() or output.stat().st_size == 0:
                            failures.append('UTF-8 output image was not created at the requested path')
                finally:
                    check('release bitmap', library.HFReleaseImageBitmap(bitmap))

            for directory_mode in (False, True):
                location = root / ('database file \u2603.db' if not directory_mode else 'database directory \u2603')
                if directory_mode:
                    location.mkdir()
                configuration = HubConfiguration(0, 1, encoded(location), 0.48, 1)
                label = 'directory' if directory_mode else 'file'
                if check('open UTF-8 database ' + label, library.HFFeatureHubDataEnable(configuration)):
                    try:
                        values = (ct.c_float * 512)(1.0, *([0.0] * 511))
                        feature = Feature(512, values)
                        identifier = ct.c_int64(-1)
                        check('insert ' + label, library.HFFeatureHubInsertFeature(Identity(-1, ct.pointer(feature)), ct.byref(identifier)))
                    finally:
                        check('close database ' + label, library.HFFeatureHubDataDisable())
                    if check('reopen UTF-8 database ' + label, library.HFFeatureHubDataEnable(configuration)):
                        try:
                            count = ct.c_int32()
                            if check('count persisted entries ' + label, library.HFFeatureHubGetFaceCount(ct.byref(count))) and count.value != 1:
                                failures.append('Persisted {} database count is {}'.format(label, count.value))
                        finally:
                            check('close reopened database ' + label, library.HFFeatureHubDataDisable())
        finally:
            library.HFFeatureHubDataDisable()
            check('terminate', library.HFTerminateInspireFace())
    print(json.dumps({'unicode_path_failures': failures}, ensure_ascii=True))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
