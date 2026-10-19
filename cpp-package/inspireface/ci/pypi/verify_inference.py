"""Verify a published InspireFace installation on Linux, macOS, or Windows.

Run with a fresh environment's isolated interpreter:
    python -I ci/pypi/verify_inference.py --expected-version 1.2.4.post2 \
        --image test_res/data/bulk/kun.jpg \
        --output-dir build/pypi-inference

InspireFace downloads and caches its default model through launch().
"""

import argparse
import base64
import ctypes
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import platform
import sys


PLATFORM_LIBRARIES = {
    "linux-x64": "linux/x64/libInspireFace.so",
    "linux-arm64": "linux/arm64/libInspireFace.so",
    "darwin-x64": "darwin/x64/libInspireFace.dylib",
    "darwin-arm64": "darwin/arm64/libInspireFace.dylib",
    "windows-x64": "windows/x64/libInspireFace.dll",
}


def platform_key(system=None, machine=None):
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    architecture = {"amd64": "x64", "x86_64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    key = "{}-{}".format(system, architecture)
    if key not in PLATFORM_LIBRARIES:
        raise RuntimeError("Unsupported test platform: {} / {}".format(system, machine))
    return key


def verify_installation(package, isf, native, expected_version, platform_name, environment_prefix):
    """Reject another version, source imports, and native libraries outside the wheel."""
    if package.version != expected_version or isf.__version__ != expected_version:
        raise RuntimeError("Expected InspireFace {}, installed metadata={} package={}".format(
            expected_version, package.version, isf.__version__))
    package_path = Path(isf.__file__).resolve()
    if package_path != Path(package.locate_file("inspireface/__init__.py")).resolve():
        raise RuntimeError("InspireFace was not imported from the installed distribution")
    try:
        package_path.relative_to(Path(environment_prefix).resolve())
    except ValueError as error:
        raise RuntimeError("InspireFace must be installed in the current Python environment") from error

    library_record = "inspireface/modules/core/libs/" + PLATFORM_LIBRARIES[platform_name]
    library_path = Path(package.locate_file(library_record)).resolve()
    try:
        library_path.relative_to(package_path.parent)
    except ValueError as error:
        raise RuntimeError("The native library must be inside the installed package") from error
    if not library_path.is_file():
        raise RuntimeError("The installed distribution has no native library for " + platform_name)
    if Path(native._LIBRARY_FILENAME).resolve() != library_path:
        raise RuntimeError("InspireFace selected a native library outside the installed wheel")
    loaded = native._libs[native._LIBRARY_FILENAME].access["cdecl"]
    if Path(loaded._name).resolve() != library_path:
        raise RuntimeError("InspireFace loaded a native library outside the installed wheel")

    records = {str(item).replace("\\", "/"): item for item in package.files or ()}
    for name in ("inspireface/__init__.py", library_record):
        record = records.get(name)
        if record is None or record.hash is None or record.size is None:
            raise RuntimeError("Missing wheel RECORD hash or size: " + name)
        data = Path(package.locate_file(record)).read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.new(record.hash.mode, data).digest()).decode("ascii").rstrip("=")
        if digest != record.hash.value or len(data) != record.size:
            raise RuntimeError("Installed file does not match wheel RECORD: " + name)
    return package_path, library_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-platform", choices=sorted(PLATFORM_LIBRARIES))
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not sys.flags.isolated:
        raise RuntimeError("Run this check with python -I")
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("This check requires 64-bit Python")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Install the published wheel in a fresh virtual environment")
    target = platform_key()
    if args.expected_platform and target != args.expected_platform:
        raise RuntimeError("Expected platform {}, running {}".format(args.expected_platform, target))
    for name in ("PYTHONPATH", "INSPIREFACE_LIBRARY_PATH", "INSPIREFACE_TEST_NATIVE_OVERRIDE"):
        if os.environ.get(name):
            raise RuntimeError("Installed package verification must not use " + name)
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    import cv2
    import inspireface as isf
    from inspireface.modules.core import native
    import numpy as np

    package = distribution("inspireface")
    package_path, library_path = verify_installation(
        package, isf, native, args.expected_version, target, sys.prefix)

    image = cv2.imread(str(args.image.resolve()), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("OpenCV could not decode the test image")
    height, width = image.shape[:2]
    annotated = image.copy()
    records = []
    features = []
    all_landmarks = []

    isf.launch()
    try:
        isf.switch_image_processing_backend(isf.HF_IMAGE_PROCESSING_CPU)
        with isf.InspireFaceSession(
            isf.SessionCustomParameter(enable_recognition=True, enable_detect_mode_landmark=True),
            isf.HF_DETECT_MODE_ALWAYS_DETECT,
            auto_launch=False,
        ) as session:
            faces = session.face_detection(image)
            if not faces:
                raise RuntimeError("No face was detected in the test image")
            for index, face in enumerate(faces):
                box = np.asarray(face.location, dtype=np.float64)
                if box.shape != (4,) or not np.isfinite(box).all():
                    raise RuntimeError("Invalid bounding box for face {}".format(index))
                x1, y1, x2, y2 = box
                if not (x2 > x1 and y2 > y1 and x2 > 0 and y2 > 0 and x1 < width and y1 < height):
                    raise RuntimeError("Face {} has an empty or out-of-image box".format(index))

                landmarks = np.asarray(session.get_face_dense_landmark(face))
                if (
                    landmarks.ndim != 2
                    or landmarks.shape[1] != 2
                    or len(landmarks) <= 5
                    or not np.isfinite(landmarks).all()
                    or not (np.ptp(landmarks, axis=0) > 0).all()
                ):
                    raise RuntimeError("Invalid dense landmarks for face {}".format(index))

                feature = np.asarray(session.face_feature_extract(image, face))
                if (
                    feature.ndim != 1
                    or not feature.size
                    or not np.issubdtype(feature.dtype, np.floating)
                    or not np.isfinite(feature).all()
                ):
                    raise RuntimeError("Invalid feature vector for face {}".format(index))
                norm = float(np.linalg.norm(feature))
                if not np.isfinite(norm) or norm <= 0:
                    raise RuntimeError("Invalid feature norm for face {}".format(index))
                features.append(feature.copy())
                all_landmarks.append(landmarks.copy())
                records.append({
                    "index": index,
                    "box": box.tolist(),
                    "landmark_count": len(landmarks),
                    "landmarks": landmarks.tolist(),
                    "feature_shape": list(feature.shape),
                    "feature_dtype": str(feature.dtype),
                    "feature_norm": norm,
                })

                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
                for x, y in landmarks:
                    cv2.circle(annotated, (int(round(x)), int(round(y))), 1, (0, 180, 255), -1)
    finally:
        isf.terminate()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output_dir / "annotated.png"), annotated):
        raise RuntimeError("Could not save the annotated image")
    np.save(args.output_dir / "features.npy", np.stack(features), allow_pickle=False)
    np.save(args.output_dir / "landmarks.npy", np.stack(all_landmarks), allow_pickle=False)
    summary = {
        "status": "passed",
        "expected_version": args.expected_version,
        "package_version": package.version,
        "native_version": isf.__native_version__,
        "package_path": str(package_path),
        "native_library_path": str(library_path),
        "native_library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
        "platform_key": target,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "model": "Pikachu",
        "image": args.image.name,
        "image_size": [width, height],
        "face_count": len(records),
        "faces": records,
    }
    summary_json = json.dumps(summary, indent=2, allow_nan=False)
    (args.output_dir / "summary.json").write_text(summary_json + "\n", encoding="utf-8")
    print(summary_json)


if __name__ == "__main__":
    main()
