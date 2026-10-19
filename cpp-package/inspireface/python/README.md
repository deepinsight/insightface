# InspireFace Python API

InspireFace provides an easy-to-use Python API that wraps the underlying dynamic link library through ctypes. You can install the latest release version via pip or configure it using the project's self-compiled dynamic library.

## Quick Installation

### Install via pip (Recommended)

CPU wheels are available for Windows x64 and for Linux/macOS on x86_64 and ARM64. On Windows, use x64 Python and install the Microsoft Visual C++ 2022 x64 Redistributable.

```bash
pip install inspireface
```

Python 3.7 automatically selects `modelscope<1.22.1`, and Python 3.8 selects
`modelscope<1.29.2`. Those newer ModelScope releases require Python 3.8 syntax
and Python 3.9's `zoneinfo`, respectively, without declaring the minimum in
their package metadata. Python 3.9 and newer keep the normal ModelScope
dependency selection. These compatibility constraints apply on all supported
operating systems.

### Manual Installation

1. Copy the compiled native library to `inspireface/modules/core/libs/SYSTEM/CORE_ARCH/`, matching the running Python process. Use `linux` or `darwin` with `x64` or `arm64`, or `windows/x64` for Windows. Library names are `libInspireFace.so` (Linux), `libInspireFace.dylib` (macOS), and `libInspireFace.dll` (Windows):
```bash
# Linux x64 example, from this python directory
mkdir -p inspireface/modules/core/libs/linux/x64
cp YOUR_BUILD_DIR/install/InspireFace/lib/libInspireFace.so inspireface/modules/core/libs/linux/x64/
```

2. Install the Python package and its declared dependencies:
```bash
pip install .
```

For Windows wheel builds, run `.\command\build_wheel_windows.ps1 -PythonExecutable python` from the repository root in an x64 Visual Studio Native Tools PowerShell environment. The output is `python/dist/inspireface-<version>-py3-none-win_amd64.whl`.

## Quick Start

Here's a simple example showing how to use InspireFace for face detection and landmark drawing. `isf.launch()` automatically downloads the default model on first use:

```python
import cv2
import inspireface as isf

isf.launch()
try:
    with isf.InspireFaceSession(
        param=isf.HF_ENABLE_NONE,
        detect_mode=isf.HF_DETECT_MODE_ALWAYS_DETECT,
        auto_launch=False,
    ) as session:
        session.set_detection_confidence_threshold(0.5)

        image = cv2.imread("path/to/your/image.jpg")
        if image is None:
            raise FileNotFoundError("Unable to read the input image")

        faces = session.face_detection(image)
        print(f"Detected {len(faces)} faces")

        draw = image.copy()
        for face in faces:
            x1, y1, x2, y2 = face.location
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            size = (x2 - x1, y2 - y1)
            rect = (center, size, face.roll)
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(draw, [box], 0, (100, 180, 29), 2)

            landmarks = session.get_face_dense_landmark(face)
            for x, y in landmarks.astype(int):
                cv2.circle(draw, (x, y), 0, (220, 100, 0), 2)
finally:
    if isf.query_launch_status():
        isf.terminate()
```

## More Examples

The project provides multiple example files demonstrating different features:

- `sample_face_detection.py`: Basic face detection
- `sample_face_track_from_video.py`: Video face tracking
- `sample_face_recognition.py`: Face recognition
- `sample_face_comparison.py`: Face comparison
- `sample_feature_hub.py`: Feature extraction
- `sample_system_resource_statistics.py`: System resource statistics

## Running Tests

The comprehensive Python suite shares the same `test_res` fixture tree as the C++ API tests:

```bash
python -m sample_testcase.run --native-lib ../build/lib/libInspireFace.so
```

On Windows, run `python -m sample_testcase.run --native-lib ../build/windows-x64-Release-shared/install/InspireFace/lib/libInspireFace.dll` from this `python` directory after building the SDK and preparing the same test resources.

## Notes

1. Ensure that OpenCV and other necessary dependencies are installed on your system
2. Make sure the dynamic library is correctly installed before use
3. Python 3.7 or higher is required; Windows wheels require x64 Python
4. PyPI wheels use the CPU backend. GPU, CoreML, and NPU builds depend on the target platform; refer to the [documentation](https://doc.inspireface.online/guides/python-rockchip-device.html) for packaging a matching native library. Windows wheels currently use the CPU/MNN backend.
