# InspireFace Python API

InspireFace provides an easy-to-use Python API that wraps the underlying dynamic link library through ctypes. You can install the latest release version via pip or configure it using the project's self-compiled dynamic library.

## Quick Installation

### Install via pip (Recommended)

```bash
pip install inspireface
```

### Manual Installation

1. First install the necessary dependencies:
```bash
pip install loguru tqdm opencv-python
```

2. Copy the compiled dynamic library to the specified directory:
```bash
# Copy the compiled dynamic library to the corresponding system architecture directory
cp YOUR_BUILD_DIR/libInspireFace.so inspireface/modules/core/SYSTEM/CORE_ARCH/
```

3. Install the Python package:
```bash
python setup.py install
```

## Quick Start

Here's a simple example showing how to use InspireFace for face detection and landmark drawing:

```python
import cv2
import inspireface as isf

# Create session with required features enabled
session = isf.InspireFaceSession(
    opt=isf.HF_ENABLE_NONE,  # Optional features
    detect_mode=isf.HF_DETECT_MODE_ALWAYS_DETECT  # Detection mode
)

# Set detection confidence threshold
session.set_detection_confidence_threshold(0.5)

# Read image
image = cv2.imread("path/to/your/image.jpg")
assert image is not None, "Please check if the image path is correct"

# Perform face detection
faces = session.face_detection(image)
print(f"Detected {len(faces)} faces")

# Draw detection results on image
draw = image.copy()
for idx, face in enumerate(faces):
    # Get face bounding box coordinates
    x1, y1, x2, y2 = face.location
    
    # Calculate rotated box parameters
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    size = (x2 - x1, y2 - y1)
    angle = face.roll
    
    # Draw rotated box
    rect = ((center[0], center[1]), (size[0], size[1]), angle)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    cv2.drawContours(draw, [box], 0, (100, 180, 29), 2)
    
    # Draw landmarks
    landmarks = session.get_face_dense_landmark(face)
    for x, y in landmarks.astype(int):
        cv2.circle(draw, (x, y), 0, (220, 100, 0), 2)
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

The project includes unit tests. You can adjust test content by modifying parameters in `test/test_settings.py`:

```bash
python -m unittest discover -s test
```

## Notes

1. Ensure that OpenCV and other necessary dependencies are installed on your system
2. Make sure the dynamic library is correctly installed before use
3. Python 3.7 or higher is recommended
4. The default version is CPU, if you want to use the GPU, CoreML, or NPU backend version, you can refer to the [documentation](https://doc.inspireface.online/guides/python-rockchip-device.html) to replace the so and make a Python installation package