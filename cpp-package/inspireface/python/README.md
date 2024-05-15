# PyInspireFace

## Setup Library

You need to compile the dynamic linking library in the main project and then place it in **inspireface/modules/core**.

```Bash
# copy or link
cp YOUR_BUILD_DIR/libInspireFace.so inspireface/modules/core
```

## Require

You need to install some dependencies beforehand.

```Bash
pip install loguru
pip install tqdm
pip install opencv-python
```

## Quick Start

You can easily call the api to implement a number of functions:

```Python
import cv2
import inspireface as ifac
from inspireface.param import *

# Step 1: Initialize the SDK and load the algorithm resource files.
resource_path = "pack/Pikachu"
ret = ifac.launch(resource_path)
assert ret, "Launch failure. Please ensure the resource path is correct."

# Optional features, loaded during session creation based on the modules specified.
opt = HF_ENABLE_NONE
session = ifac.InspireFaceSession(opt, HF_DETECT_MODE_IMAGE)

# Load the image using OpenCV.
image = cv2.imread(image_path)
assert image is not None, "Please check that the image path is correct."

# Perform face detection on the image.
faces = session.face_detection(image)
print(f"face detection: {len(faces)} found")

# Copy the image for drawing the bounding boxes.
draw = image.copy()
for idx, face in enumerate(faces):
    print(f"{'==' * 20}")
    print(f"idx: {idx}")
    # Print Euler angles of the face.
    print(f"roll: {face.roll}, yaw: {face.yaw}, pitch: {face.pitch}")
    # Draw bounding box around the detected face.
    x1, y1, x2, y2 = face.location
    cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 0, 255), 2)

```


You can also check out other sample files, which contain more diverse examples of functionality.

## Test


In the Python API, we have integrated a relatively simple unit test. You can adjust the content of the unit test by modifying the parameters in the configuration file **test/test_settings.py**.

```Bash
# Run total test
python -m unittest discover -s test
```

