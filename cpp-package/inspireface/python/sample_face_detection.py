import os
import cv2
import inspireface as ifac
from inspireface.param import *
import click

@click.command()
@click.argument("resource_path")
@click.argument('image_path')
def case_face_detection_image(resource_path, image_path):
    """
    This is a sample application for face detection and tracking using an image.
    It also includes pipeline extensions such as RGB liveness, mask detection, and face quality evaluation.
    """
    # Step 1: Initialize the SDK and load the algorithm resource files.
    ret = ifac.launch(resource_path)
    assert ret, "Launch failure. Please ensure the resource path is correct."

    # Optional features, loaded during session creation based on the modules specified.
    opt = HF_ENABLE_FACE_RECOGNITION | HF_ENABLE_QUALITY | HF_ENABLE_MASK_DETECT | HF_ENABLE_LIVENESS
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

    # Features must be enabled during session creation to use them here.
    select_exec_func = HF_ENABLE_QUALITY | HF_ENABLE_MASK_DETECT | HF_ENABLE_LIVENESS
    # Execute the pipeline to obtain richer face information.
    extends = session.face_pipeline(image, faces, select_exec_func)
    for idx, ext in enumerate(extends):
        print(f"{'==' * 20}")
        print(f"idx: {idx}")
        # For these pipeline results, you can set thresholds based on the specific scenario to make judgments.
        print(f"quality: {ext.quality_confidence}")
        print(f"rgb liveness: {ext.rgb_liveness_confidence}")
        print(f"face mask: {ext.mask_confidence}")

    # Save the annotated image to the 'tmp/' directory.
    save_path = os.path.join("tmp/", "det.jpg")
    cv2.imwrite(save_path, draw)
    print(f"\nSave annotated image to {save_path}")


if __name__ == '__main__':
    os.makedirs("tmp", exist_ok=True)
    case_face_detection_image()
