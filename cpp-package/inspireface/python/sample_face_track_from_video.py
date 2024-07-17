import time

import click
import cv2
import inspireface as ifac
from inspireface.param import *
import numpy as np


def generate_color(id):
    """
    Generate a bright color based on the given integer ID. Ensures 50 unique colors.

    Args:
        id (int): The ID for which to generate a color.

    Returns:
        tuple: A tuple representing the color in BGR format.
    """
    max_id = 50  # Number of unique colors
    id = id % max_id

    # Generate HSV color
    hue = int((id * 360 / max_id) % 360)  # Distribute hue values equally
    saturation = 200 + (55 * id) % 55  # High saturation for bright colors
    value = 200 + (55 * id) % 55  # High value for bright colors

    hsv_color = np.uint8([[[hue, saturation, value]]])
    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

    return (int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))

@click.command()
@click.argument("resource_path")
@click.argument('source')
@click.option('--show', is_flag=True, help='Display the video stream or video file in a window.')
@click.option('--out', type=str, default=None, help='Path to save the processed video.')
def case_face_tracker_from_video(resource_path, source, show, out):
    """
    Launch a face tracking process from a video source. The 'source' can either be a webcam index (0, 1, ...)
    or a path to a video file. Use the --show option to display the video.

    Args:
        resource_path (str): Path to the resource directory for face tracking algorithms.
        source (str): Webcam index or path to the video file.
        show (bool): If set, the video will be displayed in a window.
        out (str): Path to save the processed video.
    """
    # Initialize the face tracker or other resources.
    print(f"Initializing with resources from: {resource_path}")
    # Step 1: Initialize the SDK and load the algorithm resource files.
    ret = ifac.launch(resource_path)
    assert ret, "Launch failure. Please ensure the resource path is correct."

    # Optional features, loaded during session creation based on the modules specified.
    opt = HF_ENABLE_NONE | HF_ENABLE_INTERACTION
    session = ifac.InspireFaceSession(opt, HF_DETECT_MODE_ALWAYS_DETECT, max_detect_num=25, detect_pixel_level=320)    # Use video mode
    session.set_filter_minimum_face_pixel_size(0)
    # Determine if the source is a digital webcam index or a video file path.
    try:
        source_index = int(source)  # Try to convert source to an integer.
        cap = cv2.VideoCapture(source_index)
        print(f"Using webcam at index {source_index}.")
    except ValueError:
        # If conversion fails, treat source as a file path.
        cap = cv2.VideoCapture(source)
        print(f"Opening video file at {source}.")

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # VideoWriter to save the processed video if out is provided.
    if out:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_video = cv2.VideoWriter(out, fourcc, fps, (frame_width, frame_height))
        print(f"Saving video to: {out}")

    # Main loop to process video frames.
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no more frames or error occurs.

        # Process frame here (e.g., face detection/tracking).
        faces = session.face_detection(frame)

        exts = session.face_pipeline(frame, faces, HF_ENABLE_INTERACTION)
        print(exts)

        for idx, face in enumerate(faces):
            # Get face bounding box
            x1, y1, x2, y2 = face.location

            # Calculate center, size, and angle
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            size = (x2 - x1, y2 - y1)
            angle = face.roll

            # Apply rotation to the bounding box corners
            rect = ((center[0], center[1]), (size[0], size[1]), angle)
            box = cv2.boxPoints(rect)
            box = box.astype(int)

            color = generate_color(face.track_id)

            # Draw the rotated bounding box
            cv2.drawContours(frame, [box], 0, color, 4)

            # Draw landmarks
            lmk = session.get_face_dense_landmark(face)
            for x, y in lmk.astype(int):
                cv2.circle(frame, (x, y), 0, color, 4)

            # Draw track ID at the top of the bounding box
            text = f"ID: {face.track_id}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_x = min(box[:, 0])
            text_y = min(box[:, 1]) - 10
            if text_y < 0:
                text_y = min(box[:, 1]) + text_size[1] + 10
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if show:
            cv2.imshow("Face Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit loop if 'q' is pressed.

        if out:
            out_video.write(frame)

    # Cleanup: release video capture and close any open windows.
    cap.release()
    if out:
        out_video.release()
    cv2.destroyAllWindows()
    print("Released all resources and closed windows.")


if __name__ == '__main__':
    case_face_tracker_from_video()
