import click
import cv2
import inspireface as ifac
from inspireface.param import *
import numpy as np

@click.command()
@click.argument("resource_path")
@click.argument('source')
@click.option('--show', is_flag=True, help='Display the video stream or video file in a window.')
def case_face_tracker_from_video(resource_path, source, show):
    """
    Launch a face tracking process from a video source. The 'source' can either be a webcam index (0, 1, ...)
    or a path to a video file. Use the --show option to display the video.

    Args:
        resource_path (str): Path to the resource directory for face tracking algorithms.
        source (str): Webcam index or path to the video file.
        show (bool): If set, the video will be displayed in a window.
    """
    # Initialize the face tracker or other resources.
    print(f"Initializing with resources from: {resource_path}")
    # Step 1: Initialize the SDK and load the algorithm resource files.
    ret = ifac.launch(resource_path)
    assert ret, "Launch failure. Please ensure the resource path is correct."

    # Optional features, loaded during session creation based on the modules specified.
    opt = HF_ENABLE_NONE
    session = ifac.InspireFaceSession(opt, HF_DETECT_MODE_LIGHT_TRACK)    # Use video mode

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

    # Main loop to process video frames.
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no more frames or error occurs.

        # Process frame here (e.g., face detection/tracking).
        faces = session.face_detection(frame)
        for idx, face in enumerate(faces):
            print(f"{'==' * 20}")
            print(f"idx: {idx}")
            # Print Euler angles of the face.
            print(f"roll: {face.roll}, yaw: {face.yaw}, pitch: {face.pitch}")

            # Get face bounding box
            x1, y1, x2, y2 = face.location

            # Calculate center, size, and angle
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            size = (x2 - x1, y2 - y1)
            angle = face.roll 

            # Get rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Apply rotation to the bounding box corners
            rect = ((center[0], center[1]), (size[0], size[1]), angle)
            box = cv2.boxPoints(rect)
            box = box.astype(int)

            # Draw the rotated bounding box
            cv2.drawContours(frame, [box], 0, (100, 180, 29), 2)

            # Draw landmarks
            lmk = session.get_face_dense_landmark(face)
            for x, y in lmk.astype(int):
                cv2.circle(frame, (x, y), 0, (220, 100, 0), 2)

        if show:
            cv2.imshow("Face Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit loop if 'q' is pressed.

    # Cleanup: release video capture and close any open windows.
    cap.release()
    cv2.destroyAllWindows()
    print("Released all resources and closed windows.")


if __name__ == '__main__':
    case_face_tracker_from_video()
