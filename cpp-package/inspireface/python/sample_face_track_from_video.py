import time

import click
import cv2
import inspireface as isf
import numpy as np
import time

def generate_color(id):
    """
    Generate a bright color based on the given integer ID. Ensures 50 unique colors.

    Args:
        id (int): The ID for which to generate a color.

    Returns:
        tuple: A tuple representing the color in BGR format.
    """
    # Handle invalid ID (-1)
    if id < 0:
        return (128, 128, 128)  # Return gray color for invalid ID
        
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
@click.argument('source')
@click.option('--show', is_flag=True, help='Display the video stream or video file in a window.')
@click.option('--out', type=str, default=None, help='Path to save the processed video.')
def case_face_tracker_from_video(source, show, out):
    """
    Launch a face tracking process from a video source. The 'source' can either be a webcam index (0, 1, ...)
    or a path to a video file. Use the --show option to display the video.

    Args:
        resource_path (str): Path to the resource directory for face tracking algorithms.
        source (str): Webcam index or path to the video file.
        show (bool): If set, the video will be displayed in a window.
        out (str): Path to save the processed video.
    """
    # Optional features, loaded during session creation based on the modules specified.
    opt = isf.HF_ENABLE_NONE | isf.HF_ENABLE_INTERACTION
    session = isf.InspireFaceSession(opt, isf.HF_DETECT_MODE_LIGHT_TRACK, max_detect_num=25, detect_pixel_level=320)    # Use video mode
    session.set_track_mode_smooth_ratio(0.06)
    session.set_track_mode_num_smooth_cache_frame(15)
    session.set_filter_minimum_face_pixel_size(0)
    session.set_track_model_detect_interval(0)
    session.set_track_lost_recovery_mode(True)
    session.set_enable_track_cost_spend(True)
    # Determine if the source is a digital webcam index or a video file path.
    try:
        source_index = int(source)  # Try to convert source to an integer.
        print(f"Using webcam at index {source_index}.")
        cap = cv2.VideoCapture(source_index)
    except ValueError:
        # If conversion fails, treat source as a file path.
        print(f"Opening video file at {source}.")
        cap = cv2.VideoCapture(source)

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
        t1 = time.time()
        faces = session.face_detection(frame)
        t2 = time.time()
        # print(f"Face detection time: {t2 - t1} seconds")
        session.print_track_cost_spend()
        exts = session.face_pipeline(frame, faces, isf.HF_ENABLE_INTERACTION)

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

            actions = []
            if exts[idx].action_normal:
                actions.append("Normal")
            if exts[idx].action_jaw_open:
                actions.append("Jaw Open") 
            if exts[idx].action_shake:
                actions.append("Shake")
            if exts[idx].action_blink:
                actions.append("Blink")
            if exts[idx].action_head_raise:
                actions.append("Head Raise")
            print("Actions:", actions)
            
            color = generate_color(face.track_id)

            # Draw the rotated bounding box
            cv2.drawContours(frame, [box], 0, color, 4)

            # Draw landmarks
            lmk = session.get_face_dense_landmark(face)
            for x, y in lmk.astype(int):
                cv2.circle(frame, (x, y), 0, color, 4)

            five_key_points = session.get_face_five_key_points(face)
            for x, y in five_key_points.astype(int):
                cv2.circle(frame, (x, y), 0, (255-color[0], 255-color[1], 255-color[2]), 6)

            # Draw track ID at the top of the bounding box
            text = f"ID: {face.track_id}, Count: {face.track_count}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_x = min(box[:, 0])
            text_y = min(box[:, 1]) - 10
            if text_y < 0:
                text_y = min(box[:, 1]) + text_size[1] + 10
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if show:
            cv2.imshow("Face Tracker", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
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
