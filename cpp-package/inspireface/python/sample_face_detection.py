import os
import cv2
import inspireface as isf
import click

race_tags = ["Black", "Asian", "Latino/Hispanic", "Middle Eastern", "White"]
gender_tags = ["Female", "Male"]
age_bracket_tags = [
    "0-2 years old", "3-9 years old", "10-19 years old", "20-29 years old", "30-39 years old",
    "40-49 years old", "50-59 years old", "60-69 years old", "more than 70 years old"
]
emotion_tags = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]

@click.command()
@click.argument('image_path')
@click.option('--show', is_flag=True, help='Display the image with detected faces.')
def case_face_detection_image(image_path, show):
    """
    This is a sample application for face detection and tracking using an image.
    It also includes pipeline extensions such as RGB liveness, mask detection, and face quality evaluation.
    """
    isf.switch_image_processing_backend(isf.HF_IMAGE_PROCESSING_CPU)
    opt = isf.HF_ENABLE_FACE_RECOGNITION | isf.HF_ENABLE_QUALITY | isf.HF_ENABLE_MASK_DETECT | \
          isf.HF_ENABLE_LIVENESS | isf.HF_ENABLE_INTERACTION | isf.HF_ENABLE_FACE_ATTRIBUTE | isf.HF_ENABLE_FACE_EMOTION
    session = isf.InspireFaceSession(opt, isf.HF_DETECT_MODE_ALWAYS_DETECT)
    session.set_detection_confidence_threshold(0.5)
    # Load image
    image = cv2.imread(image_path)
    assert image is not None, "Please check that the image path is correct."

    # Dynamic drawing parameters (adjusted to image size)
    h, w = image.shape[:2]
    scale = max(w, h) / 1000.0
    line_thickness = max(1, int(2 * scale))
    circle_radius = max(1, int(1.5 * scale))
    font_scale = 0.5 * scale

    # Detect faces
    faces = session.face_detection(image)
    print(faces)
    print(f"face detection: {len(faces)} found")

    draw = image.copy()
    for idx, face in enumerate(faces):
        print(f"{'==' * 20}")
        print(f"idx: {idx}")
        print(f"detection confidence: {face.detection_confidence}")
        print(f"roll: {face.roll}, yaw: {face.yaw}, pitch: {face.pitch}")

        x1, y1, x2, y2 = face.location
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        size = (x2 - x1, y2 - y1)
        angle = face.roll

        rect = ((center[0], center[1]), (size[0], size[1]), angle)
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(draw, [box], 0, (100, 180, 29), line_thickness)

        # Draw landmark
        lmk = session.get_face_dense_landmark(face)
        for x, y in lmk.astype(int):
            cv2.circle(draw, (x, y), circle_radius, (220, 100, 0), -1)

        # Optional: Add detection confidence (text) on the face box
        # label = f"{face.detection_confidence:.2f}"
        # cv2.putText(draw, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
        #             font_scale, (255, 255, 255), line_thickness)

    # Execute extended functions (optional modules)
    select_exec_func = isf.HF_ENABLE_QUALITY | isf.HF_ENABLE_MASK_DETECT | \
                       isf.HF_ENABLE_LIVENESS | isf.HF_ENABLE_INTERACTION | isf.HF_ENABLE_FACE_ATTRIBUTE | isf.HF_ENABLE_FACE_EMOTION
    extends = session.face_pipeline(image, faces, select_exec_func)
    print(extends)
    for idx, ext in enumerate(extends):
        print(f"{'==' * 20}")
        print(f"idx: {idx}")
        print(f"quality: {ext.quality_confidence}")
        print(f"rgb liveness: {ext.rgb_liveness_confidence}")
        print(f"face mask: {ext.mask_confidence}")
        print(f"face eyes status: left eye: {ext.left_eye_status_confidence} right eye: {ext.right_eye_status_confidence}")
        print(f"gender: {gender_tags[ext.gender]}")
        print(f"race: {race_tags[ext.race]}")
        print(f"age: {age_bracket_tags[ext.age_bracket]}")
        print(f"emotion: {emotion_tags[ext.emotion]}")

    # Save the annotated image
    save_path = os.path.join("tmp", "det.jpg")
    os.makedirs("tmp", exist_ok=True)
    cv2.imwrite(save_path, draw)
    if show:
        cv2.imshow("Face Detection", draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(f"\nSave annotated image to {save_path}")

if __name__ == '__main__':
    case_face_detection_image()
