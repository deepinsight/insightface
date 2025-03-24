import os
import cv2
import inspireface as isf
import numpy as np
import os
import cv2


def get_quality(image, session: isf.InspireFaceSession) -> float:
    select_exec_func = isf.HF_ENABLE_QUALITY | isf.HF_ENABLE_MASK_DETECT | isf.HF_ENABLE_LIVENESS | isf.HF_ENABLE_INTERACTION | isf.HF_ENABLE_FACE_ATTRIBUTE
    faces = session.face_detection(image)
    if len(faces) > 0:
        extends = session.face_pipeline(image, faces, select_exec_func)
        if len(faces) == 0:
            return 0
        for idx, ext in enumerate(extends):
            print(f"{'==' * 20}")
            print(f"idx: {idx}")
            # For these pipeline results, you can set thresholds based on the specific scenario to make judgments.
            print(f"quality: {ext.quality_confidence}")
            print(f"rgb liveness: {ext.rgb_liveness_confidence}")
            print(f"face mask: {ext.mask_confidence}")
            print(
                f"face eyes status: left eye: {ext.left_eye_status_confidence} right eye: {ext.right_eye_status_confidence}")
            print(f"gender: {ext.gender}")
            print(f"race: {ext.race}")
            print(f"age: {ext.age_bracket}")


if __name__ == "__main__":
    register_exec_func = isf.HF_ENABLE_QUALITY | isf.HF_ENABLE_MASK_DETECT | isf.HF_ENABLE_LIVENESS | isf.HF_ENABLE_INTERACTION | isf.HF_ENABLE_FACE_ATTRIBUTE
    session = isf.InspireFaceSession(register_exec_func, isf.HF_DETECT_MODE_ALWAYS_DETECT)
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if not ret:
            break
            
        quality = get_quality(frame, session)
        
        faces = session.face_detection(frame)
        for face in faces:
            x1, y1, x2, y2 = face.location
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cam.release()
    cv2.destroyAllWindows()