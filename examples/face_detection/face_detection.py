import cv2
from insightface.app import FaceAnalysis

# Initialize face analysis model
app = FaceAnalysis(
    name="buffalo_l", providers=["CPUExecutionProvider"]
)  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU


def detect_faces(image_path):
    """Detect faces in an image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = app.get(img)

    if len(faces) < 1:
        print("No faces detected")
        return img

    # Draw rectangles on detected faces
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add confidence score
        confidence = face.det_score
        cv2.putText(
            img,
            f"{confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    print(f"Detected {len(faces)} face(s)")
    return img


# Path to your image
image_path = "path/to/image.jpg"

try:
    result = detect_faces(image_path)

    # Display result
    cv2.imshow("Face Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error: {str(e)}")
