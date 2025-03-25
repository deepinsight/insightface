import cv2
import inspireface as isf
import click

@click.command()
@click.argument('image_path1')
@click.argument('image_path2') 
def case_face_comparison(image_path1, image_path2):
    """
    This is a sample application for comparing two face images.
    Args:
        image_path1 (str): Path to the first face image
        image_path2 (str): Path to the second face image
    """
    # Enable face recognition features
    opt = isf.HF_ENABLE_FACE_RECOGNITION
    session = isf.InspireFaceSession(opt, isf.HF_DETECT_MODE_ALWAYS_DETECT)

    # Load and check the first image
    image1 = cv2.imread(image_path1)
    assert image1 is not None, "Failed to load first image"
    
    # Load and check the second image  
    image2 = cv2.imread(image_path2)
    assert image2 is not None, "Failed to load second image"

    # Detect faces in first image
    faces1 = session.face_detection(image1)
    assert faces1, "No face detected in first image"
    face1 = faces1[0]  # Use the first detected face

    # Detect faces in second image
    faces2 = session.face_detection(image2)
    assert faces2, "No face detected in second image"
    face2 = faces2[0]  # Use the first detected face

    # Extract features
    feature1 = session.face_feature_extract(image1, face1)
    feature2 = session.face_feature_extract(image2, face2)

    # Calculate similarity score between the two faces
    similarity = isf.feature_comparison(feature1, feature2)
    
    print(f"The cosine similarity score: {similarity:.4f}")
    print(f"{'Same person' if similarity > isf.get_recommended_cosine_threshold() else 'Different person'}")

    percentage = isf.cosine_similarity_convert_to_percentage(similarity)
    print(f"The percentage similarity: {percentage:.4f}")


if __name__ == '__main__':
    case_face_comparison()
