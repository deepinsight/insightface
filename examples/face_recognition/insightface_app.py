import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis

# Initialize face analysis model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU

def get_face_embedding(image_path):
    """Extract face embedding from an image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    faces = app.get(img)
    
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")
    
    return faces[0].embedding

def compare_faces(emb1, emb2, threshold=0.65): # Adjust this threshold according to your usecase.
    """Compare two embeddings using cosine similarity"""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity, similarity > threshold

# Paths to your Indian face images
image1_path = "path/to/face1.jpg"
image2_path = "path/to/face2.jpg"

try:
    # Get embeddings
    emb1 = get_face_embedding(image1_path)
    emb2 = get_face_embedding(image2_path)
    
    # Compare faces
    similarity_score, is_same_person = compare_faces(emb1, emb2)
    
    print(f"Similarity Score: {similarity_score:.4f}")
    print(f"Same person? {'YES' if is_same_person else 'NO'}")
    
except Exception as e:
    print(f"Error: {str(e)}")
