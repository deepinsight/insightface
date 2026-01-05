"""
dlib-based Face Analysis for Commercial Ad Signage
Age and Gender Detection using Pre-trained Models

License: CC0 v1.0 Universal (Public Domain)
Commercial Use: ✅ ALLOWED

Models:
- Face Detection: mmod_human_face_detector.dat
- Gender Classification: dnn_gender_classifier_v1.dat (97.3% accuracy)
- Age Prediction: dnn_age_predictor_v1.dat (SOTA)
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    import dlib
except ImportError:
    raise ImportError(
        "dlib not installed. Install with: pip install dlib\n"
        "Or build from source for better performance: https://github.com/davisking/dlib"
    )

try:
    import cv2
except ImportError:
    raise ImportError("OpenCV not installed. Install with: pip install opencv-python")


class FaceAnalyzer:
    """
    Complete face analysis pipeline using dlib pre-trained models

    Features:
    - Face detection (CNN-based, high accuracy)
    - Gender classification (97.3% accuracy on LFW)
    - Age prediction (state-of-the-art)
    - Optimized for real-time ad signage applications

    Usage:
        analyzer = FaceAnalyzer(models_dir="models/")
        results = analyzer.analyze_image("crowd.jpg")

        for face in results:
            print(f"Person {face['id']}: Age {face['age']}, Gender {face['gender']}")
    """

    def __init__(self, models_dir: str = "models", upsample_num: int = 0):
        """
        Initialize face analyzer with pre-trained models

        Args:
            models_dir: Directory containing .dat model files
            upsample_num: Number of times to upsample image for detection
                         0 = no upsampling (fastest, may miss small faces)
                         1 = 1x upsampling (2x slower, better for distant faces)
                         2 = 2x upsampling (4x slower, catches very small faces)
        """
        self.models_dir = Path(models_dir)
        self.upsample_num = upsample_num

        # Load models
        print("Loading dlib models...")
        self._load_models()
        print("✅ All models loaded successfully!")

    def _load_models(self):
        """Load all required dlib models"""

        # Face detector
        detector_path = self.models_dir / "mmod_human_face_detector.dat"
        if not detector_path.exists():
            raise FileNotFoundError(
                f"Face detector not found: {detector_path}\n"
                f"Run: bash scripts/download_models.sh"
            )
        self.face_detector = dlib.cnn_face_detection_model_v1(str(detector_path))
        print(f"  ✓ Face detector loaded: {detector_path.name}")

        # Shape predictor (for alignment)
        shape_path = self.models_dir / "shape_predictor_5_face_landmarks.dat"
        if not shape_path.exists():
            raise FileNotFoundError(
                f"Shape predictor not found: {shape_path}\n"
                f"Run: bash scripts/download_models.sh"
            )
        self.shape_predictor = dlib.shape_predictor(str(shape_path))
        print(f"  ✓ Shape predictor loaded: {shape_path.name}")

        # Gender classifier
        gender_path = self.models_dir / "dnn_gender_classifier_v1.dat"
        if not gender_path.exists():
            raise FileNotFoundError(
                f"Gender classifier not found: {gender_path}\n"
                f"Run: bash scripts/download_models.sh"
            )
        self.gender_classifier = dlib.cnn_gender_classifier(str(gender_path))
        print(f"  ✓ Gender classifier loaded: {gender_path.name}")

        # Age predictor
        age_path = self.models_dir / "dnn_age_predictor_v1.dat"
        if not age_path.exists():
            raise FileNotFoundError(
                f"Age predictor not found: {age_path}\n"
                f"Run: bash scripts/download_models.sh"
            )
        self.age_predictor = dlib.age_predictor(str(age_path))
        print(f"  ✓ Age predictor loaded: {age_path.name}")

    def analyze_image(self, image_path: str, return_timing: bool = False) -> List[Dict]:
        """
        Analyze all faces in an image

        Args:
            image_path: Path to image file
            return_timing: If True, include timing information

        Returns:
            List of dictionaries, one per detected face:
            [
                {
                    'id': 0,
                    'bbox': (x, y, w, h),
                    'confidence': 0.98,
                    'gender': 'Male',
                    'gender_confidence': 0.95,
                    'age': 28.5,
                    'timing': {'detection': 15.2, 'gender': 12.3, 'age': 11.8}  # Optional
                },
                ...
            ]
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        return self.analyze_frame(img, return_timing=return_timing)

    def analyze_frame(self, frame: np.ndarray, return_timing: bool = False) -> List[Dict]:
        """
        Analyze all faces in a video frame or image array

        Args:
            frame: BGR image as numpy array (from cv2.imread or camera)
            return_timing: If True, include timing information

        Returns:
            List of face analysis results (same format as analyze_image)
        """
        timing = {}

        # Convert BGR to RGB (dlib expects RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        t0 = time.time()
        detections = self.face_detector(rgb, self.upsample_num)
        timing['detection'] = (time.time() - t0) * 1000  # Convert to ms

        results = []
        for i, detection in enumerate(detections):
            face_result = {
                'id': i,
                'bbox': self._get_bbox(detection),
                'confidence': detection.confidence,
            }

            # Get face landmarks for alignment
            shape = self.shape_predictor(rgb, detection.rect)

            # Get aligned face chip (150x150 RGB)
            face_chip = dlib.get_face_chip(rgb, shape, size=150)

            # Predict gender
            t0 = time.time()
            gender_prediction = self.gender_classifier(face_chip)
            timing_gender = (time.time() - t0) * 1000

            # gender_prediction is 0 or 1
            # According to dlib docs: 0 = female, 1 = male
            face_result['gender'] = 'Male' if gender_prediction == 1 else 'Female'
            face_result['gender_value'] = int(gender_prediction)

            # Predict age
            t0 = time.time()
            age = self.age_predictor(face_chip)
            timing_age = (time.time() - t0) * 1000

            face_result['age'] = float(age)

            if return_timing:
                face_result['timing'] = {
                    'gender_ms': timing_gender,
                    'age_ms': timing_age
                }

            results.append(face_result)

        # Add overall timing
        if return_timing and results:
            results[0]['timing']['detection_ms'] = timing['detection']

        return results

    def _get_bbox(self, detection) -> Tuple[int, int, int, int]:
        """Convert dlib detection to (x, y, w, h) bbox"""
        rect = detection.rect
        x = rect.left()
        y = rect.top()
        w = rect.right() - rect.left()
        h = rect.bottom() - rect.top()
        return (x, y, w, h)

    def draw_results(self, image_path: str, output_path: str,
                     font_scale: float = 0.6, thickness: int = 2):
        """
        Analyze image and draw results (bboxes + age/gender labels)

        Args:
            image_path: Input image path
            output_path: Output image path
            font_scale: Font size for labels
            thickness: Line thickness for boxes
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Analyze
        results = self.analyze_frame(img)

        # Draw results
        for face in results:
            x, y, w, h = face['bbox']

            # Draw bbox
            color = (0, 255, 0) if face['gender'] == 'Male' else (255, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

            # Draw label
            label = f"{face['gender']}, {face['age']:.0f}y"
            label_y = y - 10 if y > 30 else y + h + 20

            # Draw background for text
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(img, (x, label_y - text_h - 5),
                         (x + text_w, label_y + 5), color, -1)

            # Draw text
            cv2.putText(img, label, (x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       (255, 255, 255), thickness)

        # Save output
        cv2.imwrite(output_path, img)
        print(f"✅ Saved annotated image: {output_path}")
        print(f"   Detected {len(results)} face(s)")

    def get_statistics(self, results: List[Dict]) -> Dict:
        """
        Calculate demographic statistics from analysis results

        Args:
            results: List of face analysis results

        Returns:
            Dictionary with statistics:
            {
                'total_faces': 10,
                'male_count': 6,
                'female_count': 4,
                'avg_age': 32.5,
                'age_distribution': {
                    '0-18': 1,
                    '19-35': 5,
                    '36-60': 3,
                    '60+': 1
                }
            }
        """
        if not results:
            return {
                'total_faces': 0,
                'male_count': 0,
                'female_count': 0,
                'avg_age': 0,
                'age_distribution': {}
            }

        stats = {
            'total_faces': len(results),
            'male_count': sum(1 for f in results if f['gender'] == 'Male'),
            'female_count': sum(1 for f in results if f['gender'] == 'Female'),
            'avg_age': sum(f['age'] for f in results) / len(results),
        }

        # Age distribution
        age_ranges = {'0-18': 0, '19-35': 0, '36-60': 0, '60+': 0}
        for face in results:
            age = face['age']
            if age <= 18:
                age_ranges['0-18'] += 1
            elif age <= 35:
                age_ranges['19-35'] += 1
            elif age <= 60:
                age_ranges['36-60'] += 1
            else:
                age_ranges['60+'] += 1

        stats['age_distribution'] = age_ranges

        return stats


if __name__ == '__main__':
    # Quick test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python face_analyzer.py <image_path>")
        print("Example: python face_analyzer.py test_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    # Initialize analyzer
    analyzer = FaceAnalyzer(models_dir="../models")

    # Analyze image
    print(f"\nAnalyzing: {image_path}")
    results = analyzer.analyze_image(image_path, return_timing=True)

    # Print results
    print(f"\n✅ Detected {len(results)} face(s):\n")
    for face in results:
        print(f"Face {face['id']}:")
        print(f"  Gender: {face['gender']}")
        print(f"  Age: {face['age']:.1f} years")
        print(f"  Bbox: {face['bbox']}")
        print(f"  Confidence: {face['confidence']:.3f}")
        if 'timing' in face:
            print(f"  Timing: {face['timing']}")
        print()

    # Statistics
    stats = analyzer.get_statistics(results)
    print("Demographics:")
    print(f"  Total: {stats['total_faces']}")
    print(f"  Male: {stats['male_count']} ({stats['male_count']/stats['total_faces']*100:.1f}%)")
    print(f"  Female: {stats['female_count']} ({stats['female_count']/stats['total_faces']*100:.1f}%)")
    print(f"  Avg Age: {stats['avg_age']:.1f} years")
    print(f"  Age Distribution: {stats['age_distribution']}")
