#!/usr/bin/env python3
"""
Demo: Age and Gender Detection using dlib
Commercial Ad Signage Application

This demo shows how to:
1. Analyze a single image
2. Display demographics statistics
3. Save annotated output
4. Measure performance

License: CC0 v1.0 Universal (Public Domain)
Commercial Use: ALLOWED
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path
script_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(script_dir))

from face_analyzer import FaceAnalyzer


def demo_single_image(image_path: str, models_dir: str = "../models"):
    """
    Demo: Analyze a single image and display results

    Args:
        image_path: Path to test image
        models_dir: Directory containing dlib .dat files
    """
    print("=" * 60)
    print("dlib Age & Gender Detection Demo")
    print("Commercial Ad Signage Application")
    print("=" * 60)
    print()

    # Initialize analyzer
    print("Initializing FaceAnalyzer...")
    analyzer = FaceAnalyzer(models_dir=models_dir, upsample_num=0)
    print()

    # Analyze image
    print(f"Analyzing image: {image_path}")
    print("-" * 60)
    results = analyzer.analyze_image(image_path, return_timing=True)
    print()

    # Display results
    if not results:
        print("No faces detected in the image.")
        return

    print(f"Detected {len(results)} face(s):\n")

    for face in results:
        print(f"Face #{face['id'] + 1}:")
        print(f"  Gender:     {face['gender']}")
        print(f"  Age:        {face['age']:.1f} years")
        print(f"  Confidence: {face['confidence']:.3f}")
        print(f"  BBox:       {face['bbox']}")

        if 'timing' in face:
            timing = face['timing']
            if 'detection_ms' in timing:
                print(f"  Detection:  {timing['detection_ms']:.1f}ms")
            print(f"  Gender:     {timing['gender_ms']:.1f}ms")
            print(f"  Age:        {timing['age_ms']:.1f}ms")
        print()

    # Statistics
    print("=" * 60)
    print("DEMOGRAPHIC STATISTICS")
    print("=" * 60)
    stats = analyzer.get_statistics(results)

    print(f"\nTotal Faces: {stats['total_faces']}")
    print(f"\nGender Distribution:")
    print(f"  Male:   {stats['male_count']:2d} ({stats['male_count']/stats['total_faces']*100:5.1f}%)")
    print(f"  Female: {stats['female_count']:2d} ({stats['female_count']/stats['total_faces']*100:5.1f}%)")

    print(f"\nAverage Age: {stats['avg_age']:.1f} years")

    print(f"\nAge Distribution:")
    for age_range, count in stats['age_distribution'].items():
        percentage = count / stats['total_faces'] * 100
        bar = "█" * int(percentage / 5)  # Scale to 20 chars max
        print(f"  {age_range:>6}: {count:2d} ({percentage:5.1f}%) {bar}")

    print()

    # Save annotated output
    output_path = image_path.replace('.', '_annotated.')
    print(f"Saving annotated image: {output_path}")
    analyzer.draw_results(image_path, output_path)
    print()

    # Performance summary
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    if results and 'timing' in results[0]:
        timing = results[0]['timing']
        detection_time = timing.get('detection_ms', 0)
        avg_face_time = sum(
            r['timing']['gender_ms'] + r['timing']['age_ms']
            for r in results
        ) / len(results)
        total_time = detection_time + sum(
            r['timing']['gender_ms'] + r['timing']['age_ms']
            for r in results
        )

        print(f"\nDetection Time:      {detection_time:.1f}ms")
        print(f"Avg per face:        {avg_face_time:.1f}ms")
        print(f"Total Time:          {total_time:.1f}ms")
        print(f"Throughput:          {1000/total_time:.1f} images/sec")

    print()


def demo_batch_processing(image_dir: str, models_dir: str = "../models"):
    """
    Demo: Process multiple images in a directory

    Args:
        image_dir: Directory containing images
        models_dir: Directory containing dlib .dat files
    """
    print("=" * 60)
    print("Batch Processing Demo")
    print("=" * 60)
    print()

    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_dir = Path(image_dir)
    images = [
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(images)} images")
    print()

    # Initialize analyzer
    print("Initializing FaceAnalyzer...")
    analyzer = FaceAnalyzer(models_dir=models_dir, upsample_num=0)
    print()

    # Process all images
    all_results = []
    for i, image_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Processing: {image_path.name}...", end=" ")

        try:
            results = analyzer.analyze_image(str(image_path))
            all_results.extend(results)
            print(f"✅ {len(results)} face(s)")
        except Exception as e:
            print(f"❌ Error: {e}")

    print()

    # Overall statistics
    if all_results:
        print("=" * 60)
        print("OVERALL STATISTICS")
        print("=" * 60)
        stats = analyzer.get_statistics(all_results)

        print(f"\nTotal Images:  {len(images)}")
        print(f"Total Faces:   {stats['total_faces']}")
        print(f"Avg per image: {stats['total_faces'] / len(images):.1f}")

        print(f"\nGender Distribution:")
        print(f"  Male:   {stats['male_count']} ({stats['male_count']/stats['total_faces']*100:.1f}%)")
        print(f"  Female: {stats['female_count']} ({stats['female_count']/stats['total_faces']*100:.1f}%)")

        print(f"\nAverage Age: {stats['avg_age']:.1f} years")
        print(f"\nAge Distribution: {stats['age_distribution']}")
        print()
    else:
        print("No faces detected in any images.")


def demo_video_frame(video_path: str, models_dir: str = "../models"):
    """
    Demo: Process video frames (analyze every Nth frame)

    Args:
        video_path: Path to video file or camera index (0, 1, ...)
        models_dir: Directory containing dlib .dat files
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Install with: pip install opencv-python")
        return

    print("=" * 60)
    print("Video Processing Demo")
    print("=" * 60)
    print()

    # Initialize analyzer
    print("Initializing FaceAnalyzer...")
    analyzer = FaceAnalyzer(models_dir=models_dir, upsample_num=0)
    print()

    # Open video
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
        source = f"Camera {video_path}"
    else:
        cap = cv2.VideoCapture(video_path)
        source = video_path

    if not cap.isOpened():
        print(f"Error: Could not open {source}")
        return

    print(f"Processing: {source}")
    print("Press 'q' to quit, 's' to save current frame")
    print()

    frame_count = 0
    process_every = 5  # Process every 5th frame for performance

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process only every Nth frame
        if frame_count % process_every == 0:
            results = analyzer.analyze_frame(frame, return_timing=True)

            # Draw results
            for face in results:
                x, y, w, h = face['bbox']

                # Draw bbox
                color = (0, 255, 0) if face['gender'] == 'Male' else (255, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Draw label
                label = f"{face['gender']}, {face['age']:.0f}y"
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Show stats
            stats_text = f"Faces: {len(results)} | Frame: {frame_count}"
            cv2.putText(frame, stats_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display
        cv2.imshow('Age & Gender Detection', frame)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = f"frame_{frame_count:05d}.jpg"
            cv2.imwrite(save_path, frame)
            print(f"Saved: {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nVideo processing completed.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='dlib Age & Gender Detection Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python demo_age_gender.py --image crowd.jpg

  # Batch processing
  python demo_age_gender.py --batch ./test_images/

  # Video file
  python demo_age_gender.py --video demo.mp4

  # Webcam (camera 0)
  python demo_age_gender.py --video 0

  # Custom models directory
  python demo_age_gender.py --image crowd.jpg --models /path/to/models
        """
    )

    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, help='Directory with multiple images')
    parser.add_argument('--video', type=str, help='Path to video file or camera index (0, 1, ...)')
    parser.add_argument('--models', type=str, default='../models',
                       help='Directory containing dlib .dat model files (default: ../models)')

    args = parser.parse_args()

    # Check if models exist
    models_dir = Path(__file__).parent.parent / args.models
    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        print("\nPlease run: bash scripts/download_models.sh")
        sys.exit(1)

    required_models = [
        'mmod_human_face_detector.dat',
        'shape_predictor_5_face_landmarks.dat',
        'dnn_gender_classifier_v1.dat',
        'dnn_age_predictor_v1.dat'
    ]

    missing_models = [m for m in required_models if not (models_dir / m).exists()]
    if missing_models:
        print("Error: Missing model files:")
        for m in missing_models:
            print(f"  - {m}")
        print("\nPlease run: bash scripts/download_models.sh")
        sys.exit(1)

    # Run appropriate demo
    if args.image:
        demo_single_image(args.image, str(models_dir))
    elif args.batch:
        demo_batch_processing(args.batch, str(models_dir))
    elif args.video:
        demo_video_frame(args.video, str(models_dir))
    else:
        parser.print_help()
        print("\nError: Please specify --image, --batch, or --video")
        sys.exit(1)
