#!/usr/bin/env python3
"""
Batch Processing Script for Ad Signage Analytics
Process multiple images and generate comprehensive report

Usage:
    python batch_process.py --input ./images/ --output ./results/

License: CC0 v1.0 Universal (Public Domain)
Commercial Use: ALLOWED
"""

import sys
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add scripts directory to path
script_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(script_dir))

from face_analyzer import FaceAnalyzer


def process_directory(input_dir: str, output_dir: str, models_dir: str = "../models"):
    """
    Process all images in a directory and generate reports

    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output (annotated images + reports)
        models_dir: Directory containing dlib .dat files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directories
    annotated_dir = output_path / "annotated"
    reports_dir = output_path / "reports"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [
        f for f in input_path.rglob('*')
        if f.suffix.lower() in image_extensions and f.is_file()
    ]

    if not images:
        print(f"No images found in {input_dir}")
        return

    print("=" * 70)
    print("BATCH PROCESSING - Ad Signage Analytics")
    print("=" * 70)
    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Images: {len(images)}")
    print()

    # Initialize analyzer
    print("Initializing FaceAnalyzer...")
    analyzer = FaceAnalyzer(models_dir=models_dir, upsample_num=0)
    print()

    # Process all images
    all_results = []
    image_stats = []
    failed_images = []

    print("Processing images:")
    print("-" * 70)

    for i, image_path in enumerate(images, 1):
        try:
            # Analyze image
            results = analyzer.analyze_image(str(image_path), return_timing=True)

            # Store per-image statistics
            stats = analyzer.get_statistics(results)
            image_stats.append({
                'filename': image_path.name,
                'path': str(image_path.relative_to(input_path)),
                'faces': len(results),
                'male': stats['male_count'],
                'female': stats['female_count'],
                'avg_age': stats['avg_age'],
                'age_dist': stats['age_distribution']
            })

            # Add image source to each result
            for result in results:
                result['source_image'] = image_path.name

            all_results.extend(results)

            # Save annotated image
            output_image = annotated_dir / image_path.name
            analyzer.draw_results(str(image_path), str(output_image))

            # Progress
            status = f"✅ {len(results):2d} face(s)"
            print(f"[{i:3d}/{len(images)}] {image_path.name:40s} {status}")

        except Exception as e:
            failed_images.append({
                'filename': image_path.name,
                'error': str(e)
            })
            print(f"[{i:3d}/{len(images)}] {image_path.name:40s} ❌ Error: {e}")

    print()

    # Generate reports
    print("=" * 70)
    print("GENERATING REPORTS")
    print("=" * 70)
    print()

    # 1. Overall statistics
    overall_stats = analyzer.get_statistics(all_results)
    _save_overall_report(reports_dir, overall_stats, len(images), len(failed_images))

    # 2. Per-image CSV
    _save_image_csv(reports_dir, image_stats)

    # 3. Detailed JSON
    _save_detailed_json(reports_dir, all_results, image_stats, failed_images)

    # 4. Summary
    _print_summary(overall_stats, len(images), len(failed_images), output_path)


def _save_overall_report(reports_dir: Path, stats: Dict, total_images: int, failed: int):
    """Save overall statistics report"""
    report_path = reports_dir / "overall_statistics.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("OVERALL STATISTICS - Ad Signage Analytics\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"Total Images Processed: {total_images}\n")
        f.write(f"Failed Images:          {failed}\n")
        f.write(f"Total Faces Detected:   {stats['total_faces']}\n")
        f.write(f"Average per Image:      {stats['total_faces'] / max(total_images - failed, 1):.1f}\n\n")

        f.write("GENDER DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        if stats['total_faces'] > 0:
            male_pct = stats['male_count'] / stats['total_faces'] * 100
            female_pct = stats['female_count'] / stats['total_faces'] * 100
            f.write(f"Male:   {stats['male_count']:4d} ({male_pct:5.1f}%)\n")
            f.write(f"Female: {stats['female_count']:4d} ({female_pct:5.1f}%)\n\n")
        else:
            f.write("No faces detected\n\n")

        f.write("AGE STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Average Age: {stats['avg_age']:.1f} years\n\n")

        f.write("AGE DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        for age_range, count in stats['age_distribution'].items():
            if stats['total_faces'] > 0:
                pct = count / stats['total_faces'] * 100
                bar = "█" * int(pct / 2)
                f.write(f"{age_range:>6}: {count:4d} ({pct:5.1f}%) {bar}\n")
            else:
                f.write(f"{age_range:>6}: 0\n")

    print(f"✅ Saved: {report_path}")


def _save_image_csv(reports_dir: Path, image_stats: List[Dict]):
    """Save per-image statistics as CSV"""
    csv_path = reports_dir / "per_image_statistics.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'filename', 'faces', 'male', 'female', 'avg_age',
            'age_0_18', 'age_19_35', 'age_36_60', 'age_60_plus'
        ])

        writer.writeheader()
        for stat in image_stats:
            writer.writerow({
                'filename': stat['filename'],
                'faces': stat['faces'],
                'male': stat['male'],
                'female': stat['female'],
                'avg_age': f"{stat['avg_age']:.1f}" if stat['avg_age'] > 0 else "N/A",
                'age_0_18': stat['age_dist']['0-18'],
                'age_19_35': stat['age_dist']['19-35'],
                'age_36_60': stat['age_dist']['36-60'],
                'age_60_plus': stat['age_dist']['60+']
            })

    print(f"✅ Saved: {csv_path}")


def _save_detailed_json(reports_dir: Path, all_results: List[Dict],
                        image_stats: List[Dict], failed: List[Dict]):
    """Save detailed results as JSON"""
    json_path = reports_dir / "detailed_results.json"

    data = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_images': len(image_stats),
            'failed_images': len(failed),
            'total_faces': len(all_results)
        },
        'per_image': image_stats,
        'all_faces': all_results,
        'failed': failed
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved: {json_path}")


def _print_summary(stats: Dict, total_images: int, failed: int, output_path: Path):
    """Print summary to console"""
    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print()
    print(f"Images Processed:  {total_images - failed}/{total_images}")
    print(f"Total Faces:       {stats['total_faces']}")
    print()
    print("Output:")
    print(f"  Annotated images: {output_path / 'annotated'}")
    print(f"  Reports:          {output_path / 'reports'}")
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Batch process images for ad signage analytics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Process all images in a directory
  python batch_process.py --input ./mall_photos/ --output ./results/

  # Custom models directory
  python batch_process.py --input ./images/ --output ./results/ --models /path/to/models

Output structure:
  results/
  ├── annotated/               # Annotated images with bboxes
  │   ├── image1.jpg
  │   └── image2.jpg
  └── reports/
      ├── overall_statistics.txt      # Summary report
      ├── per_image_statistics.csv    # CSV for Excel
      └── detailed_results.json       # Full JSON data
        """
    )

    parser.add_argument('--input', '-i', required=True,
                       help='Input directory containing images')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for results')
    parser.add_argument('--models', '-m', default='../models',
                       help='Directory containing dlib .dat model files (default: ../models)')

    args = parser.parse_args()

    # Validate input directory
    if not Path(args.input).exists():
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)

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

    # Run batch processing
    process_directory(args.input, args.output, str(models_dir))
