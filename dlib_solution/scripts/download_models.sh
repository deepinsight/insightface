#!/bin/bash
# Download all required dlib pre-trained models
# License: All models are CC0 v1.0 Universal (Public Domain)
# Commercial use: ✅ ALLOWED

set -e  # Exit on error

MODELS_DIR="$(dirname "$0")/../models"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

echo "================================================"
echo "Downloading dlib Pre-trained Models"
echo "License: CC0 v1.0 (Public Domain)"
echo "Commercial Use: ALLOWED"
echo "================================================"
echo ""

# Function to download and extract
download_model() {
    local url=$1
    local filename=$2
    local description=$3

    echo "Downloading $description..."
    echo "URL: $url"

    if [ -f "${filename%.bz2}" ]; then
        echo "✅ Already exists: ${filename%.bz2}"
    else
        wget -q --show-progress "$url" -O "$filename"
        echo "Extracting..."
        bunzip2 -f "$filename"
        echo "✅ Downloaded: ${filename%.bz2}"
    fi
    echo ""
}

# 1. Face Detector (CNN-based)
download_model \
    "http://dlib.net/files/mmod_human_face_detector.dat.bz2" \
    "mmod_human_face_detector.dat.bz2" \
    "Face Detector (CNN)"

# 2. Gender Classifier
download_model \
    "http://dlib.net/files/dnn_gender_classifier_v1.dat.bz2" \
    "dnn_gender_classifier_v1.dat.bz2" \
    "Gender Classifier (97.3% accuracy)"

# 3. Age Predictor
download_model \
    "http://dlib.net/files/dnn_age_predictor_v1.dat.bz2" \
    "dnn_age_predictor_v1.dat.bz2" \
    "Age Predictor (SOTA)"

# 4. Shape Predictor (5 landmarks - for face alignment)
download_model \
    "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2" \
    "shape_predictor_5_face_landmarks.dat.bz2" \
    "5-point Face Landmark Detector"

echo "================================================"
echo "✅ All models downloaded successfully!"
echo "================================================"
echo ""
echo "Models location: $MODELS_DIR"
echo ""
echo "Available models:"
ls -lh "$MODELS_DIR"/*.dat 2>/dev/null || echo "No .dat files found"
echo ""
echo "Total size:"
du -sh "$MODELS_DIR"
echo ""
echo "================================================"
echo "Next steps:"
echo "1. Install dlib: pip install dlib"
echo "2. Run demo: python examples/demo_age_gender.py"
echo "================================================"
