#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

ROOT_DIR="$(pwd)"
TARGET_DIR="test_res"
DOWNLOAD_URL="https://github.com/tunmx/inspireface-store/raw/main/resource/test_res-lite.zip"
ZIP_FILE="test_res-lite.zip"
BUILD_DIRNAME="ubuntu18_shared"

# Check if the target directory already exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Directory '$TARGET_DIR' does not exist. Downloading..."

    # Download the dataset zip file
    wget -q "$DOWNLOAD_URL" -O "$ZIP_FILE"

    echo "Extracting '$ZIP_FILE' to '$TARGET_DIR'..."
    # Unzip the downloaded file
    unzip "$ZIP_FILE"

    # Remove the downloaded zip file and unnecessary folders
    rm "$ZIP_FILE"
    rm -rf "__MACOSX"

    echo "Download and extraction complete."
else
    echo "Directory '$TARGET_DIR' already exists. Skipping download."
fi

# Get the absolute path of the target directory
FULL_TEST_DIR="$(realpath ${TARGET_DIR})"

# Create the build directory if it doesn't exist
mkdir -p build/${BUILD_DIRNAME}/

# Change directory to the build directory
# Disable the shellcheck warning for potential directory changes
# shellcheck disable=SC2164
cd build/${BUILD_DIRNAME}/

# Configure the CMake build system
cmake -DCMAKE_BUILD_TYPE=Release \
  -DISF_BUILD_WITH_SAMPLE=OFF \
  -DISF_BUILD_WITH_TEST=OFF \
  -DISF_ENABLE_BENCHMARK=OFF \
  -DISF_ENABLE_USE_LFW_DATA=OFF \
  -DISF_ENABLE_TEST_EVALUATION=OFF \
  -DOpenCV_DIR=3rdparty/inspireface-precompile/opencv/4.5.1/opencv-ubuntu18-x86/lib/cmake/opencv4 \
  -DISF_BUILD_SHARED_LIBS=ON ../../

# Compile the project using 4 parallel jobs
make -j4

# Come back to project root dir
cd ${ROOT_DIR}

# Important: You must copy the compiled dynamic library to this path!
cp build/${BUILD_DIRNAME}/lib/libInspireFace.so python/inspireface/modules/core/

# Install dependency
pip install opencv-python
pip install click
pip install loguru

cd python/

# Run sample
python sample_face_detection.py ../test_res/pack/Pikachu ../test_res/data/bulk/woman.png

