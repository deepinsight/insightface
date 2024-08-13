#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

TARGET_DIR="test_res"
DOWNLOAD_URL="https://github.com/tunmx/inspireface-store/raw/main/resource/test_res-lite.zip"
ZIP_FILE="test_res-lite.zip"
BUILD_DIRNAME="quick_test_build"
TEST_DIR="./build/${BUILD_DIRNAME}/test"
TEST_EXECUTABLE="./test/Test"

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
  -DISF_BUILD_WITH_TEST=ON \
  -DISF_ENABLE_BENCHMARK=ON \
  -DISF_ENABLE_USE_LFW_DATA=OFF \
  -DISF_ENABLE_TEST_EVALUATION=OFF \
  -DISF_BUILD_SHARED_LIBS=OFF ../../

# Compile the project using 4 parallel jobs
make -j4

# Create a symbolic link to the extracted test data directory
ln -s ${FULL_TEST_DIR} .

# Check if the test executable file exists
if [ ! -f "$TEST_EXECUTABLE" ]; then
    # If not, print an error message and exit with a non-zero status code
    echo "Error: Test executable '$TEST_EXECUTABLE' not found. Please ensure it is built correctly."
    exit 1
else
    # If it exists, print a message and run the test executable
    echo "Test executable found. Running tests..."
    "$TEST_EXECUTABLE"
fi

# Executing python scripts