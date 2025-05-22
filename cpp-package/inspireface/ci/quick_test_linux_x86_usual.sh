#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

TARGET_DIR="test_res"
BUILD_DIRNAME="ci_ubuntu18"
TEST_DIR="./build/${BUILD_DIRNAME}/test"
TEST_EXECUTABLE="./test/Test"

# Make dir
mkdir -p ${TARGET_DIR}/save/video_frames

# Download models
bash command/download_models_general.sh Pikachu

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
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DISF_BUILD_WITH_SAMPLE=OFF \
  -DISF_BUILD_WITH_TEST=ON \
  -DISF_ENABLE_BENCHMARK=ON \
  -DISF_ENABLE_USE_LFW_DATA=OFF \
  -DISF_ENABLE_TEST_EVALUATION=OFF \
  -DISF_BUILD_SHARED_LIBS=OFF ../../

# Compile the project using 4 parallel jobs
make -j4

# Check if the symbolic link or directory already exists
if [ ! -e "$(basename ${FULL_TEST_DIR})" ]; then
    # Create a symbolic link to the extracted test data directory
    ln -s ${FULL_TEST_DIR} .
    echo "Symbolic link to '${TARGET_DIR}' created."
else
    echo "Symbolic link or directory '$(basename ${FULL_TEST_DIR})' already exists. Skipping creation."
fi

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
