#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

ROOT_DIR="$(pwd)"
BUILD_DIRNAME="ubuntu18_shared"

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
  -DISF_BUILD_WITH_TEST=OFF \
  -DISF_ENABLE_BENCHMARK=OFF \
  -DISF_ENABLE_USE_LFW_DATA=OFF \
  -DISF_ENABLE_TEST_EVALUATION=OFF \
  -DISF_BUILD_SHARED_LIBS=ON ../../

# Compile the project using 4 parallel jobs
make -j4

# Come back to project root dir
cd ${ROOT_DIR}

# Important: You must copy the compiled dynamic library to this path!
mkdir -p python/inspireface/modules/core/libs/linux/x64/
cp build/${BUILD_DIRNAME}/lib/libInspireFace.so python/inspireface/modules/core/libs/linux/x64/

# Install dependency
pip install opencv-python
pip install click
pip install loguru
pip install filelock
pip install modelscope

cd python/

# Run sample
python sample_face_detection.py ../test_res/data/bulk/woman.png

