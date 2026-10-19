#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
BUILD_DIRNAME="ubuntu18_shared"
TEST_RES_DIR="$ROOT_DIR/test_res"
MODEL_PATH="$TEST_RES_DIR/pack/Pikachu"

# Image fixtures are checked in, but model packs are ignored by Git.
# Prepare the model before compiling so a failed download stops the job early.
bash command/download_models_general.sh Pikachu
test -s "$MODEL_PATH"

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
cd "$ROOT_DIR"

# Important: You must copy the compiled dynamic library to this path!
mkdir -p python/inspireface/modules/core/libs/linux/x64/
cp build/${BUILD_DIRNAME}/lib/libInspireFace.so python/inspireface/modules/core/libs/linux/x64/

# Install the package through its declared build metadata plus test-only OpenCV.
python -m pip install -e python opencv-python

cd python/

# Run the complete Python API contract, result, resource, and timing gates.
python -m sample_testcase.run \
  --test-dir "$TEST_RES_DIR" \
  --pack-path "$MODEL_PATH" \
  --verbosity 1
