#!/bin/bash

# Reusable function to handle 'install' directory operations
move_install_files() {
    local root_dir="$1"
    local install_dir="$root_dir/install"

    # Step 1: Check if the 'install' directory exists
    if [ ! -d "$install_dir" ]; then
        echo "Error: 'install' directory does not exist in $root_dir"
        exit 1
    fi

    # Step 2: Delete all other files/folders except 'install'
    find "$root_dir" -mindepth 1 -maxdepth 1 -not -name "install" -exec rm -rf {} +

    # Step 3: Move all files from 'install' to the root directory
    mv "$install_dir"/* "$root_dir" 2>/dev/null

    # Step 4: Remove the empty 'install' directory
    rmdir "$install_dir"

    echo "Files from 'install' moved to $root_dir, and 'install' directory deleted."
}

if [ -n "$VERSION" ]; then
    TAG="-$VERSION"
else
    TAG=""
fi

SCRIPT_DIR=$(pwd)  # Project dir
BUILD_FOLDER_PATH="build/inspireface-linux-aarch64${TAG}"

mkdir -p ${BUILD_FOLDER_PATH}
# shellcheck disable=SC2164
cd ${BUILD_FOLDER_PATH}
# export cross_compile_toolchain=/home/jingyuyan/software/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu
cmake -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_VERSION=1 \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=$ARM_CROSS_COMPILE_TOOLCHAIN/bin/aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=$ARM_CROSS_COMPILE_TOOLCHAIN/bin/aarch64-linux-gnu-g++ \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -flax-vector-conversions" \
  -DTARGET_PLATFORM=armlinux \
  -DISF_BUILD_LINUX_AARCH64=ON \
  -DISF_BUILD_LINUX_ARM7=OFF \
  -DISF_BUILD_WITH_SAMPLE=OFF \
  -DISF_BUILD_WITH_TEST=OFF \
  -DISF_ENABLE_BENCHMARK=OFF \
  -DISF_ENABLE_USE_LFW_DATA=OFF \
  -DISF_ENABLE_TEST_EVALUATION=OFF \
  -DISF_BUILD_SHARED_LIBS=ON ${SCRIPT_DIR}

make -j4
make install

move_install_files "$(pwd)"
