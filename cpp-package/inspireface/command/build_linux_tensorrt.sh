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

get_cuda_ubuntu_tag() {
    # If CUDA_TAG is set, use it
    if [ -n "${CUDA_TAG}" ]; then
        echo "${CUDA_TAG}"
        return 0
    fi
    
    # Get CUDA version
    CUDA_VERSION="_none"
    if command -v nvcc &> /dev/null; then
        # Try to get version from nvcc
        CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -d',' -f1 | tr -d '.')
        if [ -z "${CUDA_VERSION}" ]; then
            CUDA_VERSION="_none"
        else
            CUDA_VERSION="${CUDA_VERSION}"
        fi
    elif [ -f "/usr/local/cuda/version.txt" ]; then
        # Get version from CUDA installation directory
        CUDA_VERSION=$(cat /usr/local/cuda/version.txt 2>/dev/null | grep "CUDA Version" | awk '{print $3}' | tr -d '.')
        if [ -z "${CUDA_VERSION}" ]; then
            CUDA_VERSION="_none"
        fi
    elif [ -d "/usr/local/cuda" ] && ls -l /usr/local/cuda 2>/dev/null | grep -q "cuda-"; then
        # Get version from symbolic link
        CUDA_LINK=$(ls -l /usr/local/cuda 2>/dev/null | grep -o "cuda-[0-9.]*" | head -n 1)
        CUDA_VERSION=$(echo "${CUDA_LINK}" | cut -d'-' -f2 | tr -d '.')
        if [ -z "${CUDA_VERSION}" ]; then
            CUDA_VERSION="_none"
        fi
    fi
    
    # Get Ubuntu version
    UBUNTU_VERSION="_none"
    if [ -f "/etc/os-release" ]; then
        # Check if it is Ubuntu
        if grep -q "Ubuntu" /etc/os-release 2>/dev/null; then
            UBUNTU_VERSION=$(grep "VERSION_ID" /etc/os-release 2>/dev/null | cut -d'"' -f2)
            if [ -z "${UBUNTU_VERSION}" ]; then
                UBUNTU_VERSION="_none"
            fi
        fi
    elif [ -f "/etc/lsb-release" ]; then
        # Get version from lsb-release
        if grep -q "Ubuntu" /etc/lsb-release 2>/dev/null; then
            UBUNTU_VERSION=$(grep "DISTRIB_RELEASE" /etc/lsb-release 2>/dev/null | cut -d'=' -f2)
            if [ -z "${UBUNTU_VERSION}" ]; then
                UBUNTU_VERSION="_none"
            fi
        fi
    fi
    
    # Generate and return tag
    echo "cuda${CUDA_VERSION}_ubuntu${UBUNTU_VERSION}"
}

CUDA_TAG=$(get_cuda_ubuntu_tag)
echo "Cuda Tag: ${CUDA_TAG}"

if [ -n "$VERSION" ]; then
    TAG="-$VERSION"
else
    TAG=""
fi

SCRIPT_DIR=$(pwd)  
BUILD_FOLDER_NAME="inspireface-linux-tensorrt-${CUDA_TAG}${TAG}"

mkdir -p build/${BUILD_FOLDER_NAME}
cd build/${BUILD_FOLDER_NAME}

echo "TENSORRT_ROOT: ${TENSORRT_ROOT}"

cmake  \
  -DCMAKE_BUILD_TYPE=Release \
  -DISF_BUILD_WITH_SAMPLE=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DISF_BUILD_WITH_TEST=ON \
  -DISF_ENABLE_BENCHMARK=ON \
  -DISF_ENABLE_USE_LFW_DATA=OFF \
  -DISF_ENABLE_TEST_EVALUATION=OFF \
  -DTENSORRT_ROOT=${TENSORRT_ROOT} \
  -DISF_ENABLE_TENSORRT=ON \
  -Wno-dev \
  ../..

make -j4

make install

if [ $? -eq 0 ] && [ -d "$(pwd)/install" ]; then
  move_install_files "$(pwd)"
else
  echo "Build failed or the installation directory does not exist"
  exit 1
fi