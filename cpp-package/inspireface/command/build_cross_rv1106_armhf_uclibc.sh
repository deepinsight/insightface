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

# Create .rknpu2_cache directory if it doesn't exist
CACHE_DIR="$(pwd)/.rknpu2_cache"
mkdir -p "$CACHE_DIR"

# Check if MNN-2.3.0 directory already exists
if [ ! -d "$CACHE_DIR/MNN-2.3.0" ]; then
    echo "Downloading MNN 2.3.0..."
    # Download MNN 2.3.0
    if ! wget -P "$CACHE_DIR" https://github.com/alibaba/MNN/archive/refs/tags/2.3.0.zip; then
        echo "Error: Failed to download MNN 2.3.0"
        exit 1
    fi
    
    # Extract the zip file
    cd "$CACHE_DIR"
    if ! unzip 2.3.0.zip; then
        echo "Error: Failed to extract MNN 2.3.0"
        exit 1
    fi
    
    # Remove the zip file
    rm 2.3.0.zip
    
    echo "MNN 2.3.0 downloaded and extracted"
else
    echo "MNN-2.3.0 already exists in cache"
fi

# Set absolute path to MNN source
export ISF_MNN_CUSTOM_SOURCE="$CACHE_DIR/MNN-2.3.0"

echo "ISF_MNN_CUSTOM_SOURCE: ${ISF_MNN_CUSTOM_SOURCE}"
cd ${SCRIPT_DIR}

# export ARM_CROSS_COMPILE_TOOLCHAIN=/root/arm-rockchip830-linux-uclibcgnueabihf/

BUILD_FOLDER_PATH="build/inspireface-linux-armv7-rv1106-armhf-uclibc${TAG}"

mkdir -p ${BUILD_FOLDER_PATH}
# shellcheck disable=SC2164
cd ${BUILD_FOLDER_PATH}

# export cross_compile_toolchain=/home/jingyuyan/software/arm-rockchip830-linux-uclibcgnueabihf

cmake -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_SYSTEM_VERSION=1 \
  -DCMAKE_SYSTEM_PROCESSOR=armv7 \
  -DCMAKE_C_COMPILER=$ARM_CROSS_COMPILE_TOOLCHAIN/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  -DCMAKE_CXX_COMPILER=$ARM_CROSS_COMPILE_TOOLCHAIN/bin/arm-rockchip830-linux-uclibcgnueabihf-g++ \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -flax-vector-conversions" \
  -DTARGET_PLATFORM=armlinux \
  -DISF_BUILD_LINUX_ARM7=ON \
  -DISF_MNN_CUSTOM_SOURCE=${ISF_MNN_CUSTOM_SOURCE} \
  -DMNN_SEP_BUILD=off \
  -DISF_ENABLE_RKNN=ON \
  -DISF_RK_DEVICE_TYPE=RV1106 \
  -DISF_RKNPU_MAJOR=rknpu2 \
  -DISF_RK_COMPILER_TYPE=armhf-uclibc \
  -DISF_ENABLE_RGA=ON \
  -DISF_ENABLE_COST_TIME=OFF \
  -DISF_BUILD_WITH_SAMPLE=OFF \
  -DISF_BUILD_WITH_TEST=OFF \
  -DISF_ENABLE_BENCHMARK=OFF \
  -DISF_ENABLE_USE_LFW_DATA=OFF \
  -DISF_ENABLE_TEST_EVALUATION=OFF \
  -Wno-dev \
  -DISF_BUILD_SHARED_LIBS=ON ${SCRIPT_DIR}

make -j4
make install

move_install_files "$(pwd)"