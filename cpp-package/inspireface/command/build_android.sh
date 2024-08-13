#!/bin/bash

reorganize_structure() {
    local base_path=$1

    # Define the new main directories
    local main_dirs=("lib" "sample" "test")

    # Check if the base path exists
    if [[ ! -d "$base_path" ]]; then
        echo "Error: The path '$base_path' does not exist."
        return 1
    fi

    # Create new main directories at the base path
    for dir in "${main_dirs[@]}"; do
        mkdir -p "$base_path/$dir"
    done

    # Find all architecture directories (e.g., arm64-v8a, armeabi-v7a)
    local arch_dirs=($(find "$base_path" -maxdepth 1 -type d -name "arm*"))

    for arch_dir in "${arch_dirs[@]}"; do
        # Get the architecture name (e.g., arm64-v8a)
        local arch=$(basename "$arch_dir")

        # Operate on each main directory
        for main_dir in "${main_dirs[@]}"; do
            # Create a specific directory for each architecture under the main directory
            mkdir -p "$base_path/$main_dir/$arch"

            # Selectively copy content based on the directory type
            case "$main_dir" in
                lib)
                    # Copy the lib directory
                    if [ -d "$arch_dir/InspireFace/lib" ]; then
                        cp -r "$arch_dir/InspireFace/lib/"* "$base_path/$main_dir/$arch/"
                    fi
                    ;;
                sample)
                    # Copy the sample directory
                    if [ -d "$arch_dir/InspireFace/sample" ]; then
                        cp -r "$arch_dir/InspireFace/sample/"* "$base_path/$main_dir/$arch/"
                    fi
                    ;;
                test)
                    # Copy the test directory
                    if [ -d "$arch_dir/InspireFace/test" ]; then
                        cp -r "$arch_dir/InspireFace/test/"* "$base_path/$main_dir/$arch/"
                    fi
                    ;;
            esac
        done

        # Copy version.txt file to the base path, ignoring duplicates
        if [ -f "$arch_dir/version.txt" ]; then
            cp -f "$arch_dir/version.txt" "$base_path/version.txt"
        fi
    done

    # Delete the original architecture directories
    for arch_dir in "${arch_dirs[@]}"; do
        rm -rf "$arch_dir"
    done

    echo "Reorganization complete."
}


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

build() {
    arch=$1
    NDK_API_LEVEL=$2
    mkdir -p ${BUILD_FOLDER_PATH}/${arch}
    pushd ${BUILD_FOLDER_PATH}/${arch}
    cmake ${SCRIPT_DIR} \
        -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
        -DANDROID_TOOLCHAIN=clang \
        -DANDROID_ABI=${arch} \
        -DANDROID_NATIVE_API_LEVEL=${NDK_API_LEVEL} \
        -DANDROID_STL=c++_static \
        -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
        -DISF_BUILD_WITH_SAMPLE=OFF \
        -DISF_BUILD_WITH_TEST=OFF \
        -DISF_ENABLE_BENCHMARK=OFF \
        -DISF_ENABLE_USE_LFW_DATA=OFF \
        -DISF_ENABLE_TEST_EVALUATION=OFF \
        -DISF_BUILD_SHARED_LIBS=ON \
        -DOpenCV_DIR=${OPENCV_DIR}
    make -j4
    make install
    popd
    move_install_files "${BUILD_FOLDER_PATH}/${arch}"
}

if [ -n "$VERSION" ]; then
    TAG="-$VERSION"
else
    TAG=""
fi

SCRIPT_DIR=$(pwd)  # Project dir
BUILD_FOLDER_PATH="build/inspireface-android${TAG}"

build arm64-v8a 21
build armeabi-v7a 21

reorganize_structure "${BUILD_FOLDER_PATH}"

