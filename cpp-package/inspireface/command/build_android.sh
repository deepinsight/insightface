#!/bin/bash
set -eo pipefail

reorganize_structure() {
    local base_path=$1
    local arch arch_dir main_dir
    local copied_common=false
    # Iterate only known ABIs: include/java from an earlier build are not ABIs.
    for arch in arm64-v8a armeabi-v7a x86_64; do
        arch_dir="$base_path/$arch"
        for main_dir in lib sample test; do
            mkdir -p "$base_path/$main_dir/$arch"
            if [[ -d "$arch_dir/InspireFace/$main_dir" ]]; then
                cp -R "$arch_dir/InspireFace/$main_dir/." "$base_path/$main_dir/$arch/"
            fi
        done
        # Remove the obsolete adapter when rebuilding into a former two-library SDK.
        rm -f "$base_path/lib/$arch/libInspireFaceJNI.so"
        if [[ "$copied_common" == false ]]; then
            mkdir -p "$base_path/include" "$base_path/java"
            cp -R "$arch_dir/InspireFace/include/." "$base_path/include/"
            cp "$arch_dir/version.txt" "$base_path/version.txt"
            # Keep the supplementary legacy Android classes alongside the full JNI JAR.
            cp -R "$arch_dir/InspireFace/java/." "$base_path/java/"
            if [[ -f "$arch_dir/Java/inspireface.jar" ]]; then
                cp "$arch_dir/Java/inspireface.jar" "$arch_dir/Java/api-manifest.json" "$base_path/java/"
                cp -R "$arch_dir/Java/sources" "$arch_dir/Java/examples" "$base_path/java/"
                cp "$SCRIPT_DIR/java/consumer-rules.pro" "$base_path/java/"
            fi
            copied_common=true
        fi
    done
    # Remove intermediate ABI installs only after all artifacts have been copied.
    for arch in arm64-v8a armeabi-v7a x86_64; do
        rm -rf "$base_path/$arch"
    done
    echo "Reorganization complete (C/C++ SDK, legacy Android sources, and portable JNI/JAR)."
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
    local arch=$1
    local NDK_API_LEVEL=$2
    local android_neon_args=()
    # Let the NDK toolchain select valid ARMv7 NEON flags for this ABI only.
    if [[ "${arch}" == "armeabi-v7a" ]]; then
        android_neon_args+=("-DANDROID_ARM_NEON=TRUE")
    fi
    mkdir -p ${BUILD_FOLDER_PATH}/${arch}
    pushd ${BUILD_FOLDER_PATH}/${arch}
    cmake ${SCRIPT_DIR} \
        -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_C_FLAGS="-g0 ${CMAKE_C_FLAGS}" \
        -DCMAKE_CXX_FLAGS="-g0 ${CMAKE_CXX_FLAGS}" \
        -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
        -DANDROID_TOOLCHAIN=clang \
        -DANDROID_ABI=${arch} \
        "${android_neon_args[@]}" \
        -DANDROID_NATIVE_API_LEVEL=${NDK_API_LEVEL} \
        -DANDROID_STL=c++_static \
        -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
        -DISF_BUILD_JAVA=ON \
        -DISF_BUILD_JAVA_TESTS=OFF \
        -DISF_BUILD_WITH_SAMPLE=OFF \
        -DISF_BUILD_WITH_TEST=OFF \
        -DISF_ENABLE_BENCHMARK=OFF \
        -DISF_ENABLE_USE_LFW_DATA=OFF \
        -DISF_ENABLE_TEST_EVALUATION=OFF \
        -DISF_BUILD_SHARED_LIBS=ON \
        -Wno-dev
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
build x86_64 21

reorganize_structure "${BUILD_FOLDER_PATH}"
