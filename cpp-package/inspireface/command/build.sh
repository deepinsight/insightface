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

# Detect the operating system
OS_NAME=$(uname)
BUILD_DIR="build"
SCRIPT_DIR=$(pwd)  # Project dir



# Determine the appropriate build directory based on the OS
case "$OS_NAME" in
    Darwin)
        # macOS system
        BUILD_DIR="${BUILD_DIR}/inspireface-macos${TAG}"
        ;;
    Linux)
        # Linux system, further identify the distribution if necessary
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            case "$ID" in
                ubuntu)
                    BUILD_DIR="${BUILD_DIR}/inspireface-linux-ubuntu${TAG}"
                    ;;
                centos)
                    BUILD_DIR="${BUILD_DIR}/inspireface-linux-centos${TAG}"
                    ;;
                *)
                    # If an unknown Linux distribution, default to generic 'linux'
                    BUILD_DIR="${BUILD_DIR}/inspireface-linux${TAG}"
                    ;;
            esac
        else
            # If unable to detect Linux distribution, default to 'linux'
            BUILD_DIR="${BUILD_DIR}/inspireface-linux${TAG}"
        fi
        ;;
    *)
        # If OS is not recognized, default to 'generic'
        BUILD_DIR="${BUILD_DIR}/inspireface-generic${TAG}"
        ;;
esac

# Create the build directory and navigate into it
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit 1

# Run CMake configuration (adjust the options as needed)
cmake -DCMAKE_BUILD_TYPE=Release \
  -DISF_BUILD_WITH_SAMPLE=ON \
  -DISF_BUILD_WITH_TEST=OFF \
  -DISF_ENABLE_BENCHMARK=OFF \
  -DISF_ENABLE_USE_LFW_DATA=OFF \
  -DISF_ENABLE_TEST_EVALUATION=OFF \
  -DISF_BUILD_SHARED_LIBS=ON "$SCRIPT_DIR"

# Compile and install
make -j4
make install

# Move 'install' files to the build root directory using an absolute path
move_install_files "$(pwd)"
