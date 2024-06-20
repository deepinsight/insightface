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

# Define download URLs
MNN_IOS_URL="https://github.com/alibaba/MNN/releases/download/2.8.1/mnn_2.8.1_ios_armv82_cpu_metal_coreml.zip"
OPENCV_IOS_URL="https://github.com/opencv/opencv/releases/download/4.5.1/opencv-4.5.1-ios-framework.zip"

# Set the cache directory
MACOS_CACHE="$PWD/.macos_cache/"

# Create the directory if it does not exist
mkdir -p "${MACOS_CACHE}"

# Function to download and unzip a file if the required framework does not exist
download_and_unzip() {
    local url=$1
    local dir=$2
    local framework_name=$3  # Name of the framework directory to check

    # Check if the framework already exists
    if [ ! -d "${dir}${framework_name}" ]; then
        local file_name=$(basename "$url")
        local full_path="${dir}${file_name}"

        # Check if the zip file already exists
        if [ ! -f "$full_path" ]; then
            echo "Downloading ${file_name}..."
            # Download the file
            curl -sL "$url" -o "$full_path"
        else
            echo "${file_name} already downloaded. Proceeding to unzip."
        fi

        # Unzip the file to a temporary directory
        echo "Unzipping ${file_name}..."
        unzip -q "$full_path" -d "${dir}"
        rm "$full_path"

        # Move the framework if it's in a subdirectory specific to the iOS build
        if [ "${framework_name}" == "MNN.framework" ]; then
            mv "${dir}ios_build/Release-iphoneos/${framework_name}" "${dir}"
            rm -rf "${dir}ios_build"  # Clean up the subdirectory
        fi

        echo "${framework_name} has been set up."
    else
        echo "${framework_name} already exists in ${dir}. Skipping download and unzip."
    fi
}

# Download and unzip MNN iOS package
download_and_unzip "$MNN_IOS_URL" "$MACOS_CACHE" "MNN.framework"

# Download and unzip OpenCV iOS package
download_and_unzip "$OPENCV_IOS_URL" "$MACOS_CACHE" "opencv2.framework"

if [ -n "$VERSION" ]; then
    TAG="-$VERSION"
else
    TAG=""
fi


TOOLCHAIN="$PWD/toolchain/ios.toolchain.cmake"

BUILD_DIR="build/inspireface-ios$TAG"

mkdir -p "$BUILD_DIR"

cd "$BUILD_DIR"

cmake \
    -DIOS_3RDPARTY="${MACOS_CACHE}" \
    -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN} \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DENABLE_BITCODE=0 \
    -DIOS_DEPLOYMENT_TARGET=11.0 \
    -DISF_BUILD_WITH_SAMPLE=OFF \
    -DISF_BUILD_WITH_TEST=OFF \
    -DISF_BUILD_SHARED_LIBS=OFF \
    ../..

make -j8

make install

move_install_files "$(pwd)"

# Set the framework name
FRAMEWORK_NAME=InspireFace

# Specify the version of the framework
FRAMEWORK_VERSION=1.0.0

# Root build directory
BUILD_DIR="$(pwd)"

BUILD_LIB_DIR="$BUILD_DIR/InspireFace"

# Create the framework structure
FRAMEWORK_DIR=$BUILD_DIR/$FRAMEWORK_NAME.framework
mkdir -p $FRAMEWORK_DIR
mkdir -p $FRAMEWORK_DIR/Headers
mkdir -p $FRAMEWORK_DIR/Resources

# Copy the static library to the framework directory
cp $BUILD_LIB_DIR/lib/libInspireFace.a $FRAMEWORK_DIR/$FRAMEWORK_NAME

# Copy header files to the framework's Headers directory
cp $BUILD_LIB_DIR/include/*.h $FRAMEWORK_DIR/Headers/

# Create Info.plist
cat <<EOF >$FRAMEWORK_DIR/Resources/Info.plist
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$FRAMEWORK_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>com.example.$FRAMEWORK_NAME</string>
    <key>CFBundleName</key>
    <string>$FRAMEWORK_NAME</string>
    <key>CFBundleVersion</key>
    <string>$FRAMEWORK_VERSION</string>
    <key>CFBundleShortVersionString</key>
    <string>$FRAMEWORK_VERSION</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
</dict>
</plist>
EOF

echo "Framework $FRAMEWORK_NAME.framework has been created at $FRAMEWORK_DIR"
