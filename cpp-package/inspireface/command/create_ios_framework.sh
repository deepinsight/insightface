#!/bin/bash

# Set the framework name
FRAMEWORK_NAME=InspireFace

# Specify the version of the framework
FRAMEWORK_VERSION=1.0.0

# Root build directory
BUILD_DIR=build/inspireface-ios/install/InspireFace

# Create the framework structure
FRAMEWORK_DIR=$BUILD_DIR/$FRAMEWORK_NAME.framework
mkdir -p $FRAMEWORK_DIR
mkdir -p $FRAMEWORK_DIR/Headers
mkdir -p $FRAMEWORK_DIR/Resources

# Copy the static library to the framework directory
cp $BUILD_DIR/lib/libInspireFace.a $FRAMEWORK_DIR/$FRAMEWORK_NAME

# Copy header files to the framework's Headers directory
cp $BUILD_DIR/include/*.h $FRAMEWORK_DIR/Headers/

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
