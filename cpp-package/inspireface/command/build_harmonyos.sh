#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_NATIVE_DIR="${PROJECT_DIR}/build/harmony-tools/openharmony-6.1/native-sdk/native"
NATIVE_DIR="${OHOS_NATIVE_HOME:-${DEFAULT_NATIVE_DIR}}"
BUILD_DIR="${OHOS_BUILD_DIR:-${PROJECT_DIR}/build/inspireface-harmonyos-arm64-v8a}"
TCPKG_COMPAT_DIR="${PROJECT_DIR}/cmake/ohos/node_modules"

if [[ ! -f "${NATIVE_DIR}/build/cmake/ohos.toolchain.cmake" ]]; then
    echo "OpenHarmony Native SDK was not found at: ${NATIVE_DIR}" >&2
    echo "Set OHOS_NATIVE_HOME to the Native SDK directory containing build/cmake/ohos.toolchain.cmake." >&2
    exit 1
fi

if [[ ! -f "${TCPKG_COMPAT_DIR}/@ali/tcpkg/tcpkg.cmake" ]]; then
    echo "MNN OHOS compatibility module was not found at: ${TCPKG_COMPAT_DIR}" >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}"

NODE_PATH="${TCPKG_COMPAT_DIR}" cmake \
    -S "${PROJECT_DIR}" \
    -B "${BUILD_DIR}" \
    -G "Unix Makefiles" \
    -DCMAKE_TOOLCHAIN_FILE="${NATIVE_DIR}/build/cmake/ohos.toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DOHOS_ARCH=arm64-v8a \
    -DOHOS_STL=c++_static \
    -DMNN_BUILD_FOR_ANDROID_COMMAND=ON \
    -DMNN_BUILD_SHARED_LIBS=OFF \
    -DMNN_BUILD_TOOLS=OFF \
    -DMNN_BUILD_TEST=OFF \
    -DMNN_BUILD_BENCHMARK=OFF \
    -DMNN_USE_LOGCAT=OFF \
    -DMNN_USE_SSE=OFF \
    -DMNN_SUPPORT_BF16=OFF \
    -DISF_BUILD_WITH_SAMPLE=OFF \
    -DISF_BUILD_WITH_TEST=OFF \
    -DBUILD_TESTING=OFF \
    -DISF_ENABLE_RKNN=OFF \
    -DISF_ENABLE_RGA=OFF \
    -DISF_ENABLE_TENSORRT=OFF \
    -DISF_ENABLE_APPLE_EXTENSION=OFF \
    -DISF_GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA=OFF

cmake --build "${BUILD_DIR}" --parallel "${OHOS_BUILD_JOBS:-4}"
cmake --install "${BUILD_DIR}"

echo "OpenHarmony arm64-v8a SDK: ${BUILD_DIR}/install"
