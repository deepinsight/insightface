#!/bin/bash

set -ex

RELEASE_HOME=$(cd $(dirname $0)/../..; pwd)
BUILD_DIR=${RELEASE_HOME}/build/release_android

[[ -d ${BUILD_DIR} ]] && rm -r ${BUILD_DIR}

build() {
    arch=$1
    NDK_API_LEVEL=$2
    mkdir -p ${BUILD_DIR}/${arch}
    pushd ${BUILD_DIR}/${arch}
    cmake ${RELEASE_HOME} \
        -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
        -DANDROID_TOOLCHAIN=clang \
        -DANDROID_ABI=${arch} \
        -DANDROID_NATIVE_API_LEVEL=${NDK_API_LEVEL} \
        -DANDROID_STL=c++_static \
        -DBUILD_CUDA=OFF \
        -DBUILD_3RDPARTY_PATH=${BUILD_3RDPARTY_PATH} \
        -DOpenCV_DIR=${OPENCV_DIR} \
        -DMNN_LIBS=${BUILD_3RDPARTY_PATH}/MNN-2.2.0/android-static/${arch} \
        -DMNN_INCLUDE_DIRS=${BUILD_3RDPARTY_PATH}/MNN-2.2.0/android-static/include \
        -DYAML_CPP_LIBS=${BUILD_3RDPARTY_PATH}/yaml-cpp/android-static/${arch} \
        -DYAML_CPP_INCLUDE_DIRS=${BUILD_3RDPARTY_PATH}/yaml-cpp/android-static/include
#        -DNCNN_DIR=${RELEASE_HOME}/3rdparty/ncnn/android/${arch} \
    make -j$(nproc) 
#    ls ${BUILD_DIR}/${arch}| grep -v so| xargs rm -r
    #make -j$(nproc) track_tool
    popd
}


build arm64-v8a 21
build armeabi-v7a 21

date -R > ${BUILD_DIR}/release_note.txt
cd ${BUILD_DIR}
find . -type f |xargs md5sum >>release_note.txt
cd -
#cp -r ${RELEASE_HOME}/samples/c_api_demo.cpp ${RELEASE_HOME}/release_android
#cp -r ${RELEASE_HOME}/samples/utils ${RELEASE_HOME}/release_android
