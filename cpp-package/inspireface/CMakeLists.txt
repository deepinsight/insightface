#cmake_minimum_required(VERSION 3.19)
cmake_minimum_required(VERSION 3.10)
project(InspireFace)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++14")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set the THIRD_PARTY_DIR variable to allow it to be set externally from the command line, or use the default path if it is not set
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty" CACHE PATH "Path to the third-party libraries directory")

# Check that the SANITIZE compile option is enabled
set(SANITIZE_ADDRESS OFF CACHE BOOL "Enable AddressSanitizer")
set(SANITIZE_LEAK OFF CACHE BOOL "Enable LeakSanitizer")

if (SANITIZE_ADDRESS AND SANITIZE_LEAK)
    message(FATAL_ERROR "Cannot enable both AddressSanitizer and LeakSanitizer.")
endif()

# Enable AddressSanitizer
if(SANITIZE_ADDRESS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()

# Enable LeakSanitizer
if(SANITIZE_LEAK)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=leak")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=leak")
endif()

set(TARGET_PLATFORM "drawin-x86" CACHE STRING "Target platform")

# OpenCV dependency configuration
option(USE_MOBILE_OPENCV_IN_LOCAL "Use mobile-opencv in local environment." OFF)
set(MOBILE_OPENCV_VERSION "4.9.0" CACHE STRING "The mobile-opencv version number")
if (USE_MOBILE_OPENCV_IN_LOCAL)
    message("Use mobile-opencv ${MOBILE_OPENCV_VERSION}, Target platform ${TARGET_PLATFORM}")
endif ()

# RKNN dependency configuration
option(ENABLE_RKNN "Use RKNPU." OFF)
set(RK_DEVICE_TYPE "RV1109RV1126" CACHE STRING "Type of the device")
if (ENABLE_RKNN)
    add_definitions("-DENABLE_RKNN")
    # Device list
    set(RKNPU1_DEVICES "RV1109RV1126")
    set(RKNPU2_DEVICES "RK356X" "RK3588" "RV1106")
    set(RKNPU_MAJOR "")
    list(FIND RKNPU1_DEVICES "${RK_DEVICE_TYPE}" INDEX_RKNPU1)
    list(FIND RKNPU2_DEVICES "${RK_DEVICE_TYPE}" INDEX_RKNPU2)
    if(INDEX_RKNPU1 GREATER_EQUAL 0)
        # match rknn1
        set(RKNPU_MAJOR "rknpu1")
    elseif(INDEX_RKNPU2 GREATER_EQUAL 0)
        # match rknn2
        set(RKNPU_MAJOR "rknpu2")
    endif()
    # Result
    message(STATUS "Use ${RKNPU_MAJOR}")
endif ()


option(BUILD_LINUX_ARM7 "Platform Armv7." OFF)
option(BUILD_LINUX_AARCH64 "Platform Armv8." OFF)
option(GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA "The global inference backend uses MNN CUDA." OFF)


if (BUILD_LINUX_ARM7)
    set(CPU_ARCH "armhf")
endif()
if (BUILD_LINUX_AARCH64)
    set(CPU_ARCH "aarch64")
endif()

option(BUILD_WITH_TEST "Open Build Unit-Test." ON)
option(BUILD_WITH_SAMPLE "Open Build Sample Exec." ON)

option(LINUX_FETCH_MNN "Fetch and build MNN from git" OFF)


if (USE_MOBILE_OPENCV_IN_LOCAL)
    add_definitions(-DUSE_MOBILE_OPENCV_IN_LOCAL)
endif ()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/)

# OpenCV Configuration
string(REPLACE "." ";" VERSION_LIST ${MOBILE_OPENCV_VERSION})
list(GET VERSION_LIST 0 VERSION_MAJOR)
if (APPLE)
    set(PLAT darwin)
    if (USE_MOBILE_OPENCV_IN_LOCAL)
        set(OPENCV_FRAMEWORK "${THIRD_PARTY_DIR}/opencv-mobile/${MOBILE_OPENCV_VERSION}/${TARGET_PLATFORM}/opencv2.framework")
        list(APPEND CMAKE_FRAMEWORK_PATH "${OPENCV_FRAMEWORK}")
        include_directories("${OPENCV_FRAMEWORK}/Headers")
        message(OPENCV_FRAMEWORK=${OPENCV_FRAMEWORK})
        find_library(ACCELERATE_FRAMEWORK Accelerate)
    else ()
        find_package(OpenCV REQUIRED)
    endif ()
else()
    if (BUILD_LINUX_ARM7 OR BUILD_LINUX_AARCH64)
        add_definitions("-DDISABLE_GUI")
        #        set(OpenCV_DIR ${THIRD_PARTY_DIR}/opencv/opencv-linux-armhf/share/OpenCV)
        #        set(OpenCV_STATIC_INCLUDE_DIR ${PATH_3RDPARTY}/opencv/opencv-linux-armhf/include/)
        if (RK_DEVICE_TYPE STREQUAL "RV1109RV1126")
            # In special cases, specialize for that version
            message("The OpenCV that builds the RV1109RV1126 version depends on is specialized!")
            set(OpenCV_DIR ${THIRD_PARTY_DIR}/opencv/3.4.5/opencv-linux-armhf/share/OpenCV)
            set(OpenCV_STATIC_INCLUDE_DIR ${PATH_3RDPARTY}/opencv/3.4.5/opencv-linux-armhf/include/)
            set(PLAT linux-arm7)
        else()
            if (VERSION_MAJOR STREQUAL "3")
                set(CV_CMAKE_FOLDER share/OpenCV)
            elseif(VERSION_MAJOR STREQUAL "4")
                set(CV_CMAKE_FOLDER lib/cmake/opencv4)
            endif ()
            if(BUILD_LINUX_ARM7)
                set(PLAT linux-arm7)
                set(OpenCV_DIR "${THIRD_PARTY_DIR}/opencv-mobile/${MOBILE_OPENCV_VERSION}/${TARGET_PLATFORM}/arm-linux-gnueabihf/${CV_CMAKE_FOLDER}")
            elseif(BUILD_LINUX_AARCH64)
                set(OpenCV_DIR "${THIRD_PARTY_DIR}/opencv-mobile/${MOBILE_OPENCV_VERSION}/${TARGET_PLATFORM}/aarch64-linux-gnu/${CV_CMAKE_FOLDER}")
            endif()
        endif()
    else ()
        set(PLAT linux)
        if (USE_MOBILE_OPENCV_IN_LOCAL)
            set(OpenCV_DIR "${THIRD_PARTY_DIR}/opencv-mobile/${MOBILE_OPENCV_VERSION}/ubuntu-2004/lib/cmake/opencv4")
        endif ()
    endif ()
    find_package(OpenCV REQUIRED)
endif ()

# Set install path
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install")

add_subdirectory(cpp/inspireface)  # Add a child project: InspireFace Source
get_property(InspireFace TARGET InspireFace PROPERTY InspireFace)

if (BUILD_WITH_SAMPLE)
    add_subdirectory(cpp/sample)  # Add a child project: Samples
endif ()

if (BUILD_WITH_TEST)
    add_subdirectory(cpp/test)  # Add a child project: Unit-Test
endif ()

# Print Message
message(STATUS "InspireFace Project Global:")
message(STATUS "\t CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "\t CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "\t THIRD_PARTY_DIR: ${THIRD_PARTY_DIR}")
message(STATUS "\t SANITIZE_ADDRESS: ${SANITIZE_ADDRESS}")
message(STATUS "\t SANITIZE_LEAK: ${SANITIZE_LEAK}")
message(STATUS "\t USE_MOBILE_OPENCV_IN_LOCAL: ${USE_MOBILE_OPENCV_IN_LOCAL}")
message(STATUS "\t OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "\t ENABLE_RKNN: ${ENABLE_RKNN}")
if (ENABLE_RKNN)
    message(STATUS "\t RKNPU_MAJOR: ${RKNPU_MAJOR}")
    message(STATUS "\t RK_DEVICE_TYPE: ${RK_DEVICE_TYPE}")
endif ()
message(STATUS "\t BUILD_LINUX_ARM7: ${BUILD_LINUX_ARM7}")
message(STATUS "\t BUILD_LINUX_AARCH64: ${BUILD_LINUX_AARCH64}")
message(STATUS "\t BUILD_WITH_TEST: ${BUILD_WITH_TEST}")
message(STATUS "\t BUILD_WITH_SAMPLE: ${BUILD_WITH_SAMPLE}")
message(STATUS "\t CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
