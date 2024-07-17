cmake_minimum_required(VERSION 3.10)
project(InspireFace)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Current version
set(INSPIRE_FACE_VERSION_MAJOR 1)
set(INSPIRE_FACE_VERSION_MINOR 1)
set(INSPIRE_FACE_VERSION_PATCH 6)

# Converts the version number to a string
string(CONCAT INSPIRE_FACE_VERSION_MAJOR_STR ${INSPIRE_FACE_VERSION_MAJOR})
string(CONCAT INSPIRE_FACE_VERSION_MINOR_STR ${INSPIRE_FACE_VERSION_MINOR})
string(CONCAT INSPIRE_FACE_VERSION_PATCH_STR ${INSPIRE_FACE_VERSION_PATCH})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cpp/inspireface/information.h.in ${CMAKE_CURRENT_SOURCE_DIR}/cpp/inspireface/information.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cpp/inspireface/version.txt.in ${CMAKE_CURRENT_SOURCE_DIR}/cpp/inspireface/version.txt)

# Set the ISF_THIRD_PARTY_DIR variable to allow it to be set externally from the command line, or use the default path if it is not set
set(ISF_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty" CACHE PATH "Path to the third-party libraries directory")

# Check that the SANITIZE compile option is enabled
set(ISF_SANITIZE_ADDRESS OFF CACHE BOOL "Enable AddressSanitizer")
set(ISF_SANITIZE_LEAK OFF CACHE BOOL "Enable LeakSanitizer")

if (ISF_SANITIZE_ADDRESS AND ISF_SANITIZE_LEAK)
    message(FATAL_ERROR "Cannot enable both AddressSanitizer and LeakSanitizer.")
endif()

# Enable AddressSanitizer
if(ISF_SANITIZE_ADDRESS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()

# Enable LeakSanitizer
if(ISF_SANITIZE_LEAK)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=leak")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=leak")
endif()

set(TARGET_PLATFORM "drawin-x86" CACHE STRING "Target platform")


# RKNN dependency configuration
option(ISF_ENABLE_RKNN "Use RKNPU." OFF)
set(ISF_RK_DEVICE_TYPE "RV1109RV1126" CACHE STRING "Type of the device")
if (ISF_ENABLE_RKNN)
    add_definitions("-DISF_ENABLE_RKNN")
    # Device list
    set(RKNPU1_DEVICES "RV1109RV1126")
    set(RKNPU2_DEVICES "RK356X" "RK3588" "RV1106")
    set(ISF_RKNPU_MAJOR "")
    list(FIND RKNPU1_DEVICES "${ISF_RK_DEVICE_TYPE}" INDEX_RKNPU1)
    list(FIND RKNPU2_DEVICES "${ISF_RK_DEVICE_TYPE}" INDEX_RKNPU2)
    if(INDEX_RKNPU1 GREATER_EQUAL 0)
        # match rknn1
        set(ISF_RKNPU_MAJOR "rknpu1")
    elseif(INDEX_RKNPU2 GREATER_EQUAL 0)
        # match rknn2
        set(ISF_RKNPU_MAJOR "rknpu2")
    endif()
    # Result
    message(STATUS "Use ${ISF_RKNPU_MAJOR}")
endif ()


option(ISF_BUILD_LINUX_ARM7 "Platform Armv7." OFF)
option(ISF_BUILD_LINUX_AARCH64 "Platform Armv8." OFF)
option(ISF_GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA "The global inference backend uses MNN CUDA." OFF)


if (ISF_BUILD_LINUX_ARM7)
    set(CPU_ARCH "armhf")
endif()
if (ISF_BUILD_LINUX_AARCH64)
    set(CPU_ARCH "aarch64")
endif()

option(ISF_BUILD_WITH_TEST "Open Build Unit-Test." ON)
option(ISF_BUILD_WITH_SAMPLE "Open Build Sample Exec." ON)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/)

# OpenCV dependency configuration
if (APPLE)
    if (IOS)
        message(IOS_3RDPARTY=${IOS_3RDPARTY})
        set(CMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH NO)
        set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE NO)

        link_directories(${IOS_3RDPARTY})
        include_directories(${IOS_3RDPARTY}/opencv2.framework)

        set(MNN_FRAMEWORK_PATH "${IOS_3RDPARTY}/MNN.framework")
        include_directories("${MNN_FRAMEWORK_PATH}/")

        set(OpenCV_DIR "${IOS_3RDPARTY}/opencv2.framework")
#        find_package(OpenCV REQUIRED)
    else()
        message("Use apple device")
        set(PLAT darwin)
        find_package(OpenCV REQUIRED)
    endif ()
else()
    if (ISF_BUILD_LINUX_ARM7 OR ISF_BUILD_LINUX_AARCH64)
        set(DISABLE_GUI ON)
        add_definitions("-DDISABLE_GUI")
        #        set(OpenCV_DIR ${ISF_THIRD_PARTY_DIR}/opencv/opencv-linux-armhf/share/OpenCV)
        #        set(OpenCV_STATIC_INCLUDE_DIR ${PATH_3RDPARTY}/opencv/opencv-linux-armhf/include/)
        if (ISF_RK_DEVICE_TYPE STREQUAL "RV1109RV1126" AND ISF_ENABLE_RKNN)
            # In special cases, specialize for that version
            message("The OpenCV that builds the RV1109RV1126 version depends on is specialized!")
            set(OpenCV_DIR ${ISF_THIRD_PARTY_DIR}/inspireface-precompile/opencv/3.4.5/opencv-linux-armhf/share/OpenCV)
            set(OpenCV_STATIC_INCLUDE_DIR ${PATH_3RDPARTY}/inspireface-precompile/opencv/3.4.5/opencv-linux-armhf/include/)
            set(PLAT linux-arm7)
        else()
            if (VERSION_MAJOR STREQUAL "3")
                set(CV_CMAKE_FOLDER share/OpenCV)
            elseif(VERSION_MAJOR STREQUAL "4")
                set(CV_CMAKE_FOLDER lib/cmake/opencv4)
            endif ()
            if(ISF_BUILD_LINUX_ARM7)
                set(PLAT linux-arm7)
                message("The OpenCV that builds the gnueabihf version depends on is specialized!")
                set(OpenCV_DIR ${ISF_THIRD_PARTY_DIR}/inspireface-precompile/opencv/3.4.5/opencv-linux-armhf/share/OpenCV)
                set(OpenCV_STATIC_INCLUDE_DIR ${PATH_3RDPARTY}/inspireface-precompile/opencv/3.4.5/opencv-linux-armhf/include/)
            elseif(ISF_BUILD_LINUX_AARCH64)
                set(PLAT linux-aarch64)
                message("The OpenCV that builds the aarch64 version depends on is specialized!")
                set(OpenCV_DIR ${ISF_THIRD_PARTY_DIR}/inspireface-precompile/opencv/3.4.5/opencv-linux-aarch64/share/OpenCV)
                set(OpenCV_STATIC_INCLUDE_DIR ${PATH_3RDPARTY}/inspireface-precompile/opencv/3.4.5/opencv-linux-aarch64/include/)
            endif()
        endif()
    else ()
        set(PLAT linux)
    endif ()
    find_package(OpenCV REQUIRED)
endif ()

set(ISF_LINUX_MNN_CUDA "" CACHE STRING "Path to CUDA directory")

if (ISF_GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA)
    message("Global MNN CUDA device inference")
    add_definitions("-DISF_GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA")
    # Use MNN Cuda
    if (NOT DEFINED ISF_LINUX_MNN_CUDA)
        # MNN Options
        set(MNN_CUDA ON)
        add_subdirectory(${ISF_THIRD_PARTY_DIR}/MNN EXCLUDE_FROM_ALL)
        set(MNN_INCLUDE_DIRS "${ISF_THIRD_PARTY_DIR}/MNN/include")
        set(MNN_LIBS MNN)
    else()
        message(ISF_LINUX_MNN_CUDA=${ISF_LINUX_MNN_CUDA})
        set(MNN_INCLUDE_DIRS ${ISF_LINUX_MNN_CUDA}/include)
        link_directories(${ISF_LINUX_MNN_CUDA}/)
        set(MNN_LIBS MNN)
    endif()

elseif(DEFINED MNN_STATIC_PATH)
    message("Using static MNN from specified path: ${MNN_STATIC_PATH}")
    set(MNN_INCLUDE_DIRS "${MNN_STATIC_PATH}/include")
    set(MNN_LIBS "${MNN_STATIC_PATH}/lib/libMNN.a")
elseif(IOS)
    message(Build iOS)
else ()
    # Default or fallback case for MNN setup
    message("Default or fallback case for MNN setup")
    # MNN Options
    set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "Build MNN as a shared library")

    add_subdirectory(${ISF_THIRD_PARTY_DIR}/MNN EXCLUDE_FROM_ALL)
    set(MNN_INCLUDE_DIRS "${ISF_THIRD_PARTY_DIR}/MNN/include")
    set(MNN_LIBS MNN)

endif ()


# Set install path
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install")

add_subdirectory(cpp/inspireface)  # Add a child project: InspireFace Source
get_property(InspireFace TARGET InspireFace PROPERTY InspireFace)

if (ISF_BUILD_WITH_SAMPLE)
    add_subdirectory(cpp/sample)  # Add a child project: Samples
endif ()

if (ISF_BUILD_WITH_TEST)
    add_subdirectory(cpp/test)  # Add a child project: Unit-Test
endif ()

# Print Message
message(STATUS "InspireFace Project Global:")
message(STATUS "\t CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "\t CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "\t ISF_THIRD_PARTY_DIR: ${ISF_THIRD_PARTY_DIR}")
message(STATUS "\t ISF_SANITIZE_ADDRESS: ${ISF_SANITIZE_ADDRESS}")
message(STATUS "\t ISF_SANITIZE_LEAK: ${ISF_SANITIZE_LEAK}")
message(STATUS "\t ISF_ENABLE_RKNN: ${ISF_ENABLE_RKNN}")
if (ISF_ENABLE_RKNN)
    message(STATUS "\t ISF_RKNPU_MAJOR: ${ISF_RKNPU_MAJOR}")
    message(STATUS "\t ISF_RK_DEVICE_TYPE: ${ISF_RK_DEVICE_TYPE}")
endif ()
message(STATUS "\t ISF_BUILD_LINUX_ARM7: ${ISF_BUILD_LINUX_ARM7}")
message(STATUS "\t ISF_BUILD_LINUX_AARCH64: ${ISF_BUILD_LINUX_AARCH64}")
message(STATUS "\t ISF_BUILD_WITH_TEST: ${ISF_BUILD_WITH_TEST}")
message(STATUS "\t ISF_BUILD_WITH_SAMPLE: ${ISF_BUILD_WITH_SAMPLE}")
message(STATUS "\t OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "\t CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
