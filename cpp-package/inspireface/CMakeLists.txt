cmake_minimum_required(VERSION 3.20)
project(InspireFace)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(INSPIRE_FACE_SERIES_NAME "Community Edition" CACHE STRING "Series name")

# Hide symbols
option(ISF_ENABLE_SYMBOL_HIDING "Enable symbol hiding." ON)
if(ISF_ENABLE_SYMBOL_HIDING)
    if(NOT WIN32)
        set(CMAKE_C_VISIBILITY_PRESET hidden)
        set(CMAKE_CXX_VISIBILITY_PRESET hidden)
        set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
    endif()
else()
    set(CMAKE_C_VISIBILITY_PRESET default)
    set(CMAKE_CXX_VISIBILITY_PRESET default)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN NO)
endif()

# Current version
set(INSPIRE_FACE_VERSION_MAJOR 1)
set(INSPIRE_FACE_VERSION_MINOR 2)
set(INSPIRE_FACE_VERSION_PATCH 3)

# Converts the version number to a string
string(CONCAT INSPIRE_FACE_VERSION_MAJOR_STR ${INSPIRE_FACE_VERSION_MAJOR})
string(CONCAT INSPIRE_FACE_VERSION_MINOR_STR ${INSPIRE_FACE_VERSION_MINOR})
string(CONCAT INSPIRE_FACE_VERSION_PATCH_STR ${INSPIRE_FACE_VERSION_PATCH})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cpp/inspireface/version.txt.in ${CMAKE_CURRENT_SOURCE_DIR}/cpp/inspireface/version.txt)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/version.txt.in ${CMAKE_CURRENT_SOURCE_DIR}/python/version.txt)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cpp/inspireface/meta.cpp.in ${CMAKE_CURRENT_SOURCE_DIR}/cpp/inspireface/meta.cpp)

# Check that the 3rdparty folder exists
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")
    message(STATUS "3rdparty directory not found, cloning from github...")
    
    # Execute git clone and directly specify it in the current directory
    execute_process(
        COMMAND git clone --recurse-submodules https://github.com/tunmx/inspireface-3rdparty.git "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE GIT_RESULT
    )
    
    if(NOT GIT_RESULT EQUAL "0")
        message(FATAL_ERROR "Failed to clone 3rdparty repository")
    else()
        message(STATUS "Successfully cloned 3rdparty repository")
    endif()
else()
    message(STATUS "3rdparty directory already exists")
endif()

# Install cpp api header file
option(ISF_INSTALL_CPP_HEADER "Install cpp api header file." ON)

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

# TensorRT dependency configuration
set(ISF_ENABLE_TENSORRT OFF CACHE BOOL "Enable TensorRT Inference")
set(TENSORRT_ROOT "/usr/local/TensorRT" CACHE PATH "Path to TensorRT installation")
if(ISF_ENABLE_TENSORRT)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/toolchain)
    include(FindTensorRT)
    add_definitions("-DISF_ENABLE_TENSORRT")
    add_definitions("-DINFERENCE_WRAPPER_ENABLE_TENSORRT")
endif()

# RKNN dependency configuration
option(ISF_ENABLE_RKNN "Use RKNPU." OFF)
set(ISF_RK_DEVICE_TYPE "RV1109RV1126" CACHE STRING "Type of the device")
set(ISF_RK_COMPILER_TYPE "armhf" CACHE STRING "Type of the compiler")
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

    if(ISF_RK_DEVICE_TYPE STREQUAL "RV1106")
        set(ISF_RKNPU_RV1106 ON)
        add_definitions("-DISF_RKNPU_RV1106")
    endif()
    
endif ()

# Enable Rockchip RGA
option(ISF_ENABLE_RGA "Use Rockchip RGA." OFF)
if(ISF_ENABLE_RGA)
    if(NOT ISF_ENABLE_RKNN)
        message(FATAL_ERROR "RGA requires RKNN to be enabled. Please enable ISF_ENABLE_RKNN first.")
    endif()
    if(NOT ISF_RKNPU_MAJOR STREQUAL "rknpu2")
        message(FATAL_ERROR "RGA requires RKNPU2. Please use a device that supports RKNPU2.")
    endif()

    add_definitions("-DISF_ENABLE_RGA")
    if(ANDROID)
        set(PLAT Android)
    else()
        set(PLAT Linux)
    endif()
    set(RGA_LIBS ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/librga/${PLAT}/${ISF_RK_COMPILER_TYPE}/librga.a)
    set(RGA_INCLUDE_DIRS ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/librga/include)
endif()

option(ISF_ENABLE_COST_TIME "Enable cost time." OFF)
if(ISF_ENABLE_COST_TIME)
    add_definitions("-DISF_ENABLE_COST_TIME")
endif()

# Apple device dedicated expansion services, support for some models coreml reasoning and so on
set(ISF_ENABLE_APPLE_EXTENSION OFF CACHE BOOL "Enable Apple device extension services, such as coreml, etc.")
if(ISF_ENABLE_APPLE_EXTENSION)  
    add_definitions("-DISF_ENABLE_APPLE_EXTENSION")
    add_definitions("-DINFERENCE_WRAPPER_ENABLE_COREML")
endif()

# Platform configuration
option(ISF_BUILD_LINUX_ARM7 "Platform Armv7." OFF)
option(ISF_BUILD_LINUX_AARCH64 "Platform Armv8." OFF)
option(ISF_GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA "The global inference backend uses MNN CUDA." OFF)


if (ISF_BUILD_LINUX_ARM7)
    set(CPU_ARCH "armhf")
endif()
if (ISF_BUILD_LINUX_AARCH64)
    set(CPU_ARCH "aarch64")
endif()

# If you want to build the unit-test, you need to set this to ON
option(ISF_BUILD_WITH_TEST "Open Build Unit-Test." ON)

# If you want to build the sample, you need to set this to ON
option(ISF_BUILD_WITH_SAMPLE "Open Build Sample Exec." ON)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/)

option(ISF_NEVER_USE_OPENCV "Never use opencv." ON)
if(ISF_NEVER_USE_OPENCV)
    set(INSPIRECV_BACKEND_OPENCV OFF)
    set(INSPIRECV_BACKEND_OKCV_USE_OPENCV OFF)
    set(INSPIRECV_BACKEND_OKCV_USE_OPENCV_IO OFF)
    set(INSPIRECV_BACKEND_OKCV_USE_OPENCV_GUI OFF)
endif()


option(ISF_ENABLE_OPENCV "Depends on opencv." OFF)
if(INSPIRECV_BACKEND_OPENCV OR INSPIRECV_BACKEND_OKCV_USE_OPENCV OR INSPIRECV_BACKEND_OKCV_USE_OPENCV_IO OR INSPIRECV_BACKEND_OKCV_USE_OPENCV_GUI)
    set(ISF_ENABLE_OPENCV ON)
    add_definitions("-DISF_ENABLE_OPENCV")
endif()


include_directories(${ISF_THIRD_PARTY_DIR}/InspireCV/3rdparty/Eigen-3.4.0-Headers)

if(APPLE)
    if(IOS)
        add_definitions(-DTARGET_OS_IOS)
        message(IOS_3RDPARTY=${IOS_3RDPARTY})
        set(CMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH NO)
        set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE NO)
            add_definitions(-DTARGET_OS_IOS)
        message(IOS_3RDPARTY=${IOS_3RDPARTY})
        set(CMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH NO)
        set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE NO)

        set(MNN_FRAMEWORK_PATH "${IOS_3RDPARTY}/MNN.framework")
        include_directories("${MNN_FRAMEWORK_PATH}/")
        link_directories(${IOS_3RDPARTY})
    else()
        message("Use apple device")
        set(PLAT darwin)
    endif()
else()
    message("Use linux device")
    set(PLAT linux)
endif()

# OpenCV dependency configuration
if(ISF_ENABLE_OPENCV)
    if (APPLE)
        if (IOS)
            link_directories(${IOS_3RDPARTY})
            include_directories(${IOS_3RDPARTY}/opencv2.framework)
            set(OpenCV_DIR "${IOS_3RDPARTY}/opencv2.framework")
        else()
            message("Use apple device")
            set(PLAT darwin)
            find_package(OpenCV REQUIRED)
        endif ()
    else()
        if (ISF_BUILD_LINUX_ARM7 OR ISF_BUILD_LINUX_AARCH64)
            set(DISABLE_GUI ON)
            add_definitions("-DDISABLE_GUI")
            if (ISF_RK_DEVICE_TYPE STREQUAL "RV1109RV1126" AND ISF_ENABLE_RKNN)
                # In special cases, specialize for that version
                message("The OpenCV that builds the RV1109RV1126 version depends on is specialized!")
                set(OpenCV_DIR ${OPENCV_PRECOMPILED_DIR}/opencv/3.4.5/opencv-linux-armhf/share/OpenCV)
                set(OpenCV_STATIC_INCLUDE_DIR ${OPENCV_PRECOMPILED_DIR}/opencv/3.4.5/opencv-linux-armhf/include/)
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
                    set(OpenCV_DIR ${OPENCV_PRECOMPILED_DIR}/opencv/3.4.5/opencv-linux-armhf/share/OpenCV)
                    set(OpenCV_STATIC_INCLUDE_DIR ${OPENCV_PRECOMPILED_DIR}/opencv/3.4.5/opencv-linux-armhf/include/)
                elseif(ISF_BUILD_LINUX_AARCH64)
                    set(PLAT linux-aarch64)
                    message("The OpenCV that builds the aarch64 version depends on is specialized!")
                    set(OpenCV_DIR ${OPENCV_PRECOMPILED_DIR}/opencv/3.4.5/opencv-linux-aarch64/share/OpenCV)
                    set(OpenCV_STATIC_INCLUDE_DIR ${OPENCV_PRECOMPILED_DIR}/opencv/3.4.5/opencv-linux-aarch64/include/)
                endif()
            endif()
        else ()
            set(PLAT linux)
        endif ()
        find_package(OpenCV REQUIRED)
    endif ()
endif ()

# InspireCV dependency configuration
set(INSPIRECV_BUILD_OBJECT_LIBS OFF CACHE BOOL "Build InspireCV as object library")
add_subdirectory(${ISF_THIRD_PARTY_DIR}/InspireCV EXCLUDE_FROM_ALL)
set(INSPIRECV_INCLUDE_PATH "${ISF_THIRD_PARTY_DIR}/InspireCV/include")
set(INSPIRECV_LIBS InspireCV)

# If you need using CUDA-enabled MNN, you need to manually configure the pre-compiled CUDA-enabled MNN library path
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

elseif(DEFINED ISF_MNN_CUSTOM_SOURCE)
    message("Using custom external MNN source path: ${ISF_MNN_CUSTOM_SOURCE}")
    # In particular, rknpu2 uses a lower version of mnn
    set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "Build MNN as a shared library")
    add_subdirectory(${ISF_MNN_CUSTOM_SOURCE} ${CMAKE_BINARY_DIR}/MNN EXCLUDE_FROM_ALL)
    set(MNN_INCLUDE_DIRS "${ISF_MNN_CUSTOM_SOURCE}/include")
    set(MNN_LIBS MNN)
    
else ()
    # Default or fallback case for MNN setup
    message("Default or fallback case for MNN setup")
    # MNN Options
    set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "Build MNN as a shared library")

    # Set MNN build output directory to CMAKE_BINARY_DIR/MNN
    add_subdirectory(${ISF_THIRD_PARTY_DIR}/MNN ${CMAKE_BINARY_DIR}/MNN EXCLUDE_FROM_ALL)
    set(MNN_INCLUDE_DIRS "${ISF_THIRD_PARTY_DIR}/MNN/include")
    set(MNN_LIBS MNN)

endif ()


# Set the installation directory to the build directory
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install")

# Add a child project: InspireFace Source
add_subdirectory(cpp/inspireface)
get_property(InspireFace TARGET InspireFace PROPERTY InspireFace)

# Add a child project: Samples
if (ISF_BUILD_WITH_SAMPLE)
    add_subdirectory(cpp/sample)  # Add a child project: Samples
endif ()

# Add a child project: Unit-Test
if (ISF_BUILD_WITH_TEST)
    add_subdirectory(cpp/test)  # Add a child project: Unit-Test
endif ()

# Print Message
message(STATUS ">>>>>>>>>>>>>")
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
    message(STATUS "\t ISF_RK_COMPILER_TYPE: ${ISF_RK_COMPILER_TYPE}")
    if(ISF_RKNPU_RV1106)
        message(STATUS "\t ISF_RKNPU_RV1106: ${ISF_RKNPU_RV1106}")
    endif()
endif ()
message(STATUS "\t ISF_BUILD_LINUX_ARM7: ${ISF_BUILD_LINUX_ARM7}")
message(STATUS "\t ISF_BUILD_LINUX_AARCH64: ${ISF_BUILD_LINUX_AARCH64}")
message(STATUS "\t ISF_BUILD_WITH_TEST: ${ISF_BUILD_WITH_TEST}")
message(STATUS "\t ISF_BUILD_WITH_SAMPLE: ${ISF_BUILD_WITH_SAMPLE}")
message(STATUS "\t OpenCV_DIR: ${OpenCV_DIR}")
if(ISF_ENABLE_OPENCV)
    message(STATUS "\t OpenCV: Enabled")
else()
    message(STATUS "\t OpenCV: Disabled")
endif()

if(ISF_ENABLE_APPLE_EXTENSION)
    message(STATUS "\t ISF_ENABLE_APPLE_EXTENSION: ${ISF_ENABLE_APPLE_EXTENSION}")
endif()

# Install cpp api header file
if(ISF_INSTALL_CPP_HEADER)
    message(STATUS "\t ISF_INSTALL_CPP_HEADER: ${ISF_INSTALL_CPP_HEADER}")
endif()


message(STATUS "\t CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
