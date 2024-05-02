cmake_minimum_required(VERSION 3.10)
project(InspireFaceSDK)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++14")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Current version
set(INSPIRE_FACE_VERSION_MAJOR 1)
set(INSPIRE_FACE_VERSION_MINOR 0)
set(INSPIRE_FACE_VERSION_PATCH 0)
# Converts the version number to a string
string(CONCAT INSPIRE_FACE_VERSION_MAJOR_STR ${INSPIRE_FACE_VERSION_MAJOR})
string(CONCAT INSPIRE_FACE_VERSION_MINOR_STR ${INSPIRE_FACE_VERSION_MINOR})
string(CONCAT INSPIRE_FACE_VERSION_PATCH_STR ${INSPIRE_FACE_VERSION_PATCH})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/information.h.in ${CMAKE_CURRENT_SOURCE_DIR}/information.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/version.txt.in ${CMAKE_CURRENT_SOURCE_DIR}/version.txt)



option(BUILD_SHARED_LIBS "Build shared libraries (DLLs)." ON)

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(SOURCE_FILES ${SOURCE_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/c_api/inspireface.cc)        # Add C_API file


if(NOT APPLE)
    find_package(OpenCV REQUIRED)
endif()

if (NOT ANDROID)

    if (LINUX_FETCH_MNN)
        # Include FetchContent module
        include(FetchContent)
        # Fetch MNN
        message("Downloading MNN from https://github.com/alibaba/MNN.git, this may take a while.")
        FetchContent_Declare(
                mnn
                GIT_REPOSITORY https://github.com/alibaba/MNN.git
                GIT_TAG        2.7.0
        )
        set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
        set(MNN_BUILD_CONVERTER OFF CACHE BOOL "" FORCE)
        if(GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA)
            set(MNN_CUDA ON CACHE BOOL "" FORCE)
        endif()

        FetchContent_MakeAvailable(mnn)
        set(MNN_INCLUDE_DIRS ${mnn_SOURCE_DIR}/include)
        if (MNN_BUILD_SHARED_LIBS)
            set(MNN_LIBS ${mnn_BINARY_DIR})
        else()
            set(MNN_LIBS ${mnn_BINARY_DIR}/libMNN.a)
        endif ()

    elseif (GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA)
        # Use MNN Cuda
        message("Global MNN CUDA device inference")
        add_definitions("-DGLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA")
        set(MNN_INCLUDE_DIRS ${LINUX_MNN_CUDA}/include)
        link_directories(${LINUX_MNN_CUDA}/lib)
        set(MNN_LIBS MNN)

    elseif(DEFINED MNN_STATIC_PATH)
        message("Using static MNN from specified path: ${MNN_STATIC_PATH}")
        set(MNN_INCLUDE_DIRS "${MNN_STATIC_PATH}/include")
        set(MNN_LIBS "${MNN_STATIC_PATH}/lib/libMNN.a")
    else ()
        # Default or fallback case for MNN setup
        message("Default or fallback case for MNN setup")
        set(MNN_INCLUDE_DIRS ${THIRD_PARTY_DIR}/MNN/${PLAT}-static/include)
        set(MNN_LIBS ${THIRD_PARTY_DIR}/MNN/${PLAT}-static/lib/libMNN.a)
    endif ()

endif()


if (ENABLE_RKNN)
    set(RKNN_API_INCLUDE_DIRS ${THIRD_PARTY_DIR}/${RKNPU_MAJOR}/runtime/${RK_DEVICE_TYPE}/Linux/librknn_api/include)
    set(RKNN_API_LIB ${THIRD_PARTY_DIR}/${RKNPU_MAJOR}/runtime/${RK_DEVICE_TYPE}/Linux/librknn_api/${CPU_ARCH}/)
    link_directories(${RKNN_API_LIB})
endif()

# OpenCV
if(APPLE AND USE_MOBILE_OPENCV_IN_LOCAL)
    set(LINK_THIRD_LIBS ${LINK_THIRD_LIBS} "${OPENCV_FRAMEWORK}" ${MNN_LIBS} "${ACCELERATE_FRAMEWORK}")
else()
    set(LINK_THIRD_LIBS ${OpenCV_LIBS} ${MNN_LIBS})
endif()

# sqlite3
set(SOURCE_FILES ${SOURCE_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/middleware/sqlite/sqlite3.c)        # Add C_API file

if (ENABLE_RKNN)
    set(LINK_THIRD_LIBS ${LINK_THIRD_LIBS} rknn_api dl)
    # InferenceHelp use RkNN
    add_definitions("-DINFERENCE_HELPER_ENABLE_RKNN")

endif()

# cpp yaml
file(GLOB_RECURSE CPP_YAML_SRC ${CMAKE_CURRENT_SOURCE_DIR}/middleware/cpp_yaml/src/*.cpp)
set(SOURCE_FILES ${SOURCE_FILES} ${CPP_YAML_SRC})
set(CPP_YAML_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/middleware/cpp_yaml/include)

# archive
set(SOURCE_FILES ${SOURCE_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/middleware/model_archive/microtar/microtar.c)

# MNN
link_directories(${MNN_LIBS})

if(BUILD_SHARED_LIBS)
    add_library(InspireFace SHARED ${SOURCE_FILES})
else()
    add_library(InspireFace STATIC ${SOURCE_FILES})
endif()

target_compile_definitions(InspireFace PUBLIC INFERENCE_HELPER_ENABLE_MNN)
target_compile_definitions(InspireFace PUBLIC FEATURE_BLOCK_ENABLE_OPENCV)

# Include files
set(NEED_INCLUDE . ${MNN_INCLUDE_DIRS})
if (ENABLE_RKNN)
    set(NEED_INCLUDE ${NEED_INCLUDE} ${RKNN_API_INCLUDE_DIRS})
endif ()
if (BUILD_LINUX_ARM7)
    set(NEED_INCLUDE ${NEED_INCLUDE} ${OpenCV_STATIC_INCLUDE_DIR})
endif ()

# add cpp yaml header
set(NEED_INCLUDE ${NEED_INCLUDE} ${CPP_YAML_INCLUDE})

if(PLAT STREQUAL "linux")
    find_package(Threads REQUIRED)
    set(LINK_THIRD_LIBS ${LINK_THIRD_LIBS} ${CMAKE_THREAD_LIBS_INIT} dl)
endif()

target_include_directories(InspireFace PUBLIC
    ${NEED_INCLUDE}
)

if (NOT ANDROID)
    target_link_libraries(InspireFace PUBLIC ${LINK_THIRD_LIBS})

    set_target_properties(InspireFace PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/
    )
else()

endif()

# Print Message
message(STATUS "InspireFace Core:")
message(STATUS "\t Version: ${INSPIRE_FACE_VERSION_MAJOR}.${INSPIRE_FACE_VERSION_MINOR}.${INSPIRE_FACE_VERSION_PATCH}")
message(STATUS "\t LINUX_FETCH_MNN: ${LINUX_FETCH_MNN}")
message(STATUS "\t MNN_INCLUDE_DIRS: ${MNN_INCLUDE_DIRS}")
message(STATUS "\t MNN_LIBS: ${MNN_LIBS}")
message(STATUS "\t ENABLE_RKNN: ${ENABLE_RKNN}")
if (ENABLE_RKNN)
    message(STATUS "\t RKNN_API_INCLUDE_DIRS: ${RKNN_API_INCLUDE_DIRS}")
    message(STATUS "\t RKNN_API_LIB: ${RKNN_API_LIB}")
endif ()
message(STATUS "\t BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
if (GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA)
    message(STATUS "\t GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA: ${GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA}")
endif ()


# Install lib
install(TARGETS InspireFace
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/lib
        )

# Install header file
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/c_api/inspireface.h DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/include)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/herror.h DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/include)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/version.txt DESTINATION ${CMAKE_INSTALL_PREFIX}/)

if (ENABLE_RKNN)
    # Install rknn 3rd lib
    install(FILES ${RKNN_API_LIB}/librknn_api.so DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/lib)
endif ()

