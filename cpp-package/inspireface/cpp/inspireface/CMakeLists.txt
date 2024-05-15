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
set(INSPIRE_FACE_VERSION_PATCH 1)
# Converts the version number to a string
string(CONCAT INSPIRE_FACE_VERSION_MAJOR_STR ${INSPIRE_FACE_VERSION_MAJOR})
string(CONCAT INSPIRE_FACE_VERSION_MINOR_STR ${INSPIRE_FACE_VERSION_MINOR})
string(CONCAT INSPIRE_FACE_VERSION_PATCH_STR ${INSPIRE_FACE_VERSION_PATCH})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/information.h.in ${CMAKE_CURRENT_SOURCE_DIR}/information.h)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/version.txt.in ${CMAKE_CURRENT_SOURCE_DIR}/version.txt)

option(ISF_BUILD_SHARED_LIBS "Build shared libraries (DLLs)." ON)

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(SOURCE_FILES ${SOURCE_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/c_api/inspireface.cc)        # Add C_API file

if (ISF_ENABLE_RKNN)
    set(ISF_RKNN_API_INCLUDE_DIRS ${ISF_THIRD_PARTY_DIR}/inspireface-precompile/rknn/${ISF_RKNPU_MAJOR}/runtime/${ISF_RK_DEVICE_TYPE}/Linux/librknn_api/include)
    set(ISF_RKNN_API_LIB ${ISF_THIRD_PARTY_DIR}/inspireface-precompile/rknn/${ISF_RKNPU_MAJOR}/runtime/${ISF_RK_DEVICE_TYPE}/Linux/librknn_api/${CPU_ARCH}/)
    link_directories(${ISF_RKNN_API_LIB})
endif()

# OpenCV
set(LINK_THIRD_LIBS ${OpenCV_LIBS} ${MNN_LIBS})

# SQLite3
set(SOURCE_FILES ${SOURCE_FILES} ${ISF_THIRD_PARTY_DIR}/inspireface-precompile/sqlite/sqlite3.c)   # Add SQLite3 C_API file
set(SQLITE_INCLUDE ${ISF_THIRD_PARTY_DIR}/inspireface-precompile/sqlite/)

if (ISF_ENABLE_RKNN)
    set(LINK_THIRD_LIBS ${LINK_THIRD_LIBS} rknn_api dl)
    # InferenceHelp use RkNN
    add_definitions("-DINFERENCE_HELPER_ENABLE_RKNN")

endif()

# cpp yaml
file(GLOB_RECURSE CPP_YAML_SRC ${ISF_THIRD_PARTY_DIR}/yaml-cpp/src/*.cpp)
set(SOURCE_FILES ${SOURCE_FILES} ${CPP_YAML_SRC})
set(CPP_YAML_INCLUDE ${ISF_THIRD_PARTY_DIR}/yaml-cpp/include)

# archive
set(SOURCE_FILES ${SOURCE_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/middleware/model_archive/microtar/microtar.c)

# MNN
link_directories(${MNN_LIBS})

if(ISF_BUILD_SHARED_LIBS)
    add_library(InspireFace SHARED ${SOURCE_FILES})
else()
    add_library(InspireFace STATIC ${SOURCE_FILES})
endif()

target_compile_definitions(InspireFace PUBLIC INFERENCE_HELPER_ENABLE_MNN)
target_compile_definitions(InspireFace PUBLIC FEATURE_BLOCK_ENABLE_OPENCV)

# Include files
set(NEED_INCLUDE . ${MNN_INCLUDE_DIRS})
if (ISF_ENABLE_RKNN)
    set(NEED_INCLUDE ${NEED_INCLUDE} ${ISF_RKNN_API_INCLUDE_DIRS})
endif ()

if (ISF_BUILD_LINUX_ARM7)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
endif()

if (ISF_BUILD_LINUX_ARM7 OR ISF_BUILD_LINUX_AARCH64)
    set(NEED_INCLUDE ${NEED_INCLUDE} ${OpenCV_STATIC_INCLUDE_DIR})
endif ()

# add cpp yaml header
set(NEED_INCLUDE ${NEED_INCLUDE} ${CPP_YAML_INCLUDE} ${SQLITE_INCLUDE})

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
message(STATUS "\t MNN_INCLUDE_DIRS: ${MNN_INCLUDE_DIRS}")
message(STATUS "\t MNN_LIBS: ${MNN_LIBS}")
message(STATUS "\t ISF_BUILD_SHARED_LIBS: ${ISF_BUILD_SHARED_LIBS}")
message(STATUS "\t ISF_ENABLE_RKNN: ${ISF_ENABLE_RKNN}")
if (ISF_ENABLE_RKNN)
    message(STATUS "\t ISF_RKNN_API_INCLUDE_DIRS: ${ISF_RKNN_API_INCLUDE_DIRS}")
    message(STATUS "\t ISF_RKNN_API_LIB: ${ISF_RKNN_API_LIB}")
endif ()
if (ISF_GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA)
    message(STATUS "\t ISF_GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA: ${ISF_GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA}")
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

if (ISF_ENABLE_RKNN)
    # Install rknn 3rd lib
    install(FILES ${ISF_RKNN_API_LIB}/librknn_api.so DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/lib)
endif ()

