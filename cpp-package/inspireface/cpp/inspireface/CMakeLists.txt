cmake_minimum_required(VERSION 3.20)
project(InspireFaceSDK)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

option(ISF_BUILD_SHARED_LIBS "Build shared libraries (DLLs)." ON)

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(SOURCE_FILES ${SOURCE_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/c_api/inspireface.cc)        # Add C_API file


if (ISF_ENABLE_RKNN)
    if (ISF_RKNPU_MAJOR STREQUAL "rknpu1")
        set(ISF_RKNN_API_INCLUDE_DIRS ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/rknn/${ISF_RKNPU_MAJOR}/runtime/${ISF_RK_DEVICE_TYPE}/Linux/librknn_api/include)
        set(ISF_RKNN_API_LIB ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/rknn/${ISF_RKNPU_MAJOR}/runtime/${ISF_RK_DEVICE_TYPE}/Linux/librknn_api/${CPU_ARCH}/)
        link_directories(${ISF_RKNN_API_LIB})
        set(RKNN_LINKED rknn_api)
    else()
        if(ANDROID)
            set(RK_PLATFORM "Android")
            set(ISF_RKNN_API_INCLUDE_DIRS ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/rknn/${ISF_RKNPU_MAJOR}/runtime/${RK_PLATFORM}/librknn_api/include)
            set(ISF_RKNN_API_LIB ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/rknn/${ISF_RKNPU_MAJOR}/runtime/${RK_PLATFORM}/librknn_api/${ANDROID_ABI}/)
            link_directories(${ISF_RKNN_API_LIB})
            set(RKNN_LINKED rknnrt)
        else()
            set(RK_PLATFORM "Linux")
            set(ISF_RKNN_API_INCLUDE_DIRS ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/rknn/${ISF_RKNPU_MAJOR}/runtime/${RK_PLATFORM}/librknn_api/include)
            if (ISF_RK_COMPILER_TYPE STREQUAL "aarch64")
                set(ISF_RKNN_API_LIB ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/rknn/${ISF_RKNPU_MAJOR}/runtime/${RK_PLATFORM}/librknn_api/${ISF_RK_COMPILER_TYPE}/)
                link_directories(${ISF_RKNN_API_LIB})
            else()
                # For rknpu2 with armv7, we recommend linking static libraries by default
                set(ISF_RKNN_API_LIB ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/rknn/${ISF_RKNPU_MAJOR}/runtime/${RK_PLATFORM}/librknn_api/${ISF_RK_COMPILER_TYPE}/librknnmrt.a)
                set(RKNN_LINKED ${ISF_RKNN_API_LIB})
                set(RKNN_USE_STATIC_LIBS TRUE)
            endif()
        endif()
        
    endif()
    
endif()

set(LINK_THIRD_LIBS ${MNN_LIBS})

# OpenCV
if(ISF_ENABLE_OPENCV)
    set(LINK_THIRD_LIBS ${LINK_THIRD_LIBS} ${OpenCV_LIBS})
endif()

# SQLite3
set(SOURCE_FILES ${SOURCE_FILES} 
        ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/sqlite/sqlite3.c
        ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/sqlite/sqlite-vec.c
)   # Add SQLite3 C_API file
set(SQLITE_INCLUDE ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/sqlite/)



if (ISF_ENABLE_RKNN)
    set(LINK_THIRD_LIBS ${LINK_THIRD_LIBS} ${RKNN_LINKED} dl)
    # InferenceWrapper use RkNN
    if (ISF_RKNPU_MAJOR STREQUAL "rknpu1")
        add_definitions("-DINFERENCE_WRAPPER_ENABLE_RKNN")
    elseif(ISF_RKNPU_MAJOR STREQUAL "rknpu2")
        add_definitions("-DINFERENCE_WRAPPER_ENABLE_RKNN2")
    endif()
    if (ISF_RK_COMPILER_TYPE STREQUAL "aarch64")
        set(LINK_THIRD_LIBS ${LINK_THIRD_LIBS} rknnrt)
    endif()
endif()

# cpp yaml
file(GLOB_RECURSE CPP_YAML_SRC ${ISF_THIRD_PARTY_DIR}/yaml-cpp/src/*.cpp)
set(SOURCE_FILES ${SOURCE_FILES} ${CPP_YAML_SRC})
set(CPP_YAML_INCLUDE ${ISF_THIRD_PARTY_DIR}/yaml-cpp/include)

# archive
set(SOURCE_FILES ${SOURCE_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/middleware/model_archive/core_archive/microtar/microtar.c)
# CoreArchive source file
set(CORE_ARCHIVE_SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/middleware/model_archive/core_archive/core_archive.cc CACHE PATH "")
set(SOURCE_FILES ${SOURCE_FILES} ${CORE_ARCHIVE_SOURCE_FILE})

if(ISF_ENABLE_APPLE_EXTENSION)
    # link apple libs
    find_library(FOUNDATION_LIBRARY Foundation)
    find_library(COREML_LIBRARY CoreML)
    find_library(ACCELERATE_LIBRARY Accelerate)
    set(LINK_THIRD_LIBS ${LINK_THIRD_LIBS} ${FOUNDATION_LIBRARY} ${COREML_LIBRARY} ${ACCELERATE_LIBRARY})

    # Add objective-c files
    set(SOURCE_FILES ${SOURCE_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/middleware/inference_wrapper/coreml/CoreMLAdapter.mm)

endif()

# MNN
link_directories(${MNN_LIBS})

if(ISF_BUILD_SHARED_LIBS)
    add_definitions("-DISF_BUILD_SHARED_LIBS")
    add_library(InspireFace SHARED ${SOURCE_FILES} $<TARGET_OBJECTS:inspirecv>)
else()
    add_library(InspireFace STATIC ${SOURCE_FILES} $<TARGET_OBJECTS:inspirecv>)
endif()

target_compile_definitions(InspireFace PUBLIC INFERENCE_WRAPPER_ENABLE_MNN)
target_compile_definitions(InspireFace PUBLIC FEATURE_BLOCK_ENABLE_OPENCV)

# Include files
set(NEED_INCLUDE . ${MNN_INCLUDE_DIRS})
if (ISF_ENABLE_RKNN)
    set(NEED_INCLUDE ${NEED_INCLUDE} ${ISF_RKNN_API_INCLUDE_DIRS})
    if(ISF_ENABLE_RGA)
        set(LINK_THIRD_LIBS ${LINK_THIRD_LIBS} ${RGA_LIBS})
        set(NEED_INCLUDE ${NEED_INCLUDE} ${RGA_INCLUDE_DIRS})
    endif()
endif ()

# Add include directory
set(NEED_INCLUDE ${NEED_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/include/inspireface)

if (ISF_BUILD_LINUX_ARM7 OR ANDROID)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
endif()

if (ISF_BUILD_LINUX_ARM7 OR ISF_BUILD_LINUX_AARCH64)
    set(NEED_INCLUDE ${NEED_INCLUDE} ${OpenCV_STATIC_INCLUDE_DIR})
endif ()

if (ISF_ENABLE_TENSORRT)
    set(NEED_INCLUDE ${NEED_INCLUDE} ${ISF_TENSORRT_INCLUDE_DIRS})
    set(LINK_THIRD_LIBS ${LINK_THIRD_LIBS} ${ISF_TENSORRT_LIBRARIES})

endif()

# add cpp yaml header
set(NEED_INCLUDE ${NEED_INCLUDE} ${CPP_YAML_INCLUDE} ${SQLITE_INCLUDE})

if(PLAT STREQUAL "linux")
    find_package(Threads REQUIRED)
    set(LINK_THIRD_LIBS ${LINK_THIRD_LIBS} ${CMAKE_THREAD_LIBS_INIT} dl)
endif()

# set(LINK_THIRD_LIBS ${LINK_THIRD_LIBS} inspirecv)
set(NEED_INCLUDE ${NEED_INCLUDE} ${INSPIRECV_INCLUDE_PATH})

target_include_directories(InspireFace PUBLIC
    ${NEED_INCLUDE}
)

if (ANDROID)
    target_link_libraries(InspireFace PUBLIC ${LINK_THIRD_LIBS} jnigraphics log)

    set_target_properties(InspireFace PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/
            )
elseif(IOS)
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ObjC")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ObjC")
    target_link_libraries(InspireFace
        "-fobjc-arc"
        "-framework Metal"
        "-framework CoreML"
        "-framework Foundation"
        "-framework CoreVideo"
        "-framework CoreMedia"
        ${MNN_FRAMEWORK_PATH}
    )
    if(ISF_ENABLE_APPLE_EXTENSION)
        set(LINK_THIRD_LIBS ${LINK_THIRD_LIBS} ${ACCELERATE_LIBRARY})
    endif()
    if(ISF_ENABLE_OPENCV)
        set(LINK_THIRD_LIBS ${LINK_THIRD_LIBS} "-framework opencv2")
    endif()
    
else()
    target_link_libraries(InspireFace PUBLIC ${LINK_THIRD_LIBS})

    set_target_properties(InspireFace PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/
    )
endif()

# Print Message
message(STATUS ">>>>>>>>>>>>>")
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
if (ISF_ENABLE_RGA)
    message(STATUS "\t ISF_ENABLE_RGA: ${ISF_ENABLE_RGA}")
    message(STATUS "\t RGA_LIBS: ${RGA_LIBS}")
    message(STATUS "\t RGA_INCLUDE_DIRS: ${RGA_INCLUDE_DIRS}")
endif ()
if (ISF_GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA)
    message(STATUS "\t ISF_GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA: ${ISF_GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA}")
endif ()

message(STATUS "\t ISF_ENABLE_TENSORRT: ${ISF_ENABLE_TENSORRT}")
if (ISF_ENABLE_TENSORRT)
    message(STATUS "\t TENSORRT_INCLUDE_DIR: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "\t TENSORRT_LIBRARY_INFER: ${TENSORRT_LIBRARY_INFER}")
    message(STATUS "\t TENSORRT_LIBRARY_RUNTIME: ${TENSORRT_LIBRARY_RUNTIME}")
    message(STATUS "\t CUDA_RUNTIME_LIBRARY: ${CUDA_RUNTIME_LIBRARY}")
endif()

# Install lib
install(TARGETS InspireFace
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/lib
)

# Install header file
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/c_api/inspireface.h DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/include)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/c_api/intypedef.h DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/include)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/inspireface/herror.h DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/include)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/version.txt DESTINATION ${CMAKE_INSTALL_PREFIX}/)

if(ISF_INSTALL_CPP_HEADER)
    # Install cpp api header file
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/inspireface/ DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/include/inspireface/)

    install(DIRECTORY ${INSPIRECV_INCLUDE_PATH}/inspirecv DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/include/)
endif()

if (ISF_ENABLE_RKNN AND ISF_RKNPU_MAJOR STREQUAL "rknpu1")
    # Install rknn 3rd lib
    install(FILES ${ISF_RKNN_API_LIB}/librknn_api.so DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/lib)
endif ()

if(ISF_ENABLE_RKNN AND ISF_RKNPU_MAJOR STREQUAL "rknpu2" AND RK_PLATFORM STREQUAL "Android")
    install(FILES ${ISF_RKNN_API_LIB}/librknnrt.so DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/lib)
endif()


if (ISF_RK_COMPILER_TYPE STREQUAL "aarch64")
    install(FILES ${ISF_RKNN_API_LIB}/librknnrt.so DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/lib)
endif()

if (NOT IOS)
    if (NOT ISF_BUILD_SHARED_LIBS)
        if(MNN_BUILD_SHARED_LIBS)
            install(FILES ${CMAKE_BINARY_DIR}/MNN/libMNN.so DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/lib)
        else()
            install(FILES ${CMAKE_BINARY_DIR}/MNN/libMNN.a DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/lib)
        endif()
        if(RKNN_USE_STATIC_LIBS)
            # To be added: The compilation of the RK series needs to be added
            install(FILES ${ISF_RKNN_API_LIB} DESTINATION ${CMAKE_INSTALL_PREFIX}/InspireFace/lib)
        endif()
        
    endif()

endif()