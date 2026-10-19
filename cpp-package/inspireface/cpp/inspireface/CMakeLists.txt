cmake_minimum_required(VERSION 3.20)
project(InspireFaceSDK)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

option(ISF_BUILD_SHARED_LIBS "Build shared libraries (DLLs)." ON)

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Platform language bindings are configured explicitly rather than through the
# recursive glob. Android Java sources are added to the core when enabled;
# desktop JNI and the HarmonyOS Node-API module link the core separately.
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/platform/ohos/napi/inspireface_napi\\.cpp$")
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/platform/jni/portable/.*\\.cpp$")
# The factory has used InferenceWrapperRKNNAdapter for both RKNN generations
# since the adapter was introduced. Do not compile the older parallel
# implementation, which has no construction site and a separate unsafe output
# lifecycle.
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/middleware/inference_wrapper/inference_wrapper_rknn\\.cpp$")
set(SOURCE_FILES ${SOURCE_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/c_api/inspireface.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/c_api/face_result_snapshot.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/c_api/error_message.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/c_api/resource_pack_validator.cpp)  # Add C API implementation files


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
set(SQLITE_VEC_SOURCE ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/sqlite/sqlite-vec.c)
set(SOURCE_FILES ${SOURCE_FILES} 
        ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/sqlite/sqlite3.c
        ${SQLITE_VEC_SOURCE}
)   # Add SQLite3 C_API file
set(SQLITE_INCLUDE ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/sqlite/)

# sqlite-vec uses BSD u_int*_t aliases but does not include the header that
# declares them. glibc exposes the aliases transitively; OpenHarmony's musl
# correctly requires sys/types.h to be included explicitly.
if(OHOS)
    set_source_files_properties(${SQLITE_VEC_SOURCE} PROPERTIES
            COMPILE_OPTIONS "-include;sys/types.h")
endif()



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

    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/middleware/inference_wrapper/coreml/CoreMLAdapter.mm
        PROPERTIES COMPILE_OPTIONS "-Werror=unguarded-availability;-Werror=unguarded-availability-new")
    # Add objective-c files
    set(SOURCE_FILES ${SOURCE_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/middleware/inference_wrapper/coreml/CoreMLAdapter.mm)

endif()

set(ISF_CORE_SOURCES ${SOURCE_FILES})
if(ISF_BUILD_APPLE_FRAMEWORK)
    # Compile the core once. The original library and framework are separate
    # final links of these same objects, never dependencies of one another.
    add_library(InspireFaceObjects OBJECT ${SOURCE_FILES})
    set(ISF_CORE_SOURCES $<TARGET_OBJECTS:InspireFaceObjects>)
    set_target_properties(InspireFaceObjects PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(ISF_BUILD_SHARED_LIBS)
    add_definitions("-DISF_BUILD_SHARED_LIBS")
    add_library(InspireFace SHARED ${ISF_CORE_SOURCES} $<TARGET_OBJECTS:inspirecv>)
else()
    add_library(InspireFace STATIC ${ISF_CORE_SOURCES} $<TARGET_OBJECTS:inspirecv>)
endif()

if(WIN32)
    target_compile_definitions(InspireFace PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    if(ISF_BUILD_SHARED_LIBS)
        target_compile_definitions(InspireFace PRIVATE ISF_EXPORTS PUBLIC ISF_BUILD_SHARED_LIBS)
        # InspireCV objects are part of this DLL; retain the existing embedding
        # contract and export their public API from the containing SDK.
        target_compile_definitions(inspirecv PRIVATE "INSPIRECV_API=__declspec(dllexport)")
        target_compile_definitions(InspireFace
                PRIVATE "INSPIRECV_API=__declspec(dllexport)"
                INTERFACE "INSPIRECV_API=__declspec(dllimport)")
    endif()
    # Match the native name already used by the Python package on Windows.
    set_target_properties(InspireFace PROPERTIES PREFIX "lib"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()
if(MSVC)
    target_compile_options(InspireFace PRIVATE /utf-8 /bigobj)
endif()

if(OHOS AND ISF_BUILD_OHOS_NAPI)
    set_target_properties(InspireFace PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_compile_definitions(InspireFace PUBLIC INFERENCE_WRAPPER_ENABLE_MNN)
target_compile_definitions(InspireFace PUBLIC FEATURE_BLOCK_ENABLE_OPENCV)

# Linux and OpenHarmony reject or restrict shared objects that request an
# executable stack. MNN's hand-written assembly is linked into this target and
# linkers can conservatively mark the result executable when stack metadata is
# absent, even though the kernels do not execute code from the stack.
if((CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "OHOS")
        AND ISF_BUILD_SHARED_LIBS)
    target_link_options(InspireFace PRIVATE "-Wl,-z,noexecstack")

    set(ISF_READELF_EXECUTABLE "${CMAKE_READELF}")
    if(NOT ISF_READELF_EXECUTABLE)
        find_program(ISF_READELF_EXECUTABLE NAMES llvm-readelf readelf)
    endif()
    if(NOT ISF_READELF_EXECUTABLE)
        message(FATAL_ERROR "readelf is required to verify the Linux SDK stack permissions")
    endif()

    set(ISF_NOEXECSTACK_VERIFIER "${CMAKE_CURRENT_SOURCE_DIR}/cmake/verify_noexecstack.cmake")
    add_custom_command(TARGET InspireFace POST_BUILD
            COMMAND "${CMAKE_COMMAND}"
                    "-DISF_READELF_EXECUTABLE=${ISF_READELF_EXECUTABLE}"
                    "-DISF_ELF_FILE=$<TARGET_FILE:InspireFace>"
                    -P "${ISF_NOEXECSTACK_VERIFIER}"
            COMMENT "Verifying that libInspireFace does not request an executable stack"
            VERBATIM)

    if(BUILD_TESTING)
        add_test(NAME InspireFace.ELF.NoExecStack
                COMMAND "${CMAKE_COMMAND}"
                        "-DISF_READELF_EXECUTABLE=${ISF_READELF_EXECUTABLE}"
                        "-DISF_ELF_FILE=$<TARGET_FILE:InspireFace>"
                        -P "${ISF_NOEXECSTACK_VERIFIER}")
        set_tests_properties(InspireFace.ELF.NoExecStack PROPERTIES
                LABELS "compatibility;security")
    endif()
endif()

if(ISF_BUILD_WITH_SAMPLE AND NOT ISF_BUILD_SHARED_LIBS)
    target_compile_definitions(InspireFace PUBLIC ISF_ENABLE_MNN_CACHE_GUARD)
endif()
if(ISF_ENABLE_INSPIRECV_TASK_PREPROCESS)
    target_compile_definitions(InspireFace PRIVATE ISF_ENABLE_INSPIRECV_TASK_PREPROCESS)
endif()

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

if (ISF_BUILD_LINUX_ARM7)
    target_compile_options(InspireFace PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mfpu=neon>)
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
elseif(PLAT STREQUAL "ohos")
    find_package(Threads REQUIRED)
    set(LINK_THIRD_LIBS ${LINK_THIRD_LIBS} ${CMAKE_THREAD_LIBS_INIT})
endif()

# set(LINK_THIRD_LIBS ${LINK_THIRD_LIBS} inspirecv)
set(NEED_INCLUDE ${NEED_INCLUDE} ${INSPIRECV_INCLUDE_PATH})

target_include_directories(InspireFace PUBLIC
    ${NEED_INCLUDE}
)

if(ISF_BUILD_APPLE_FRAMEWORK)
    foreach(prop INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS)
        set_property(TARGET InspireFaceObjects PROPERTY ${prop} "$<TARGET_PROPERTY:InspireFace,${prop}>")
    endforeach()
endif()

if (ANDROID)
    target_link_options(${CMAKE_PROJECT_NAME} PRIVATE "-Wl,-z,max-page-size=16384")
    if(ANDROID_NDK_MAJOR LESS_EQUAL 22)
        target_link_options(${CMAKE_PROJECT_NAME} PRIVATE "-Wl,-z,common-page-size=16384")
    endif()
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
        ${LINK_THIRD_LIBS}
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

    if(APPLE AND ISF_BUILD_APPLE_FRAMEWORK)
        set_target_properties(InspireFace PROPERTIES INSTALL_NAME_DIR "@rpath" BUILD_WITH_INSTALL_NAME_DIR TRUE)
    endif()

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

if(ISF_BUILD_APPLE_FRAMEWORK)
    add_subdirectory(platform/apple)
endif()

# Install lib
install(TARGETS InspireFace
        RUNTIME DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/lib
        LIBRARY DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/lib
        ARCHIVE DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/lib
)

# Install header file
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/c_api/inspireface.h DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/include)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/c_api/intypedef.h DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/include)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/inspireface/herror.h DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/include)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/version.txt DESTINATION ${ISF_INSTALL_ROOT}/)

if(ANDROID)
    # Ship the Java declarations with the JNI-enabled Android library so the
    # native and managed interfaces cannot be released independently.
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/platform/jni/java/
            DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/java
            FILES_MATCHING PATTERN "*.java")
endif()

if(ISF_INSTALL_CPP_HEADER)
    # Install cpp api header file
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/inspireface/ DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/include/inspireface/)

    install(DIRECTORY ${INSPIRECV_INCLUDE_PATH}/inspirecv DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/include/)
endif()

if (ISF_ENABLE_RKNN AND ISF_RKNPU_MAJOR STREQUAL "rknpu1")
    # Install rknn 3rd lib
    install(FILES ${ISF_RKNN_API_LIB}/librknn_api.so DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/lib)
endif ()

if(ISF_ENABLE_RKNN AND ISF_RKNPU_MAJOR STREQUAL "rknpu2" AND RK_PLATFORM STREQUAL "Android")
    install(FILES ${ISF_RKNN_API_LIB}/librknnrt.so DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/lib)
endif()


if (ISF_RK_COMPILER_TYPE STREQUAL "aarch64")
    install(FILES ${ISF_RKNN_API_LIB}/librknnrt.so DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/lib)
endif()

if(APPLE AND DEFINED MNN_STATIC_PATH AND NOT ISF_BUILD_SHARED_LIBS)
    # Dependency caches may expose symlinks; ship the archive itself.
    get_filename_component(ISF_APPLE_MNN_ARCHIVE "${MNN_STATIC_PATH}/lib/libMNN.a" REALPATH)
    install(FILES "${ISF_APPLE_MNN_ARCHIVE}" DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/lib RENAME libMNN.a)
elseif(NOT IOS)
    if (NOT ISF_BUILD_SHARED_LIBS)
        if(WIN32 AND TARGET MNN)
            install(TARGETS MNN
                    RUNTIME DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/lib
                    LIBRARY DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/lib
                    ARCHIVE DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/lib)
        elseif(WIN32 AND DEFINED MNN_STATIC_PATH)
            install(FILES "${MNN_LIBS}" DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/lib)
        elseif(MNN_BUILD_SHARED_LIBS)
            install(FILES ${CMAKE_BINARY_DIR}/MNN/libMNN.so DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/lib)
        else()
            install(FILES ${CMAKE_BINARY_DIR}/MNN/libMNN.a DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/lib)
        endif()
        if(RKNN_USE_STATIC_LIBS)
            # To be added: The compilation of the RK series needs to be added
            install(FILES ${ISF_RKNN_API_LIB} DESTINATION ${ISF_INSTALL_ROOT}/InspireFace/lib)
        endif()
        
    endif()

endif()

# Keep the existing SDK layout and provide the import definitions needed by
# installed Windows C++ consumers, including the embedded InspireCV API.
if(WIN32)
    include(CMakePackageConfigHelpers)
    set(ISF_PACKAGE_DIRECTORY "InspireFace/lib/cmake/InspireFace")
    if(ISF_BUILD_SHARED_LIBS)
        set(ISF_PACKAGE_LIBRARY_TYPE SHARED)
    else()
        set(ISF_PACKAGE_LIBRARY_TYPE STATIC)
    endif()
    configure_package_config_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/InspireFaceConfig.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/InspireFaceConfig.cmake.in"
            INSTALL_DESTINATION "${ISF_PACKAGE_DIRECTORY}")
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/InspireFaceConfig-$<CONFIG>.cmake"
            INPUT "${CMAKE_CURRENT_BINARY_DIR}/InspireFaceConfig.cmake.in")
    write_basic_package_version_file(
            "${CMAKE_CURRENT_BINARY_DIR}/InspireFaceConfigVersion.cmake"
            VERSION "${INSPIRE_FACE_VERSION_MAJOR}.${INSPIRE_FACE_VERSION_MINOR}.${INSPIRE_FACE_VERSION_PATCH}"
            COMPATIBILITY SameMajorVersion)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/InspireFaceConfig-$<CONFIG>.cmake"
            DESTINATION "${ISF_PACKAGE_DIRECTORY}" RENAME InspireFaceConfig.cmake)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/InspireFaceConfigVersion.cmake"
            DESTINATION "${ISF_PACKAGE_DIRECTORY}")
endif()
