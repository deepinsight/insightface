cmake_minimum_required(VERSION 3.20)

if(NOT OHOS)
    message(FATAL_ERROR "InspireFaceNapi can only be built with the OpenHarmony toolchain")
endif()

set(ISF_OHOS_PACKAGE_VERSION
        "${INSPIRE_FACE_VERSION_MAJOR}.${INSPIRE_FACE_VERSION_MINOR}.${INSPIRE_FACE_VERSION_PATCH}")
foreach(package_manifest
        "${PROJECT_SOURCE_DIR}/harmony/inspireface/oh-package.json5"
        "${PROJECT_SOURCE_DIR}/harmony/inspireface/src/main/cpp/types/libinspireface_napi/oh-package.json5")
    file(READ "${package_manifest}" package_manifest_content)
    string(FIND "${package_manifest_content}" "\"version\": \"${ISF_OHOS_PACKAGE_VERSION}\"" version_offset)
    if(version_offset EQUAL -1)
        message(FATAL_ERROR
                "HarmonyOS package version must match InspireFace ${ISF_OHOS_PACKAGE_VERSION}: ${package_manifest}")
    endif()
endforeach()

add_library(InspireFaceNapi SHARED inspireface_napi.cpp)
target_include_directories(InspireFaceNapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
target_link_libraries(InspireFaceNapi PRIVATE
        InspireFace
        libace_napi.z.so)
target_link_options(InspireFaceNapi PRIVATE
        "-Wl,-z,noexecstack"
        # The ArkTS distribution has one public ABI: Node-API registration.
        # Hide C/C++ symbols absorbed from the static core and dependencies.
        "-Wl,--exclude-libs,ALL")
set_target_properties(InspireFaceNapi PROPERTIES
        OUTPUT_NAME inspireface_napi
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set_property(TARGET InspireFaceNapi APPEND PROPERTY LINK_DEPENDS
        "${PROJECT_SOURCE_DIR}/harmony/inspireface/src/main/cpp/types/libinspireface_napi/index.d.ts"
        "${PROJECT_SOURCE_DIR}/harmony/inspireface/src/main/ets/InspireFace.ets"
        "${CMAKE_CURRENT_SOURCE_DIR}/ohos_c_api_parity.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../c_api/inspireface.h")

set(ISF_NAPI_READELF_EXECUTABLE "${CMAKE_READELF}")
if(NOT ISF_NAPI_READELF_EXECUTABLE)
    find_program(ISF_NAPI_READELF_EXECUTABLE NAMES llvm-readelf readelf)
endif()
if(NOT ISF_NAPI_READELF_EXECUTABLE)
    message(FATAL_ERROR "readelf is required to verify the HarmonyOS Node-API library")
endif()

add_custom_command(TARGET InspireFaceNapi POST_BUILD
        COMMAND "${CMAKE_COMMAND}"
                "-DISF_NAPI_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/inspireface_napi.cpp"
                "-DISF_NAPI_DECLARATION=${PROJECT_SOURCE_DIR}/harmony/inspireface/src/main/cpp/types/libinspireface_napi/index.d.ts"
                "-DISF_ARKTS_WRAPPER=${PROJECT_SOURCE_DIR}/harmony/inspireface/src/main/ets/InspireFace.ets"
                "-DISF_C_API_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/../../../c_api/inspireface.h"
                "-DISF_C_API_PARITY=${CMAKE_CURRENT_SOURCE_DIR}/ohos_c_api_parity.cmake"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_ohos_arkts_contract.cmake"
        COMMENT "Verifying HarmonyOS Node-API and ArkTS contract parity"
        VERBATIM)

add_custom_command(TARGET InspireFaceNapi POST_BUILD
        COMMAND "${CMAKE_COMMAND}"
                "-DISF_READELF_EXECUTABLE=${ISF_NAPI_READELF_EXECUTABLE}"
                "-DISF_NAPI_LIBRARY=$<TARGET_FILE:InspireFaceNapi>"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_ohos_napi.cmake"
        COMMENT "Verifying HarmonyOS Node-API ABI and dependencies"
        VERBATIM)

install(TARGETS InspireFaceNapi
        LIBRARY DESTINATION HarmonyOS/libs/arm64-v8a)

# Stage a source HAR layout without checking generated binaries into git.
install(DIRECTORY ${PROJECT_SOURCE_DIR}/harmony/inspireface/
        DESTINATION HarmonyOS/har
        PATTERN "build" EXCLUDE
        PATTERN ".hvigor" EXCLUDE
        PATTERN "libs" EXCLUDE)
install(TARGETS InspireFaceNapi
        LIBRARY DESTINATION HarmonyOS/har/src/main/libs/arm64-v8a)
