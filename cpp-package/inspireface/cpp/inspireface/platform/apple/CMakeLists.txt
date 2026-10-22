enable_language(OBJC)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_library(ISF_FOUNDATION Foundation REQUIRED)
find_library(ISF_COREVIDEO CoreVideo REQUIRED)

set(ISF_APPLE_OUTPUT "${CMAKE_BINARY_DIR}/apple")
set(ISF_APPLE_VERSION "${INSPIRE_FACE_VERSION_MAJOR}.${INSPIRE_FACE_VERSION_MINOR}.${INSPIRE_FACE_VERSION_PATCH}")
set(MACOSX_FRAMEWORK_BUNDLE_NAME "InspireFace")
set(ISF_APPLE_HEADERS
    include/InspireFaceApple.h include/IFInspireFace.h
    ../../c_api/inspireface.h ../../c_api/intypedef.h ../../include/inspireface/herror.h)
set(ISF_APPLE_SDK macosx)
set(ISF_APPLE_LINKAGE SHARED)
set(ISF_APPLE_FRAMEWORK_BINARY Versions/A/InspireFaceSwift)
set(ISF_APPLE_PACKAGE_ARGS)
set(ISF_APPLE_SWIFT_PLATFORM_ARGS)
if(ISF_ENABLE_APPLE_EXTENSION)
    list(APPEND ISF_APPLE_PACKAGE_ARGS --coreml)
endif()
if(IOS)
    set(ISF_APPLE_LINKAGE STATIC)
    set(ISF_APPLE_FRAMEWORK_BINARY InspireFaceSwift)
    if(CMAKE_OSX_SYSROOT MATCHES "[Ss]imulator")
        set(ISF_APPLE_SDK iphonesimulator)
    else()
        set(ISF_APPLE_SDK iphoneos)
    endif()
    if(NOT DEFINED MNN_STATIC_PATH)
        message(FATAL_ERROR "Apple iOS frameworks require MNN_STATIC_PATH for the selected platform/architecture")
    endif()
    list(APPEND ISF_APPLE_PACKAGE_ARGS --archive "${MNN_STATIC_PATH}/lib/libMNN.a")
    list(APPEND ISF_APPLE_SWIFT_PLATFORM_ARGS --static)
endif()
add_library(InspireFaceApple ${ISF_APPLE_LINKAGE} IFInspireFace.m ${ISF_APPLE_HEADERS}
    $<TARGET_OBJECTS:InspireFaceObjects> $<TARGET_OBJECTS:inspirecv>)
get_target_property(ISF_APPLE_CORE_LINKS InspireFace LINK_LIBRARIES)
if(IOS)
    # MNN is merged into the framework archive, not exposed as a second copy.
    target_link_libraries(InspireFaceApple PRIVATE ${ISF_FOUNDATION} ${ISF_COREVIDEO} "-framework CoreML" "-framework Accelerate")
else()
    target_link_libraries(InspireFaceApple PRIVATE ${ISF_APPLE_CORE_LINKS} ${ISF_FOUNDATION} ${ISF_COREVIDEO})
endif()
# Avoid adding the internal C++ include directory: launch.h shadows Apple's
# system launch.h when Foundation is compiled as Objective-C.
set(ISF_PUBLIC_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/public")
file(MAKE_DIRECTORY "${ISF_PUBLIC_INCLUDE}")
foreach(header IN LISTS ISF_APPLE_HEADERS)
    get_filename_component(name "${header}" NAME)
    configure_file("${header}" "${ISF_PUBLIC_INCLUDE}/${name}" COPYONLY)
endforeach()
target_include_directories(InspireFaceApple PRIVATE "${ISF_PUBLIC_INCLUDE}")
target_compile_options(InspireFaceApple PRIVATE "$<$<COMPILE_LANGUAGE:OBJC>:-fobjc-arc>")
set(ISF_APPLE_SWIFT_SANITIZER_ARGS)
if(ISF_SANITIZE_ADDRESS)
    target_compile_options(InspireFaceApple PRIVATE -fsanitize=address)
    target_link_options(InspireFaceApple PRIVATE -fsanitize=address)
    list(APPEND ISF_APPLE_SWIFT_SANITIZER_ARGS --sanitize-address)
endif()
option(ISF_APPLE_ENABLE_COVERAGE "Instrument the Apple adapter for per-language API execution coverage." OFF)
if(ISF_APPLE_ENABLE_COVERAGE)
    target_compile_options(InspireFaceApple PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(InspireFaceApple PRIVATE -fprofile-instr-generate)
endif()
set_target_properties(InspireFaceApple PROPERTIES
    OUTPUT_NAME InspireFace FRAMEWORK TRUE FRAMEWORK_VERSION A
    VERSION "${ISF_APPLE_VERSION}" SOVERSION "${INSPIRE_FACE_VERSION_MAJOR}"
    MACOSX_FRAMEWORK_IDENTIFIER "org.inspireface.sdk"
    MACOSX_FRAMEWORK_BUNDLE_VERSION "${ISF_APPLE_VERSION}"
    MACOSX_FRAMEWORK_SHORT_VERSION_STRING "${ISF_APPLE_VERSION}"
    PUBLIC_HEADER "${ISF_APPLE_HEADERS}"
    LIBRARY_OUTPUT_DIRECTORY "${ISF_APPLE_OUTPUT}"
    ARCHIVE_OUTPUT_DIRECTORY "${ISF_APPLE_OUTPUT}"
    INSTALL_NAME_DIR "@rpath" BUILD_WITH_INSTALL_NAME_DIR TRUE)
set_property(TARGET InspireFaceApple APPEND PROPERTY LINK_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/module.modulemap" "${CMAKE_CURRENT_SOURCE_DIR}/package_framework.py"
    "${CMAKE_SOURCE_DIR}/command/apple/binary_info.py")
if(IOS)
    set_property(TARGET InspireFaceApple APPEND PROPERTY LINK_DEPENDS "${MNN_STATIC_PATH}/lib/libMNN.a")
endif()
add_custom_command(TARGET InspireFaceApple POST_BUILD
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/package_framework.py"
        --framework "$<TARGET_BUNDLE_DIR:InspireFaceApple>"
        --modulemap "${CMAKE_CURRENT_SOURCE_DIR}/module.modulemap"
        --sdk "${ISF_APPLE_SDK}" --minimum "${CMAKE_OSX_DEPLOYMENT_TARGET}"
        ${ISF_APPLE_PACKAGE_ARGS}
    VERBATIM)

set(ISF_SWIFT_FRAMEWORK "${ISF_APPLE_OUTPUT}/InspireFaceSwift.framework")
add_custom_command(OUTPUT "${ISF_SWIFT_FRAMEWORK}/${ISF_APPLE_FRAMEWORK_BINARY}"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/build_swift.py"
        --sdk "${ISF_APPLE_SDK}" ${ISF_APPLE_SWIFT_PLATFORM_ARGS}
        --framework-dir "${ISF_APPLE_OUTPUT}" --version "${ISF_APPLE_VERSION}"
        --source "${CMAKE_CURRENT_SOURCE_DIR}/swift/InspireFace.swift"
        --deployment-target "${CMAKE_OSX_DEPLOYMENT_TARGET}"
        --architectures "${CMAKE_OSX_ARCHITECTURES}"
        ${ISF_APPLE_SWIFT_SANITIZER_ARGS}
    DEPENDS InspireFaceApple swift/InspireFace.swift build_swift.py "${CMAKE_SOURCE_DIR}/command/apple/binary_info.py"
    VERBATIM)
add_custom_target(InspireFaceSwift ALL DEPENDS "${ISF_SWIFT_FRAMEWORK}/${ISF_APPLE_FRAMEWORK_BINARY}")
install(TARGETS InspireFaceApple FRAMEWORK DESTINATION .)
install(DIRECTORY "${ISF_SWIFT_FRAMEWORK}" DESTINATION .)
if(ISF_BUILD_APPLE_TESTS)
    add_subdirectory("${PROJECT_SOURCE_DIR}/../test/apple" "${CMAKE_BINARY_DIR}/apple-tests")
endif()
