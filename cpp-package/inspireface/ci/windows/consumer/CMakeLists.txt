cmake_minimum_required(VERSION 3.20)
project(InspireFaceInstalledConsumer LANGUAGES C CXX)

find_package(InspireFace CONFIG REQUIRED)
if(NOT TARGET InspireFace::InspireFace)
    message(FATAL_ERROR "The installed package must provide InspireFace::InspireFace")
endif()

set(ISF_CONSUMER_MODEL "" CACHE FILEPATH "Optional resource pack for inference smoke tests")
if(ISF_CONSUMER_MODEL AND NOT EXISTS "${ISF_CONSUMER_MODEL}")
    message(FATAL_ERROR "ISF_CONSUMER_MODEL does not exist: ${ISF_CONSUMER_MODEL}")
endif()

get_target_property(ISF_CONSUMER_LIBRARY_TYPE InspireFace::InspireFace TYPE)
if(ISF_CONSUMER_LIBRARY_TYPE STREQUAL "SHARED_LIBRARY")
    set(ISF_CONSUMER_EXPECT_SHARED 1)
elseif(ISF_CONSUMER_LIBRARY_TYPE STREQUAL "STATIC_LIBRARY")
    set(ISF_CONSUMER_EXPECT_SHARED 0)
else()
    message(FATAL_ERROR "The package must identify its shared/static library type")
endif()

add_executable(c_consumer main.c)
set_target_properties(c_consumer PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED YES
        LINKER_LANGUAGE CXX)
add_executable(cpp_consumer main.cpp)
target_compile_features(cpp_consumer PRIVATE cxx_std_14)

include(CTest)
foreach(ISF_CONSUMER IN ITEMS c_consumer cpp_consumer)
    target_link_libraries(${ISF_CONSUMER} PRIVATE InspireFace::InspireFace)
    # This is a test expectation, not a replacement for the SDK's interface macros.
    target_compile_definitions(${ISF_CONSUMER} PRIVATE
            ISF_CONSUMER_EXPECT_SHARED=${ISF_CONSUMER_EXPECT_SHARED})
    if(WIN32 AND ISF_CONSUMER_EXPECT_SHARED)
        add_custom_command(TARGET ${ISF_CONSUMER} POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                        "$<TARGET_FILE:InspireFace::InspireFace>"
                        "$<TARGET_FILE_DIR:${ISF_CONSUMER}>"
                VERBATIM)
    endif()

    if(BUILD_TESTING)
        set(ISF_CONSUMER_COMMAND "$<TARGET_FILE:${ISF_CONSUMER}>")
        if(WIN32)
            # Only the copied SDK DLL and Windows runtime may satisfy DLL lookup.
            # In particular, a developer's build-tree PATH cannot hide a bad package.
            set(ISF_CONSUMER_COMMAND "${CMAKE_COMMAND}" -E env
                    "PATH=$ENV{SystemRoot}/System32" "$<TARGET_FILE:${ISF_CONSUMER}>")
        endif()
        add_test(NAME InspireFace.Installed.${ISF_CONSUMER}
                COMMAND ${ISF_CONSUMER_COMMAND})
        set_tests_properties(InspireFace.Installed.${ISF_CONSUMER} PROPERTIES
                WORKING_DIRECTORY "$<TARGET_FILE_DIR:${ISF_CONSUMER}>"
                LABELS "install;consumer")
        if(ISF_CONSUMER_MODEL)
            add_test(NAME InspireFace.Installed.${ISF_CONSUMER}.Model
                    COMMAND ${ISF_CONSUMER_COMMAND} "${ISF_CONSUMER_MODEL}")
            set_tests_properties(InspireFace.Installed.${ISF_CONSUMER}.Model PROPERTIES
                    WORKING_DIRECTORY "$<TARGET_FILE_DIR:${ISF_CONSUMER}>"
                    LABELS "install;consumer;inference")
        endif()
    endif()
endforeach()

if(NOT ISF_CONSUMER_MODEL)
    message(STATUS "No model supplied: only the two model-free consumer tests are registered")
endif()
