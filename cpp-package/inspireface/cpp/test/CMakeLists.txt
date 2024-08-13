cmake_minimum_required(VERSION 3.10)
project(HyperFaceTest)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
option(ISF_ENABLE_BENCHMARK "Enable the benchmark test cases." ON)
option(ISF_ENABLE_USE_LFW_DATA "Enable test cases for LFW data sets." ON)
option(ISF_ENABLE_TEST_EVALUATION "Enable evaluation function test cases." ON)

if (ISF_ENABLE_BENCHMARK)
    add_definitions("-DISF_ENABLE_BENCHMARK")
endif ()

if (ISF_ENABLE_USE_LFW_DATA)
    add_definitions("-DISF_ENABLE_USE_LFW_DATA")
endif ()

if (ISF_ENABLE_TEST_EVALUATION)
    add_definitions("-DISF_ENABLE_TEST_EVALUATION")
endif ()

if (ISF_ENABLE_RKNN)
    set(DEPEND rknn_api dl)
    set(ISF_RKNN_API_LIB ${ISF_THIRD_PARTY_DIR}/inspireface-precompile/rknn/${ISF_RKNPU_MAJOR}/runtime/${ISF_RK_DEVICE_TYPE}/Linux/librknn_api/${CPU_ARCH}/)
    message("Enable RKNN Inference")
    link_directories(${ISF_RKNN_API_LIB})
    set(DEPEND rknn_api dl)
endif ()

include_directories(${SRC_DIR})

# =======================Common File Configuration===========================
set(TEST_COMMON_FILES ${CMAKE_CURRENT_SOURCE_DIR}/settings/test_settings.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/settings)

# =======================Internal Import Tests===========================
file(GLOB_RECURSE INTERNAL_TEST_INTERNAL_FILES unit/source/*.cpp)
add_executable(TestSource ${CMAKE_CURRENT_SOURCE_DIR}/test_source.cpp ${INTERNAL_TEST_INTERNAL_FILES} ${TEST_COMMON_FILES})
target_link_libraries(TestSource InspireFace ${DEPEND})
target_include_directories(TestSource PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/
        ${ISF_THIRD_PARTY_DIR}/spdlog/include
        ${ISF_THIRD_PARTY_DIR}/Catch2/single_include/
        ${ISF_THIRD_PARTY_DIR}/indicators/include/
        ${SRC_DIR}
)

set_target_properties(TestSource PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/")


# =======================External API Testing===========================
file(GLOB_RECURSE TEST_INTERNAL_FILES unit/api/*.cpp)
add_executable(Test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp ${TEST_INTERNAL_FILES} ${TEST_COMMON_FILES}
        unit/api/test_evaluation.cpp)
target_link_libraries(Test InspireFace ${DEPEND})
target_include_directories(Test PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/
        ${ISF_THIRD_PARTY_DIR}/spdlog/include
        ${ISF_THIRD_PARTY_DIR}/Catch2/single_include/
        ${ISF_THIRD_PARTY_DIR}/indicators/include/
        ${SRC_DIR}
        )

set_target_properties(Test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/")


# Print Message
message(STATUS "InspireFace Test:")
message(STATUS "\t ISF_ENABLE_BENCHMARK: ${ISF_ENABLE_BENCHMARK}")
message(STATUS "\t ISF_ENABLE_USE_LFW_DATA: ${ISF_ENABLE_USE_LFW_DATA}")
message(STATUS "\t ISF_ENABLE_TEST_EVALUATION: ${ISF_ENABLE_TEST_EVALUATION}")

# Install bin
install(TARGETS Test RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/test)
install(TARGETS TestSource RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/test)