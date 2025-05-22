cmake_minimum_required(VERSION 3.20)
project(InspireFaceTest)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# If you want to test the benchmark, you need to set this to ON
option(ISF_ENABLE_BENCHMARK "Enable the benchmark test cases." ON)
# If you want to test the LFW data set, you need to set this to ON
option(ISF_ENABLE_USE_LFW_DATA "Enable test cases for LFW data sets." OFF)
# If you want to test the evaluation function, you need to set this to ON, need LFW data set.
option(ISF_ENABLE_TEST_EVALUATION "Enable evaluation function test cases." OFF)
# If you want to test the base test cases, you need to set this to ON, if you want to hide the symbols, you need to set ISF_ENABLE_SYMBOL_HIDING to ON
option(ISF_ENABLE_TEST_INTERNAL "Enable internal test cases." OFF)

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
    if (ISF_RKNPU_MAJOR STREQUAL "rknpu1")
    set(DEPEND rknn_api dl)
    set(ISF_RKNN_API_LIB ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/rknn/${ISF_RKNPU_MAJOR}/runtime/${ISF_RK_DEVICE_TYPE}/Linux/librknn_api/${CPU_ARCH}/)
        message("Enable RKNN Inference")
        link_directories(${ISF_RKNN_API_LIB})
        set(DEPEND rknn_api dl)
    elseif(ISF_RKNPU_MAJOR STREQUAL "rknpu2" AND ISF_RK_COMPILER_TYPE STREQUAL "aarch64")
        set(DEPEND rknnrt dl)
        if(ANDROID)
        set(RK_PLATFORM "Android")
        else()
            set(RK_PLATFORM "Linux")
        endif()
        set(ISF_RKNN_API_LIB ${ISF_THIRD_PARTY_DIR}/inspireface-precompile-lite/rknn/${ISF_RKNPU_MAJOR}/runtime/${RK_PLATFORM}/librknn_api/${ISF_RK_COMPILER_TYPE}/)
        message("ISF_RKNN_API_LIB: ${ISF_RKNN_API_LIB}")
        link_directories(${ISF_RKNN_API_LIB})
        set(DEPEND rknnrt dl)
    else()
        set(DEPEND dl)
    endif()
endif ()

include_directories(${SRC_DIR})

# =======================Common File Configuration===========================
set(TEST_COMMON_FILES ${CMAKE_CURRENT_SOURCE_DIR}/settings/test_settings.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/settings)

# =======================Internal Base Import Tests===========================
if (ISF_ENABLE_TEST_INTERNAL)
    file(GLOB_RECURSE INTERNAL_TEST_INTERNAL_FILES unit/base/*.cpp)
    add_executable(TestInternal ${CMAKE_CURRENT_SOURCE_DIR}/test_base.cpp ${INTERNAL_TEST_INTERNAL_FILES} ${TEST_COMMON_FILES})
    target_link_libraries(TestInternal InspireFace ${DEPEND})
    target_include_directories(TestInternal PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/
            ${ISF_THIRD_PARTY_DIR}/spdlog/include
            ${ISF_THIRD_PARTY_DIR}/Catch2/single_include/
            ${ISF_THIRD_PARTY_DIR}/indicators/include/
            ${SRC_DIR}
            ${INSPIRECV_INCLUDE_PATH}
    )

    set_target_properties(TestInternal PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/")
endif()

# =======================External API Testing===========================
file(GLOB_RECURSE TEST_INTERNAL_FILES unit/api/*.cpp)
add_executable(Test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp ${TEST_INTERNAL_FILES} ${TEST_COMMON_FILES})
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

# =======================CPP API Testing===========================
file(GLOB_RECURSE TEST_CPP_FILES unit/cpp_api/*.cpp)
add_executable(TestCPP ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp.cpp ${TEST_CPP_FILES} ${TEST_COMMON_FILES})
target_link_libraries(TestCPP InspireFace ${DEPEND})
target_include_directories(TestCPP PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/
        ${ISF_THIRD_PARTY_DIR}/spdlog/include
        ${ISF_THIRD_PARTY_DIR}/Catch2/single_include/
        ${ISF_THIRD_PARTY_DIR}/indicators/include/
        ${SRC_DIR}
        )

set_target_properties(TestCPP PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/")

# Print Message
message(STATUS ">>>>>>>>>>>>>")
message(STATUS "InspireFace Test:")
message(STATUS "\t ISF_ENABLE_BENCHMARK: ${ISF_ENABLE_BENCHMARK}")
message(STATUS "\t ISF_ENABLE_USE_LFW_DATA: ${ISF_ENABLE_USE_LFW_DATA}")
message(STATUS "\t ISF_ENABLE_TEST_EVALUATION: ${ISF_ENABLE_TEST_EVALUATION}")

# Install bin
install(TARGETS Test RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/test)
# install(TARGETS TestSource RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/test)