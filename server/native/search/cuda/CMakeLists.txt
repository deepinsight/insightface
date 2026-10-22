find_package(CUDAToolkit 12.9 REQUIRED)

set(IFS_SEARCH_CUDA_ARCHITECTURES
    "75-real;80-real;86-real;89-real;90-real;100-real;103-real;120"
    CACHE STRING "CUDA SASS targets plus PTX for the newest target")

add_library(ifs_search_cuda SHARED
    src/api_v2.cpp
    src/ifs_cuda_legacy.cu)

target_compile_features(ifs_search_cuda PRIVATE cxx_std_17)
set_target_properties(ifs_search_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    OUTPUT_NAME ifs_search_cuda
    VERSION ${PROJECT_VERSION}
    SOVERSION 2
    CUDA_ARCHITECTURES "${IFS_SEARCH_CUDA_ARCHITECTURES}")
install(TARGETS ifs_search_cuda
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
target_compile_definitions(ifs_search_cuda PRIVATE IFS_SEARCH_BUILDING=1)
target_include_directories(ifs_search_cuda
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ifs_search_cuda PRIVATE CUDA::cudart CUDA::cublas)
target_compile_options(ifs_search_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)

if(IFS_SEARCH_BUILD_TESTS)
    add_executable(ifs_search_cuda_smoke tests/smoke.cpp)
    target_compile_features(ifs_search_cuda_smoke PRIVATE cxx_std_17)
    target_link_libraries(ifs_search_cuda_smoke PRIVATE ifs_search_cuda)
    target_compile_options(ifs_search_cuda_smoke PRIVATE -UNDEBUG)
    add_test(NAME ifs_search_cuda_smoke COMMAND ifs_search_cuda_smoke)
    set_tests_properties(ifs_search_cuda_smoke PROPERTIES SKIP_RETURN_CODE 77)
endif()
