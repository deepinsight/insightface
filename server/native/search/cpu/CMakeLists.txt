add_library(ifs_search_cpu SHARED
    src/api_v2.cpp
    src/index_legacy.cpp
    src/kernels_portable.cpp
    src/kernels_x86.cpp)

target_compile_features(ifs_search_cpu PRIVATE cxx_std_17)
target_compile_definitions(ifs_search_cpu PRIVATE IFS_SEARCH_BUILDING=1)
target_include_directories(ifs_search_cpu
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(ifs_search_cpu PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    OUTPUT_NAME ifs_search_cpu
    VERSION ${PROJECT_VERSION}
    SOVERSION 2)
install(TARGETS ifs_search_cpu
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(MSVC)
    target_compile_options(ifs_search_cpu PRIVATE /O2 /W4)
else()
    # ISA kernels use function-level target attributes and runtime CPUID. Do
    # not use -march=native: a release .so must start on older x86_64 hosts.
    target_compile_options(ifs_search_cpu PRIVATE -O3 -Wall -Wextra -Wpedantic)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ifs_search_cpu PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

if(IFS_SEARCH_BUILD_TESTS)
    add_executable(ifs_search_cpu_smoke tests/smoke.cpp)
    target_compile_features(ifs_search_cpu_smoke PRIVATE cxx_std_17)
    target_link_libraries(ifs_search_cpu_smoke PRIVATE ifs_search_cpu)
    if(MSVC)
        target_compile_options(ifs_search_cpu_smoke PRIVATE /UNDEBUG)
    else()
        target_compile_options(ifs_search_cpu_smoke PRIVATE -UNDEBUG)
    endif()
    add_test(NAME ifs_search_cpu_smoke COMMAND ifs_search_cpu_smoke)
endif()
