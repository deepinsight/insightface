cmake_minimum_required(VERSION 3.22)
project(insightface_search_native VERSION 2.0.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(IFS_SEARCH_BUILD_CPU "Build the CPU exact-search library" ON)
option(IFS_SEARCH_BUILD_CUDA "Build the CUDA exact-search library" OFF)
option(IFS_SEARCH_BUILD_TESTS "Build native smoke tests" ON)

if(IFS_SEARCH_BUILD_TESTS)
    enable_testing()
endif()

if(IFS_SEARCH_BUILD_CPU)
    add_subdirectory(cpu)
endif()

if(IFS_SEARCH_BUILD_CUDA)
    enable_language(CUDA)
    add_subdirectory(cuda)
endif()

install(FILES include/ifs_search.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
