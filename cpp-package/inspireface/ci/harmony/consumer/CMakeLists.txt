cmake_minimum_required(VERSION 3.20)
project(HarmonySDKConsumer LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
add_library(InspireFace SHARED IMPORTED)
set_target_properties(InspireFace PROPERTIES
    IMPORTED_LOCATION "${SDK_DIR}/lib/libInspireFace.so"
    INTERFACE_INCLUDE_DIRECTORIES "${SDK_DIR}/include")
add_executable(c_consumer main.c)
add_executable(cpp_consumer main.cpp)
target_link_libraries(c_consumer PRIVATE InspireFace)
target_link_libraries(cpp_consumer PRIVATE InspireFace)
