# FindTensorRT.cmake - Simple Version
# Contains basic functionality for finding TensorRT libraries

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Find TensorRT header files
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES include)

# Find TensorRT libraries
find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64)

find_library(TENSORRT_LIBRARY_RUNTIME nvinfer_plugin
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64)

# Find CUDA runtime library
find_library(CUDA_RUNTIME_LIBRARY cudart
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib64 lib lib64/stubs lib/stubs)

# Set result variables, can be used in projects that include this module
set(ISF_TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
set(ISF_TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_RUNTIME} ${CUDA_RUNTIME_LIBRARY})

# Output status messages
message(STATUS "Found TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "Found TensorRT libraries: ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_RUNTIME}")
message(STATUS "Found CUDA runtime library: ${CUDA_RUNTIME_LIBRARY}")