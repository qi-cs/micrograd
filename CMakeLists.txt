cmake_minimum_required(VERSION 3.6)
project(micrograd_backend)

# Find Python and CUDA
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
find_package(CUDA REQUIRED)

# Manually specify cuDNN paths
set(CUDNN_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
set(CUDNN_LIBRARY /usr/lib/x86_64-linux-gnu/libcudnn.so)

# Include directories
include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include
)

# Set C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

# Set CUDA flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")

# Set the output directory for the shared libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/micrograd)

# Add CPU library
add_library(micrograd_cpu SHARED
    backend/cpu/ops.cpp
)

# Link CPU library with Python
target_link_libraries(micrograd_cpu
    ${PYTHON_LIBRARIES}
)

# Add CUDA library
cuda_add_library(micrograd_cuda SHARED
    backend/cuda/ops.cu
)

# Link CUDA library with Python and CUDA runtime
target_link_libraries(micrograd_cuda
    ${PYTHON_LIBRARIES}
    ${CUDA_LIBRARIES}
)

# Add cuDNN library
cuda_add_library(micrograd_cudnn SHARED
    backend/cudnn/ops.cu
)

# Link cuDNN library with Python, CUDA, and cuDNN
target_link_libraries(micrograd_cudnn
    ${PYTHON_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDNN_LIBRARY}
)

# Set properties for the shared libraries
set_target_properties(micrograd_cpu PROPERTIES PREFIX "")
set_target_properties(micrograd_cuda PROPERTIES PREFIX "")
set_target_properties(micrograd_cudnn PROPERTIES PREFIX "")
