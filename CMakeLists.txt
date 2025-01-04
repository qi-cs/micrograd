cmake_minimum_required(VERSION 3.6)
project(micrograd_backend)

# Find Python and CUDA
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include
)

# Set C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

# Set CUDA flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")

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

# Set properties for the shared libraries
set_target_properties(micrograd_cpu PROPERTIES PREFIX "")
set_target_properties(micrograd_cuda PROPERTIES PREFIX "")
