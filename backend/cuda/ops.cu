#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void add_kernel(float a, float b, float* result) {
    *result = a + b;
}

__global__ void multiply_kernel(float a, float b, float* result) {
    *result = a * b;
}

float cuda_add(float a, float b) {
    float* d_result;
    float h_result;
    std::cout << "CUDA Addition: " << a << " + " << b << std::endl;
    cudaMalloc(&d_result, sizeof(float));
    add_kernel<<<1, 1>>>(a, b, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

float cuda_multiply(float a, float b) {
    std::cout << "CUDA Multiplication: " << a << " * " << b << std::endl;
    float* d_result;
    float h_result;
    
    cudaMalloc(&d_result, sizeof(float));
    multiply_kernel<<<1, 1>>>(a, b, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

PYBIND11_MODULE(micrograd_cuda, m) {
    m.def("cuda_add", &cuda_add, "A function that adds two numbers using CUDA");
    m.def("cuda_multiply", &cuda_multiply, "A function that multiplies two numbers using CUDA");
}
