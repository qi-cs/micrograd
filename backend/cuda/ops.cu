#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void add_kernel(float a, float b, float* result) {
    *result = a + b;
}

__global__ void multiply_scalar_kernel(float a, float b, float* result) {
    *result = a * b;
}

__global__ void multiply_vector_kernel(float* a, float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * b[idx];
    }
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
    multiply_scalar_kernel<<<1, 1>>>(a, b, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return h_result;
}

// cpu support function overloading with same name, different arguments
// but cuda does not support this
// so we need to use different names
std::vector<float> cuda_vec_multiply(const std::vector<float>& a, const std::vector<float>& b, const std::vector<int>& a_shape, const std::vector<int>& b_shape) {
    // Allocate device memory and copy data
    float *d_a, *d_b, *d_result;
    int size = a.size();
    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_b, size * sizeof(float));
    cudaMalloc(&d_result, size * sizeof(float));
    cudaMemcpy(d_a, a.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    multiply_vector_kernel<<<numBlocks, blockSize>>>(d_a, d_b, d_result, size);

    // Copy result back to host
    std::vector<float> result(size);
    cudaMemcpy(result.data(), d_result, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return result;
}

PYBIND11_MODULE(micrograd_cuda, m) {
    m.def("cuda_add", &cuda_add, "A function that adds two numbers using CUDA");
    m.def("cuda_multiply", &cuda_multiply, "A function that multiplies two scalar numbers using CUDA");
    m.def("cuda_vector_multiply", &cuda_vec_multiply, "A function that multiplies two vectors element-wise using CUDA");
}