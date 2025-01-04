#include <pybind11/pybind11.h>
#include <cudnn.h>
#include <cuda_runtime.h>

cudnnHandle_t cudnn;

void init_cudnn() {
    cudnnCreate(&cudnn);
}

float cudnn_add(float a, float b) {
    float result;
    cudnnTensorDescriptor_t aDesc, bDesc, resultDesc;
    float alpha = 1.0f, beta = 0.0f;

    // Create tensor descriptors
    cudnnCreateTensorDescriptor(&aDesc);
    cudnnCreateTensorDescriptor(&bDesc);
    cudnnCreateTensorDescriptor(&resultDesc);

    // Set tensor descriptor for a single value
    cudnnSetTensor4dDescriptor(aDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);
    cudnnSetTensor4dDescriptor(bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);
    cudnnSetTensor4dDescriptor(resultDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);

    // Perform the addition operation
    cudnnAddTensor(cudnn, &alpha, aDesc, &a, &beta, resultDesc, &result);

    // Destroy descriptors
    cudnnDestroyTensorDescriptor(aDesc);
    cudnnDestroyTensorDescriptor(bDesc);
    cudnnDestroyTensorDescriptor(resultDesc);

    return result;
}

float cudnn_multiply(float a, float b) {
    float result;
    cudnnTensorDescriptor_t aDesc, bDesc, resultDesc;
    cudnnOpTensorDescriptor_t opDesc;
    float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;

    // Create tensor descriptors
    cudnnCreateTensorDescriptor(&aDesc);
    cudnnCreateTensorDescriptor(&bDesc);
    cudnnCreateTensorDescriptor(&resultDesc);

    // Set tensor descriptor for a single value
    cudnnSetTensor4dDescriptor(aDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);
    cudnnSetTensor4dDescriptor(bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);
    cudnnSetTensor4dDescriptor(resultDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);

    // Create and set operation descriptor
    cudnnCreateOpTensorDescriptor(&opDesc);
    cudnnSetOpTensorDescriptor(opDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);

    // Perform the multiplication operation
    cudnnOpTensor(cudnn, opDesc, &alpha1, aDesc, &a, &alpha2, bDesc, &b, &beta, resultDesc, &result);

    // Destroy descriptors
    cudnnDestroyOpTensorDescriptor(opDesc);
    cudnnDestroyTensorDescriptor(aDesc);
    cudnnDestroyTensorDescriptor(bDesc);
    cudnnDestroyTensorDescriptor(resultDesc);

    return result;
}

float cudnn_relu(float x) {
    float result;
    cudnnActivationDescriptor_t activationDesc;
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    float alpha = 1.0f, beta = 0.0f;

    // Create tensor descriptors
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);

    // Set tensor descriptor for a single value
    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);

    // Create and set activation descriptor
    cudnnCreateActivationDescriptor(&activationDesc);
    cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);

    // Perform the ReLU operation
    cudnnActivationForward(cudnn, activationDesc, &alpha, inputDesc, &x, &beta, outputDesc, &result);

    // Destroy descriptors
    cudnnDestroyActivationDescriptor(activationDesc);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);

    return result;
}

PYBIND11_MODULE(micrograd_cudnn, m) {
    m.def("cudnn_add", &cudnn_add, "A function that adds two numbers using cuDNN");
    m.def("cudnn_multiply", &cudnn_multiply, "A function that multiplies two numbers using cuDNN");
    m.def("cudnn_relu", &cudnn_relu, "A function that applies ReLU using cuDNN");
} 