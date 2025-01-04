#include <pybind11/pybind11.h>
#include <iostream>
float cpu_add(float a, float b) {
    std::cout << "CPU Addition: " << a << " + " << b << std::endl;
    return a + b;
}

float cpu_multiply(float a, float b) {
    std::cout << "CPU Multiplication: " << a << " * " << b << std::endl;
    return a * b;
}

PYBIND11_MODULE(micrograd_cpu, m) {
    m.def("cpu_add", &cpu_add, "A function that adds two numbers");
    m.def("cpu_multiply", &cpu_multiply, "A function that multiplies two numbers");
}
