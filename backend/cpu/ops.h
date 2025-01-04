#ifndef CPU_OPS_H
#define CPU_OPS_H

float cpu_add(float a, float b);
float cpu_multiply(float a, float b);
std::vector<float> cpu_multiply(const std::vector<float>& a, const std::vector<float>& b, const std::vector<int>& a_shape, const std::vector<int>& b_shape);
#endif // CPU_OPS_H
