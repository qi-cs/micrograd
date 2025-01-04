#ifndef CUDNN_OPS_H
#define CUDNN_OPS_H

void init_cudnn();
float cudnn_add(float a, float b);
float cudnn_multiply(float a, float b);
float cudnn_relu(float x);

#endif // CUDNN_OPS_H 