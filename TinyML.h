#ifndef TINYML_H
#define TINYML_H

#include <Arduino.h>

class TinyML {
public:
    TinyML(const float* weights1, const float* bias1, const float* weights2, const float* bias2);
    void predict(float* input, float* output);

private:
    const float* _weights1;
    const float* _bias1;
    const float* _weights2;
    const float* _bias2;

    void dense_forward(const float* input, const float* weights, const float* bias, float* output, int in_size, int out_size);
    float relu(float x);
    void softmax(float* x, int len);
};

#endif
