#include "TinyML.h"

TinyML::TinyML(const float* weights1, const float* bias1, const float* weights2, const float* bias2)
    : _weights1(weights1), _bias1(bias1), _weights2(weights2), _bias2(bias2) {}

void TinyML::predict(float* input, float* output) {
    float hidden[4]; // adjust size based on model
    dense_forward(input, _weights1, _bias1, hidden, 3, 4);
    for (int i = 0; i < 4; i++) hidden[i] = relu(hidden[i]);
    dense_forward(hidden, _weights2, _bias2, output, 4, 2);
    softmax(output, 2);
}

void TinyML::dense_forward(const float* input, const float* weights, const float* bias, float* output, int in_size, int out_size) {
    for (int i = 0; i < out_size; i++) {
        output[i] = bias[i];
        for (int j = 0; j < in_size; j++) {
            output[i] += input[j] * weights[i * in_size + j];
        }
    }
}

float TinyML::relu(float x) {
    return x > 0 ? x : 0;
}

void TinyML::softmax(float* x, int len) {
    float max_val = x[0];
    for (int i = 1; i < len; i++) if (x[i] > max_val) max_val = x[i];

    float sum = 0.0;
    for (int i = 0; i < len; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < len; i++) {
        x[i] /= sum;
    }
}
