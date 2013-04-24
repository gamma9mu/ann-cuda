#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
#include "backprop.h"
}

__device__ inline float
sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void
backprop(float *data, int count,
        float *w_ih, float *w_ho,
        float *theta_h, float *theta_o,
        float *expected, float rate)
{
    /* Hidden layer weights */
    __shared__ float w_hid[HIDDEN_SIZE][INPUT_SIZE];
    __shared__ float th_h[HIDDEN_SIZE];

    /* Output layer weights */
    __shared__ float w_out[OUTPUT_SIZE][HIDDEN_SIZE];
    __shared__ float th_o[HIDDEN_SIZE];

    /* Layer output values */
    __shared__ float hid[HIDDEN_SIZE];
    __shared__ float out[OUTPUT_SIZE];
    __shared__ float err[OUTPUT_SIZE];
    __shared__ float delta[OUTPUT_SIZE];

    /* Input data */
    __shared__ float input[INPUT_SIZE];

    int tx = threadIdx.x;

    /* Load the hidden layer's theta values and weight matrix. */
    if (tx < HIDDEN_SIZE) {
        th_h[tx] = theta_h[tx];
        for (int i = 0; i < INPUT_SIZE; ++i)
            w_hid[tx][i] = w_ih[tx + (INPUT_SIZE * i)];
    }

    /* Load the output layer's theta values and weight matrix. */
    if (tx < OUTPUT_SIZE) {
        th_o[tx] = theta_o[tx];
        for (int i = 0; i < HIDDEN_SIZE; ++i)
            w_out[tx][i] = w_ho[tx + (OUTPUT_SIZE * i)];
    }

    /* Process each piece of input data. */
    for (int i = 0; i < count; ++i) {
        /* Load the data item. */
        if (tx < INPUT_SIZE)
            input[tx] = data[tx + (i * INPUT_SIZE)];

        __syncthreads();
        
        /* Propagate through the hidden layer. */
        if (tx < HIDDEN_SIZE) {
            hid[tx] = 0.0f;
            for (int j = 0; j < INPUT_SIZE; ++j)
                hid[tx] += input[j] * w_hid[tx][j];
            hid[tx] -= th_h[tx];
            hid[tx] = sigmoid(hid[tx]);
        }
        
        __syncthreads();

        /* Propagate through the output layer. */
        if (tx < OUTPUT_SIZE) {
            out[tx] = 0.0f;
            for (int j = 0; j < HIDDEN_SIZE; ++j)
                out[tx] += hid[j] * w_out[tx][j];
            out[tx] -= th_o[tx];
            out[tx] = sigmoid(out[tx]);
        }

        __syncthreads();

        /* Backpropagation starts here. */

        /* Calculate the error deltas for the output layer. */
        if (tx < OUTPUT_SIZE) {
            err[tx] = expected[tx + (OUTPUT_SIZE * i)] - out[tx];
            delta[tx] = out[tx] * (1 - out[tx]) * err[tx];
        }

        __syncthreads();

        /* Calculate the error deltas for the hidden layer and update the
         * weights. */
        if (tx < HIDDEN_SIZE) {
            float hdelta = 0.0f;
            for (int j = 0; j < OUTPUT_SIZE; ++j)
                hdelta += delta[j] * hid[tx];
            hdelta *= rate;
            for (int j = 0; j < INPUT_SIZE; ++j)
                w_hid[tx][j] += hdelta * input[j];
        }

        __syncthreads();

        /* Update the input weights of the output layer.  This section cannot be
         * combined above.  The changes would affect the hidden layer updates.
         */
        if (tx < OUTPUT_SIZE)
            for (int j = 0; j < HIDDEN_SIZE; ++j)
                w_out[tx][j] += rate * hid[j] * delta[tx];

        __syncthreads();
    }

    /* Copy the hidden layer's weight matrix out to global memory. */
    if (tx < HIDDEN_SIZE) {
        for (int i = 0; i < INPUT_SIZE; ++i)
             w_ih[tx + (INPUT_SIZE * i)] = w_hid[tx][i];
    }

    /* Copy the output layer's weight matrix out to global memory. */
    if (tx < OUTPUT_SIZE) {
        for (int i = 0; i < HIDDEN_SIZE; ++i)
             w_ho[tx + (OUTPUT_SIZE * i)] = w_out[tx][i];
    }
}
