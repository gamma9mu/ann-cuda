#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
#include "backprop.h"
}

__device__ inline float
sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* Performs backpropagation learning on a feed forward neural netowrk.
 * data         column-major, 2-d array of inputs (each of INPUT_SIZE length)
 * count        the total number of items in data
 * expected     column-major, 2-d array of expected output values (each of
 *              OUTPUT_SIZE length)
 * w_ih         column-major weight matrix for the hidden layer
 * theta_h      activation weights of the hidden layer
 * w_ho         column-major weight matrix for the output layer
 * theta_o      activation weights of the output layer
 * rate         learning rate of the ANN (a float between 0 and 1)
 *
 * The updated weight matrices will be copied over their previous values in
 * device global memory after each of the items in data has been processed.
 *
 * This kernel should be called with 1 block, having a single dimension of
 * threads, where the number of threads is the maximum of the number of hidden
 * layer neurons and the number of output layer neurons.
 */
__global__ void
backprop(float *data, int count, float *expected,
        float *w_ih, float *theta_h, float *w_ho, float *theta_o,
        float rate)
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

/* Evaluates an ANN's sum of squared errors.
 * data         column-major, 2-d array of inputs (each of INPUT_SIZE length)
 * count        the total number of items in data
 * expected     column-major, 2-d array of expected output values (each of
 *              OUTPUT_SIZE length)
 * w_ih         column-major weight matrix for the hidden layer
 * theta_h      activation weights of the hidden layer
 * w_ho         column-major weight matrix for the output layer
 * theta_o      activation weights of the output layer
 * sse          address in global memory to store the SSE into
 *
 * This kernel should be called with 1 block, having a single dimension of
 * threads, where the number of threads is the maximum of the number of hidden
 * layer neurons and the number of output layer neurons.
 */
__global__ void
evaluate(float *data, int count, float *expected,
        float *w_ih, float *theta_h, float *w_ho, float *theta_o,
        float *sse)
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
    __shared__ float errors[OUTPUT_SIZE];

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
        errors[tx] = 0.0f;
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

        /* Track each output neuron's squared errors. */
        if (tx < OUTPUT_SIZE) {
            float error = expected[tx + (OUTPUT_SIZE * i)] - out[tx];
            errors[tx] += error * error;
        }

        __syncthreads();
    }

    /* Sum individual output neuron's SSE and write to global memory. */
    if (tx == 0) {
        float errsum = 0.0f;
        for (int i = 0; i < OUTPUT_SIZE; ++i)
            errsum += errors[i];
        *sse = errsum;
    }
}

/* Runs an ANN on a series of inputs.
 * data         column-major, 2-d array of inputs (each of INPUT_SIZE length)
 * count        the total number of items in data
 * w_ih         column-major weight matrix for the hidden layer
 * theta_h      activation weights of the hidden layer
 * w_ho         column-major weight matrix for the output layer
 * theta_o      activation weights of the output layer
 * output       column-major, 2-d array to hold the output values (each of
 *              OUTPUT_SIZE length)
 *
 * This kernel should be called with 1 block, having a single dimension of
 * threads, where the number of threads is the maximum of the number of hidden
 * layer neurons and the number of output layer neurons.
 */
__global__ void
run(float *data, int count, float *expected,
        float *w_ih, float *theta_h, float *w_ho, float *theta_o,
        float *output)
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

        /* Propagate through the output layer and write the results to device
         * global memory. */
        if (tx < OUTPUT_SIZE) {
            out[tx] = 0.0f;
            for (int j = 0; j < HIDDEN_SIZE; ++j)
                out[tx] += hid[j] * w_out[tx][j];
            out[tx] -= th_o[tx];
            output[tx + (OUTPUT_SIZE * i)] = sigmoid(out[tx]);
        }

        __syncthreads();
    }
}
