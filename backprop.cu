#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

extern "C" {
#include "ann.h"
}

static void cuda_perror(cudaError_t err, const char * file, int line) {
    fprintf(stderr, "%s:%d:CUDA Error: %s\n", file, line,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

#define chk(err) if (err != cudaSuccess) { cuda_perror(err, __FILE__, __LINE__); }

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
    __shared__ float w_hid[INPUT_SIZE][HIDDEN_SIZE];
    __shared__ float th_h[HIDDEN_SIZE];

    /* Output layer weights */
    __shared__ float w_out[HIDDEN_SIZE][OUTPUT_SIZE];
    __shared__ float th_o[HIDDEN_SIZE];

    /* Layer output values */
    __shared__ float hid[HIDDEN_SIZE];
    __shared__ float out[OUTPUT_SIZE];
    __shared__ float err[OUTPUT_SIZE];
    __shared__ float delta[OUTPUT_SIZE];

    /* Input data */
    __shared__ float input[INPUT_SIZE];
    __shared__ float expec[OUTPUT_SIZE];

    int tx = threadIdx.x;

    /* Load the hidden layer's theta values and weight matrix. */
    if (tx < HIDDEN_SIZE) {
        th_h[tx] = theta_h[tx];
        for (int i = 0; i < INPUT_SIZE; ++i) {
            w_hid[i][tx] = w_ih[(tx * INPUT_SIZE) + i];
        }
    }

    /* Load the output layer's theta values and weight matrix. */
    if (tx < OUTPUT_SIZE) {
        th_o[tx] = theta_o[tx];
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            w_out[i][tx] = w_ho[(tx * HIDDEN_SIZE) + i];
        }
    }

    /* Process each piece of input data. */
    for (int i = 0; i < count; ++i) {
        /* Load the data item. */
        if (tx < INPUT_SIZE) {
            input[tx] = data[tx + (i * INPUT_SIZE)];
        }
        if (tx < OUTPUT_SIZE) {
            expec[tx] = expected[tx + (i * OUTPUT_SIZE)];
        }

        __syncthreads();

        /* Propagate through the hidden layer. */
        if (tx < HIDDEN_SIZE) {
            hid[tx] = 0.0f;
            for (int j = 0; j < INPUT_SIZE; ++j) {
                hid[tx] += input[j] * w_hid[j][tx];
            }
            hid[tx] -= th_h[tx];
            hid[tx] = sigmoid(hid[tx]);
        }

        __syncthreads();

        /* Propagate through the output layer. */
        if (tx < OUTPUT_SIZE) {
            out[tx] = 0.0f;
            for (int j = 0; j < HIDDEN_SIZE; ++j) {
                out[tx] += hid[j] * w_out[j][tx];
            }
            out[tx] -= th_o[tx];
            out[tx] = sigmoid(out[tx]);
        }

        __syncthreads();

        /* Backpropagation starts here. */

        /* Calculate the error deltas for the output layer. */
        if (tx < OUTPUT_SIZE) {
            err[tx] = expec[tx] - out[tx];
            delta[tx] = out[tx] * (1 - out[tx]) * err[tx];
        }

        __syncthreads();

        /* Calculate the error deltas for the hidden layer and update the
         * weights. */
        if (tx < HIDDEN_SIZE) {
            float d_h = 0.0f;
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                d_h += delta[j] * w_out[tx][j];
            }
            d_h *= hid[tx] * (1 - hid[tx]);
            for (int j = 0; j < INPUT_SIZE; ++j) {
                w_hid[j][tx] += rate * input[j] * d_h;
            }
        }

        __syncthreads();

        /* Update the input weights of the output layer.  This section cannot be
         * combined above.  The changes would affect the hidden layer updates.
         */
        if (tx < OUTPUT_SIZE)
            for (int j = 0; j < HIDDEN_SIZE; ++j) {
                w_out[j][tx] += rate * hid[j] * delta[tx];
            }

        __syncthreads();
    }

    /* Copy the hidden layer's weight matrix out to global memory. */
    if (tx < HIDDEN_SIZE) {
        for (int i = 0; i < INPUT_SIZE; ++i) {
            w_ih[(tx * INPUT_SIZE) + i] = w_hid[i][tx];
        }
    }

    /* Copy the output layer's weight matrix out to global memory. */
    if (tx < OUTPUT_SIZE) {
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            w_ho[(tx * HIDDEN_SIZE) + i] = w_out[i][tx];
        }
    }
}


/* Evaluates an ANN's average sum of squared errors.
 * data         column-major, 2-d array of inputs (each of INPUT_SIZE length)
 * count        the total number of items in data
 * expected     column-major, 2-d array of expected output values (each of
 *              OUTPUT_SIZE length)
 * w_ih         column-major weight matrix for the hidden layer
 * theta_h      activation weights of the hidden layer
 * w_ho         column-major weight matrix for the output layer
 * theta_o      activation weights of the output layer
 * sse          address in global memory to store the average SSE into
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
    __shared__ float w_hid[INPUT_SIZE][HIDDEN_SIZE];
    __shared__ float th_h[HIDDEN_SIZE];

    /* Output layer weights */
    __shared__ float w_out[HIDDEN_SIZE][OUTPUT_SIZE];
    __shared__ float th_o[HIDDEN_SIZE];

    /* Layer output values */
    __shared__ float hid[HIDDEN_SIZE];
    __shared__ float out[OUTPUT_SIZE];

    /* Input data */
    __shared__ float input[INPUT_SIZE];
    __shared__ float expec[OUTPUT_SIZE];
    __shared__ float errors[OUTPUT_SIZE];

    int tx = threadIdx.x;

    /* Load the hidden layer's theta values and weight matrix. */
    if (tx < HIDDEN_SIZE) {
        th_h[tx] = theta_h[tx];
        for (int i = 0; i < INPUT_SIZE; ++i) {
            w_hid[i][tx] = w_ih[(tx * INPUT_SIZE) + i];
        }
    }

    /* Load the output layer's theta values and weight matrix. */
    if (tx < OUTPUT_SIZE) {
        th_o[tx] = theta_o[tx];
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            w_out[i][tx] = w_ho[(tx * HIDDEN_SIZE) + i];
        }
    }

    /* Process each piece of input data. */
    for (int i = 0; i < count; ++i) {
        /* Load the data item. */
        if (tx < INPUT_SIZE) {
            input[tx] = data[tx + (i * INPUT_SIZE)];
        }
        if (tx < OUTPUT_SIZE) {
            expec[tx] = expected[tx + (i * OUTPUT_SIZE)];
        }

        __syncthreads();

        /* Propagate through the hidden layer. */
        if (tx < HIDDEN_SIZE) {
            hid[tx] = 0.0f;
            for (int j = 0; j < INPUT_SIZE; ++j) {
                hid[tx] += input[j] * w_hid[j][tx];
            }
            hid[tx] -= th_h[tx];
            hid[tx] = sigmoid(hid[tx]);
        }

        __syncthreads();

        /* Propagate through the output layer. */
        if (tx < OUTPUT_SIZE) {
            out[tx] = 0.0f;
            for (int j = 0; j < HIDDEN_SIZE; ++j) {
                out[tx] += hid[j] * w_out[j][tx];
            }
            out[tx] -= th_o[tx];
            out[tx] = sigmoid(out[tx]);
        }

        __syncthreads();

        /* Track each output neuron's squared errors. */
        if (tx < OUTPUT_SIZE) {
            float error = expec[tx] - out[tx];
            errors[tx] += error * error;
        }

        __syncthreads();
    }

    /* Sum individual output neuron's SSE and write to global memory. */
    if (tx == 0) {
        float errsum = 0.0f;
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            errsum += errors[i];
        }
        *sse = errsum / count;
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
    __shared__ float w_hid[INPUT_SIZE][HIDDEN_SIZE];
    __shared__ float th_h[HIDDEN_SIZE];

    /* Output layer weights */
    __shared__ float w_out[HIDDEN_SIZE][OUTPUT_SIZE];
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
        for (int i = 0; i < INPUT_SIZE; ++i) {
            w_hid[i][tx] = w_ih[(tx * INPUT_SIZE) + i];
        }
    }

    /* Load the output layer's theta values and weight matrix. */
    if (tx < OUTPUT_SIZE) {
        th_o[tx] = theta_o[tx];
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            w_out[i][tx] = w_ho[(tx * HIDDEN_SIZE) + i];
        }
    }

    /* Process each piece of input data. */
    for (int i = 0; i < count; ++i) {
        /* Load the data item. */
        if (tx < INPUT_SIZE) {
            input[tx] = data[tx + (i * INPUT_SIZE)];
        }

        __syncthreads();

        /* Propagate through the hidden layer. */
        if (tx < HIDDEN_SIZE) {
            hid[tx] = 0.0f;
            for (int j = 0; j < INPUT_SIZE; ++j) {
                hid[tx] += input[j] * w_hid[j][tx];
            }
            hid[tx] -= th_h[tx];
            hid[tx] = sigmoid(hid[tx]);
        }

        __syncthreads();

        /* Propagate through the output layer. */
        if (tx < OUTPUT_SIZE) {
            out[tx] = 0.0f;
            for (int j = 0; j < HIDDEN_SIZE; ++j) {
                out[tx] += hid[j] * w_out[j][tx];
            }
            out[tx] -= th_o[tx];
            out[tx] = sigmoid(out[tx]);
            output[tx + (i * OUTPUT_SIZE)] = sigmoid(out[tx]);
        }

        __syncthreads();
    }
}

/* Copy the weight matrices and activation weight vectors of an ANN into device
 * memory.
 */
extern "C" void
copy_in_ann(d_ann_t *d_ann, float *w_ih, float *theta_h, float *w_ho,
        float *theta_o) {
    /*Determines the size for hidden sector mallocs*/
    size_t th_h_sz = HIDDEN_SIZE * sizeof(float);
    /* Determines the size for theta_o */
    size_t th_o_sz = OUTPUT_SIZE * sizeof(float);
    /* Determines the size of input weight */
    size_t w_ih_sz = (HIDDEN_SIZE * INPUT_SIZE) * sizeof(float);
    /* Determines the size of output weight */
    size_t w_ho_sz = (HIDDEN_SIZE * OUTPUT_SIZE) * sizeof(float);

    chk(cudaMalloc(&d_ann->d_w_ih, w_ih_sz));
    chk(cudaMalloc(&d_ann->d_theta_h, th_h_sz));
    chk(cudaMalloc(&d_ann->d_w_ho, w_ho_sz));
    chk(cudaMalloc(&d_ann->d_theta_o, th_o_sz));

    /* Copies the variables from the host to the global memory
     * on the device
     */
    chk(cudaMemcpy(d_ann->d_theta_h, theta_h, th_h_sz, cudaMemcpyHostToDevice));
    chk(cudaMemcpy(d_ann->d_theta_o, theta_o, th_o_sz, cudaMemcpyHostToDevice));
    chk(cudaMemcpy(d_ann->d_w_ih, w_ih, w_ih_sz, cudaMemcpyHostToDevice));
    chk(cudaMemcpy(d_ann->d_w_ho, w_ho, w_ho_sz, cudaMemcpyHostToDevice));
}

/* Copy a dataset into device memory.
 */
extern "C" void
copy_in_data(d_data_t *d_data, float *data, int count, float *expected,
        int copy_expected) {
    /*Determines the size for input mallocs*/
    size_t input_sz = (count * INPUT_SIZE) * sizeof(float);
    /*Determines the size for output mallocs*/
    size_t out_sz = (count * OUTPUT_SIZE) * sizeof(float);

    d_data->count = count;

    chk(cudaMalloc(&d_data->d_data, input_sz));
    chk(cudaMalloc(&d_data->d_expected, out_sz));

    chk(cudaMemcpy(d_data->d_data, data, input_sz, cudaMemcpyHostToDevice));
    if (copy_expected) {
        chk(cudaMemcpy(d_data->d_expected, expected, out_sz,
                    cudaMemcpyHostToDevice));
    }
}

/* Copy the ANN weights into host memory.
 * data         a dataset to process
 * w_ih         a section of memory to store the hidden layer weight matrix
 * w_ho         a section of memory to store the output layer weight matrix
 */
extern "C" void
copy_out_ann(d_ann_t *d_ann, float *w_ih, float *w_ho) {
    /* Determines the size of input weight */
    size_t w_ih_size = (HIDDEN_SIZE * INPUT_SIZE) * sizeof(float);
    /* Determines the size of output weight */
    size_t w_ho_size = (HIDDEN_SIZE * OUTPUT_SIZE) * sizeof(float);

    chk(cudaMemcpy(w_ih, d_ann->d_w_ih, w_ih_size, cudaMemcpyDeviceToHost));
    chk(cudaMemcpy(w_ho, d_ann->d_w_ho, w_ho_size, cudaMemcpyDeviceToHost));

    chk(cudaFree(d_ann->d_w_ih));
    chk(cudaFree(d_ann->d_theta_h));
    chk(cudaFree(d_ann->d_w_ho));
    chk(cudaFree(d_ann->d_theta_o));
}

/* Copy the output values of a dataset into host memory.
 * data         a dataset to process
 * output       a section of memory to store the output values
 */
extern "C" void
copy_out_data(d_data_t *d_data, float *output) {
    size_t out_sz = (d_data->count * OUTPUT_SIZE) * sizeof(float);

    chk(cudaMemcpy(output, d_data->d_expected, out_sz, cudaMemcpyDeviceToHost));

    chk(cudaFree(d_data->d_data));
    chk(cudaFree(d_data->d_expected));
}

/* Copies data to device global memory from host memory
 * ann          an ANN
 * data         a dataset to process
 * rate         learning rate of the ANN (a float between 0 and 1)
 *
 * This wrapper calls the backprop kernel with 1 block.
 */

extern "C" void backprop_wrapper(d_ann_t *ann, d_data_t *data, float rate) {

    /* Determines the number of threads based on output and hidden sizes */
    int ThreadsPerBlock = (OUTPUT_SIZE>HIDDEN_SIZE) ? OUTPUT_SIZE : HIDDEN_SIZE;
    chk(cudaThreadSynchronize());
    backprop<<<1, ThreadsPerBlock>>>(data->d_data, data->count,
            data->d_expected, ann->d_w_ih, ann->d_theta_h, ann->d_w_ho,
            ann->d_theta_o, rate);

    if( cudaSuccess != cudaGetLastError() ) {
        printf("Error running the backprop kernel\n");
        exit(cudaGetLastError());
    }
    /* chk(cudaThreadSynchronize()); */
}

/* Copies over data to device global memory for evaluate
 * ann          an ANN
 * data         a dataset to process
 *
 * Calls the evaluate kernel with one block, having a single dimension of
 * threads, where the number of threads is the maximum of the number of hidden
 * layer neurons and the number of output layer neurons.
 *
 * Returns the sum of squared errors after evaluation.
 */
extern "C" float evaluate_wrapper(d_ann_t *ann, d_data_t *data) {
    float sse;
    float *d_sse;
    int ThreadsPerBlock = (OUTPUT_SIZE>HIDDEN_SIZE) ? OUTPUT_SIZE : HIDDEN_SIZE;

    chk(cudaMalloc(&d_sse, sizeof(float)));
    evaluate<<<1, ThreadsPerBlock>>>(data->d_data, data->count,
            data->d_expected, ann->d_w_ih, ann->d_theta_h, ann->d_w_ho,
            ann->d_theta_o, d_sse);

    if( cudaSuccess != cudaGetLastError() ) {
        printf("Error while running the evaluate kernel\n");
        exit(cudaGetLastError());
    }

    chk(cudaMemcpy(&sse, d_sse, sizeof(float), cudaMemcpyDeviceToHost));
    chk(cudaFree(d_sse));

    return sse;
}

/* Copies data over to device global memory for the run kernel
 * ann          an ANN
 * data         a dataset to process, the output array will be overwritten
 *
 * This calls the run kernel with 1 block, having a single dimension of
 * threads, where the number of threads is the maximum of the number of hidden
 * layer neurons and the number of output layer neurons.
 */

extern "C" void run_wrapper(d_ann_t *ann, d_data_t *data) {
    int ThreadsPerBlock = (OUTPUT_SIZE>HIDDEN_SIZE) ? OUTPUT_SIZE : HIDDEN_SIZE;
    run<<<1, ThreadsPerBlock>>>(data->d_data, data->count, data->d_expected,
            ann->d_w_ih, ann->d_theta_h, ann->d_w_ho, ann->d_theta_o,
            data->d_expected);

    /* Checks to see if an error occurred while executing run */
    if( cudaSuccess != cudaGetLastError() ) {
        printf("Error while executing run kernel\n");
        exit(cudaGetLastError());
    }
}
