#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

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

/* Copies data to device global memory from host memory
 * data         column-major, 2-d array of inputs(each of INPUT_SIZE length
 * count        the total number of items in data
 * expected     column-major, 2-d array of expected output values (each of
 *              OUTPUT_SIZE length)
 * w_ih         column-major weight matrix for the hidden layer
 * theta_h      activation weights of the hidden layer
 * w_ho         column-major weight matrix for the output layer
 * theta_o      activation weights of the output layer
 * rate         learning rate of the ANN (a float between 0 and 1)
 *
 * This wrapper calls the backprop kernel with 1 block
 */

void backprop_wrapper(float *data, int count, float *expected, 
        float *w_ih, float *theta_h, float *w_ho, float *theta_o,
        float rate) {

    /* Determines the size for input mallocs */
    size_t input_size = (count * INPUT_SIZE) * sizeof(float);
    /* Determines the size for hiddent sector mallocs */
    size_t hidden_size = (count * HIDDEN_SIZE) * sizeof(float);
    /* Determines the size for output mallocs */
    size_t output_size = (count * OUTPUT_SIZE) * sizeof(float);
    
    /* Allocates memory for input data on the device */
    float* d_data;
    if(cudaSuccess != cudaMalloc(&d_data, input_size) )
        printf("Error allocating d_data for backprop\n");

    /* Allocates memory for expected output on the device */
    float* d_expected;
    if(cudaSuccess != cudaMalloc(&d_expected, output_size) )
        printf("Error allocating d_expected for backprop\n");

    /* Allocates memory for w_ih on the deivice */
    float* d_w_ih;
    if(cudaSuccess != cudaMalloc(&d_w_ih, hidden_size) )
        printf("Error allocating d_w_ih for backprop\n");

    /* Allocates memory for theta_h on the device */
    float* d_theta_h;
    if(cudaSuccess != cudaMalloc(&d_theta_h, hidden_size) )
        printf("Error allocating d_theta_h for backprop\n");

    /* Allocates memory for w_ho on the device */
    float* d_w_ho;
    if(cudaSuccess != cudaMalloc(&d_w_ho, output_size) )
        printf("Error allocating d_w_ho for backprop\n");

    /* Allocates memory for theta_o on the device */
    float* d_theta_o;
    if(cudaSuccess != cudaMalloc(&d_theta_o, output_size) )
        printf("Error allocating d_theta_o for backprop\n");

    /* Copies the variables from the host to the global memory
     * on the device
     */
    cudaMemcpy(d_data, data, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_expected, expected, output_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_ih, w_ih, hidden_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta_h, theta_h, hidden_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_ho, w_ho, output_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta_o, theta_o, output_size, cudaMemcpyHostToDevice);

    /* Determines the number of threads based on output and hidden sizes */
    int ThreadsPerBlock;
    if(OUTPUT_SIZE > HIDDEN_SIZE)
        ThreadsPerBlock = OUTPUT_SIZE;
    else
        ThreadsPerBlock = HIDDEN_SIZE;

    /* Executes the backprop kernel with the proper parameters */
    backprop<<<1, ThreadsPerBlock>>>(d_data, count, d_expected, d_w_ih,
            d_theta_h, d_w_ho, d_theta_o, rate);
    
    /* Check to see if any errors occurred during kernel call */
    if( cudaSuccess != cudaGetLastError() )
        printf("Error running the backprop kernel\n");

    /* Copies the output weights back to host memory */
    cudaMemcpy(w_ho, d_w_ho, output_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(w_ih, d_w_ih, hidden_size, cudaMemcpyDeviceToHost);

    /* Frees up the allocated memory on the device */
    cudaFree(d_data);
    cudaFree(d_expected);
    cudaFree(d_w_ih);
    cudaFree(d_theta_h);
    cudaFree(d_w_ho);
    cudaFree(d_theta_o);
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

/* Copies over data to device global memory for evaluate
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
 * Calls the evaluate kernel with one block, having a single dimension of
 * threads, where the number of threads is the maximum of the number of hidden
 * layer neurons and the number of output layer neurons.
 */
void evaluate_wrapper(float *data, int count, float *expected,
        float *w_ih, float *theta_h, float *w_ho, float *theta_o) {
    
    /*Determines the size for input mallocs*/
    size_t input_size = (count * INPUT_SIZE) * sizeof(float);
    /*Determines the size for hidden sector mallocs*/
    size_t hidden_size = (count * HIDDEN_SIZE) * sizeof(float);
    /*Determines the size for output mallocs*/
    size_t output_size = (count * OUTPUT_SIZE) * sizeof(float);

    /* Allocates the memory for data on the device */
    float *d_data;
    if( cudaSuccess != cudaMalloc(&d_data, input_size) )
        printf("Error allocating d_data for evaluate\n");

    /* Allocates the memory for exected on the device */
    float *d_expected;
    if( cudaSuccess != cudaMalloc(&d_expected, output_size) )
        printf("Error allocating expected for evaluate\n");

    /* Allocates the memory for w_ih on the device */
    float *d_w_ih;
    if( cudaSuccess != cudaMalloc(&d_w_ih, hidden_size) )
        printf("Error allocating w_ih for evaluate\n");

    /* Allocates the memory for theta_h on the device */
    float *d_theta_h;
    if( cudaSuccess != cudaMalloc(&d_theta_h, hidden_size) )
        printf("Error allocating theta_h for evaluate\n");
    
    /* Allocates the memory for w_ho on the device */
    float *d_w_ho;
    if( cudaSuccess != cudaMalloc(&d_w_ho, output_size) )
        printf("Error allocating w_ho for evaluate\n");

    /* Allocates the memory for theta_o on the device */
    float *d_theta_o;
    if( cudaSuccess != cudaMalloc(&d_theta_o, output_size) )
        printf("Error allocating theta_o for evaluate\n");

    /* Allocates the memory for SSE on the device */
    float *d_sse;
    if( cudaSuccess != cudaMalloc(&d_sse, input_size) )
        printf("Error allocating sse for evaluate\n");


    /* Copies all of the variables over to global memory on the device */
    cudaMemcpy(d_data, data, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_expected, expected, output_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_ih, w_ih, hidden_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta_h, theta_h, hidden_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_ho, w_ho, output_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta_o, theta_o, output_size, cudaMemcpyHostToDevice);

    /* Determines what the number of threads should be */
    int ThreadsPerBlock;
    if(OUTPUT_SIZE > HIDDEN_SIZE)
        ThreadsPerBlock = OUTPUT_SIZE;
    else
        ThreadsPerBlock = HIDDEN_SIZE;

    /* Runs the evaluate kernel with the proper parameters */
    evaluate<<<1, ThreadsPerBlock>>>(d_data, count, d_expected,
            d_w_ih, d_theta_h, d_w_ho, d_theta_o, d_sse);

    /* Perform simple error checking on kernel run */
    if( cudaSuccess != cudaGetLastError() )
        printf("Error while running the evaluate kernel");

    /* Copies the output weights back to host memory */
    cudaMemcpy(w_ih, d_w_ih, hidden_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(w_ho, d_w_ho, output_size, cudaMemcpyDeviceToHost);

    /* Frees up the allocated memory on the device */
    cudaFree(d_data);
    cudaFree(d_expected);
    cudaFree(d_w_ih);
    cudaFree(d_theta_h);
    cudaFree(d_w_ho);
    cudaFree(d_theta_o);
    cudaFree(d_sse);
    
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
