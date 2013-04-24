#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C" {
#include "backprop.h"
}

__global__ void
backprop(float *data, int size, int count,
        float *w_ih, float *w_ho,
        float *expected)
{

    __shared__ float w_out[OUTPUT_SIZE][HIDDEN_SIZE];
    __shared__ float w_hid[HIDDEN_SIZE][INPUT_SIZE];
    __shared__ float out[OUTPUT_SIZE];

    int tx = threadIdx.x;

}
