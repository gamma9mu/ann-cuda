#if !defined(_POSIX_C_SOURCE) || (_POSIX_C_SOURCE < 200112L)
#define _POSIX_C_SOURCE 200112L
#endif

#include <math.h>
#include "ann.h"

static float sigmoid(float);

void
st_backprop(float *data, int count, float *expected,
         float *w_ih, float *theta_h, float *w_ho, float *theta_o,
         float rate)
{
    int i, j, k;
    float hid[HIDDEN_SIZE];
    float out[OUTPUT_SIZE];
    float err[OUTPUT_SIZE];
    float d_o[OUTPUT_SIZE];
    float d_h[HIDDEN_SIZE];

    float (*d)[INPUT_SIZE] = (float (*)[INPUT_SIZE]) data;
    float (*e)[OUTPUT_SIZE] = (float (*)[OUTPUT_SIZE]) expected;
    float (*wh)[INPUT_SIZE] = (float (*)[INPUT_SIZE]) w_ih;
    float (*wo)[HIDDEN_SIZE] = (float (*)[HIDDEN_SIZE]) w_ho;

    for (i = 0; i < count; ++i) {
        /* Feed forward through hidden layer */
        for (j = 0; j < HIDDEN_SIZE; ++j) {
            hid[j] = 0.0f;
            for (k = 0; k < INPUT_SIZE; ++k)
                hid[j] += d[i][k] * wh[j][k];
            hid[j] = sigmoid(hid[j] - theta_h[j]);

        }
        /* Feed forward through output layer */
        for (j = 0; j < OUTPUT_SIZE; ++j) {
            out[j] = 0.0f;
            for (k = 0; k < HIDDEN_SIZE; ++k)
                out[j] += hid[k] * wo[j][k];
            out[j] = sigmoid(out[j] - theta_o[j]);
        }

        /* Backprop O */
        for (j = 0; j < OUTPUT_SIZE; ++j) {
            err[j] = e[i][j] - out[j];
            d_o[j] = out[j] * (1 - out[j]) * err[j];
        }

        /* Backprop H */
        for (j = 0; j < HIDDEN_SIZE; ++j) {
            float grad = 0.0f;
            for (k = 0; k < OUTPUT_SIZE; ++k)
                grad = wo[k][j] * d_o[k];
            d_h[j] = hid[j] * (1 - hid[j]) * grad;
        }

        /* Apply updates */
        for (j = 0; j < HIDDEN_SIZE; ++j)
            for (k = 0; k < INPUT_SIZE; ++k)
                wh[j][k] += rate * d[i][k] * d_h[j];

        for (j = 0; j < OUTPUT_SIZE; ++j)
            for (k = 0; k < HIDDEN_SIZE; ++k)
                wo[j][k] += rate * hid[k] * d_o[j];
    }
}

float
st_evaluate(float *data, int count, float *expected,
         float *w_ih, float *theta_h, float *w_ho, float *theta_o)
{
    int i, j, k;
    float err = 0.0f;
    float hid[HIDDEN_SIZE];
    float out[OUTPUT_SIZE];

    float (*d)[INPUT_SIZE] = (float (*)[INPUT_SIZE]) data;
    float (*e)[OUTPUT_SIZE] = (float (*)[OUTPUT_SIZE]) expected;
    float (*wh)[INPUT_SIZE] = (float (*)[INPUT_SIZE]) w_ih;
    float (*wo)[HIDDEN_SIZE] = (float (*)[HIDDEN_SIZE]) w_ho;

    for (i = 0; i < count; ++i) {
        /* Feed forward through hidden layer */
        for (j = 0; j < HIDDEN_SIZE; ++j) {
            hid[j] = 0.0f;
            for (k = 0; k < INPUT_SIZE; ++k)
                hid[j] += d[i][k] * wh[j][k];
            hid[j] = sigmoid(hid[j] - theta_h[j]);

        }
        /* Feed forward through output layer */
        for (j = 0; j < OUTPUT_SIZE; ++j) {
            out[j] = 0.0f;
            for (k = 0; k < HIDDEN_SIZE; ++k)
                out[j] += hid[k] * wo[j][k];
            float e_i = e[i][j] - sigmoid(out[j] - theta_o[j]);
            err += e_i * e_i;
        }
    }
    return err / count;
}
static float
sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
