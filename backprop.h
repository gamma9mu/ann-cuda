#ifndef BACKPROP_H__
#define BACKPROP_H__

#define INPUT_SIZE 4
#define OUTPUT_SIZE 3

#define HIDDEN_SIZE 20

#endif 

void backprop_wrapper(float *data, int count, float *expected,
        float *w_ih, float *theta_h, float *w_ho, float *theta_o,
        float rate);

void evaluate_wrapper(float *data, int count, float *expected,
        float *w_ih, float *theta_h, float *w_ho, float *theta_o);

void run_wrapper(float *data, int count, float *expected,
        float *w_ih, float *theta_h, float *w_ho, float *theta_o,
        float *output);

/* BACKPROP_H__ */

