#ifndef BACKPROP_H__
#define BACKPROP_H__

#define INPUT_SIZE 4
#define OUTPUT_SIZE 3

#define HIDDEN_SIZE 8


typedef struct {
    float *d_data;
    float *d_expected;
    float *d_w_ih;
    float *d_theta_h;
    float *d_w_ho;
    float *d_theta_o;
    int count;
} d_ann_t;


void copy_in(d_ann_t*, float*, int, float*, float*, float*, float*, float*);
void copy_out(d_ann_t*, float*, float*);

void backprop_wrapper(d_ann_t*, float);
float evaluate_wrapper(d_ann_t*);
void run_wrapper(float *data, int count, float *expected,
        float *w_ih, float *theta_h, float *w_ho, float *theta_o,
        float *output);

#endif /* BACKPROP_H__ */

