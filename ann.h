#ifndef ANN_H__
#define ANN_H__

#define INPUT_SIZE 4
#define HIDDEN_SIZE 8
#define OUTPUT_SIZE 3

/* Device pointer grouping for ANN */
typedef struct {
    float *d_w_ih;
    float *d_theta_h;
    float *d_w_ho;
    float *d_theta_o;
} d_ann_t;

/* Device pointer struct for data */
typedef struct {
    float *d_data;
    float *d_expected;
    int count;
} d_data_t;

/* CUDA version functions */
void copy_in_ann(d_ann_t*, float*, float*, float*, float*);
void copy_in_data(d_data_t*, float*, int, float*, int);
void copy_out_ann(d_ann_t*, float*, float*);
void copy_out_data(d_data_t*, float*);
void backprop_wrapper(d_ann_t*, d_data_t*, float);
float evaluate_wrapper(d_ann_t*, d_data_t*);
void run_wrapper(d_ann_t*, d_data_t*);

/* Single-threaded version functions */
void st_backprop(float*, int, float*, float*, float*, float*, float*, float);
float st_evaluate(float*, int, float*, float*, float*, float*, float*);

#endif /* ANN_H__ */

