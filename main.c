#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "data.h"
#include "backprop.h"
#include "st.h"


/* Learning Rate */
#define RATE 0.2

#define SSE_MAX 0.01

static void run_cuda(float*, int, float*, float*, float*, float*, float*);
static void run_st(float*, int, float*, float*, float*, float*, float*);
static float randw ( int numPrevious );
static void init_weights(float*, size_t, size_t);
static void init_theta(float*, size_t);
static void dumpmat(float*, size_t, size_t);
static void die(const char *);

int
main(int argc, char *argv[])
{
    size_t weight_input_size, weight_output_size;
    float *data, *expected, *w_ih, *theta_h, *w_ho, *theta_o;
    float *aw_ih, *aw_ho;
    int count;

    (void) argc; (void) argv;

    srand(time(NULL));

    /* Reads the data and stores it to the memory space.
     * Also returns count for later use.*/
    count = readdata( &data );
    if (readexpected(&expected) != count) {
        fprintf(stderr, "Training output values differ in number from inputs.\n");
        exit(EXIT_FAILURE);
    }

    /* Computes the sizes to allocate. */
    weight_input_size = (HIDDEN_SIZE * INPUT_SIZE) * sizeof(float);
    weight_output_size = (HIDDEN_SIZE * OUTPUT_SIZE) * sizeof(float);

    /* Allocates the memory for the different floats */
    if ((w_ih = malloc(weight_input_size)) == NULL) die("main: malloc");
    if ((aw_ih = malloc(weight_input_size)) == NULL) die("main: malloc");
    if ((w_ho = malloc(weight_output_size)) == NULL) die("main: malloc");
    if ((aw_ho = malloc(weight_output_size)) == NULL) die("main: malloc");
    if ((theta_h = malloc(HIDDEN_SIZE * sizeof(float))) == NULL)
        die("main: malloc");
    if ((theta_o = malloc(OUTPUT_SIZE * sizeof(float))) == NULL)
        die("main: malloc");

    init_weights(w_ih, INPUT_SIZE, HIDDEN_SIZE);
    init_weights(w_ho, HIDDEN_SIZE, OUTPUT_SIZE);
    init_theta(theta_h, HIDDEN_SIZE);
    init_theta(theta_o, OUTPUT_SIZE);

    memcpy(aw_ih, w_ih, weight_input_size);
    memcpy(aw_ho, w_ho, weight_output_size);

    dumpmat(w_ih, INPUT_SIZE, HIDDEN_SIZE);

    run_cuda(data, count, expected, w_ih, theta_h, w_ho, theta_o);
    run_st(data, count, expected, aw_ih, theta_h, aw_ho, theta_o);

    free(expected);
    free(data);

    return EXIT_SUCCESS;
}

static void
run_cuda(float *data, int count, float *expected,
         float *w_ih, float *theta_h, float *w_ho, float *theta_o) {
    float sse, sse_i;
    long int gen = 0;
    unsigned int growing = 0, stagnant = 0;

    d_ann_t ann;
    copy_in(&ann, data, count, expected, w_ih, theta_h, w_ho, theta_o);

    printf("*** Running CUDA ANN\n");
    printf(" Target SSE: %5.3f\n", SSE_MAX);
    sse = evaluate_wrapper(&ann);
    printf("Initial SSE: %5.3f\n", sse);

    while (sse > SSE_MAX && growing < 100 && stagnant < 250) {
        backprop_wrapper(&ann, RATE);
        sse_i = evaluate_wrapper(&ann);
        if (fabs(sse - sse_i) < 0.001) {
            ++stagnant;
            growing = 0;
        } else if (sse_i > sse) {
            ++growing;
            stagnant = 0;
        } else {
            growing = 0;
            stagnant = 0;
        }
        sse = sse_i;
        printf("Current SSE: %5.3f\tIteration: %ld [s: %d; g: %d]\t\r", sse,
                gen, stagnant, growing);
        ++gen;
    }
    fputs("\n\n", stdout);
    copy_out(&ann, w_ih, w_ho);
}

static void
run_st(float *data, int count, float *expected,
         float *w_ih, float *theta_h, float *w_ho, float *theta_o) {
    float sse, sse_i;
    long int gen;
    unsigned int growing = 0, stagnant = 0;

    printf("*** Running Single-Threaded ANN\n");
    printf(" Target SSE: %5.3f\n", SSE_MAX);
    sse = st_evaluate(data, count, expected, w_ih, theta_h, w_ho, theta_o);
    printf("Initial SSE: %5.3f\n", sse);

    while (sse > SSE_MAX && growing < 100 && stagnant < 250) {
        st_backprop(data, count, expected, w_ih, theta_h,
                w_ho, theta_o, RATE);
        sse_i = st_evaluate(data, count, expected, w_ih, theta_h,
                w_ho, theta_o);
        if (fabs(sse - sse_i) < 0.001) {
            ++stagnant;
            growing = 0;
        } else if (sse_i > sse) {
            ++growing;
            stagnant = 0;
        } else {
            growing = 0;
            stagnant = 0;
        }
        sse = sse_i;
        printf("Current SSE: %5.3f\tIteration: %ld [s: %d; g: %d]\t\r", sse,
                gen, stagnant, growing);
        ++gen;
    }
    fputs("\n\n", stdout);
}

static float
randw(int numPrevious)
{
    float weight = ((float)rand())/((float)RAND_MAX);
    weight *= numPrevious;
    weight -= numPrevious / 2.0f;

    return weight;
}

static void
init_weights(float *matrix, size_t dim_x, size_t dim_y) {
    size_t x, y;
    for (x = 0; x < dim_x; ++x)
        for (y = 0; y < dim_y; ++y)
            matrix[(y * dim_x) + x] = randw(dim_x);
}

static void
dumpmat(float *matrix, size_t dim_x, size_t dim_y) {
    size_t x, y;
    for (x = 0; x < dim_x; ++x) {
        for (y = 0; y < dim_y; ++y) {
            printf("%9.6f ", matrix[(y * dim_x) + x]);
        }
        fputc('\n', stdout);
    }
    fputc('\n', stdout);

}

static void
init_theta(float *vector, size_t length) {
    size_t i;
    for (i = 0; i < length; ++i)
        vector[i] = rand() / (float) RAND_MAX;
}

static void
die(const char *prefix) {
    perror(prefix);
    exit(EXIT_FAILURE);
}
