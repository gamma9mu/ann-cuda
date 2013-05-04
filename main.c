#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "data.h"
#include "backprop.h"


/* Learning Rate */
#define RATE 0.2

#define SSE_MAX 0.01

float randw ( int numPrevious );
static void init_weights(float*, size_t, size_t);
static void init_theta(float*, size_t);
void die(const char *);

int
main(int argc, char *argv[])
{
    (void) argc;
    (void) argv;

    float *data, *expected, *w_ih, *theta_h, *w_ho, *theta_o;
    int count, growing = 0, stagnant = 0;
    long int gen = 0;
    float sse, sse_i;

    srand(time(NULL));

    /* Reads the data and stores it to the memory space.
     * Also returns count for later use.*/
    count = readdata( &data );
    if (readexpected(&expected) != count) {
        fprintf(stderr, "Training output values differ in number from inputs.\n");
        exit(EXIT_FAILURE);
    }

    /* Computes the sizes to allocate. */
    size_t input_size = (INPUT_SIZE * count) * sizeof(float);
    size_t theta_in_size = HIDDEN_SIZE * sizeof(float);
    size_t theta_out_size = OUTPUT_SIZE * sizeof(float);
    size_t weight_input_size = (HIDDEN_SIZE * INPUT_SIZE) * sizeof(float);
    size_t weight_output_size = (HIDDEN_SIZE * OUTPUT_SIZE) * sizeof(float);
    size_t output_size = (OUTPUT_SIZE * count) * sizeof(float);

    /* Allocates the memory for the different floats */
    if ((w_ih = malloc(weight_input_size)) == NULL) die("main: malloc");
    if ((theta_h = malloc(theta_in_size)) == NULL) die("main: malloc");
    if ((w_ho = malloc(weight_output_size)) == NULL) die("main: malloc");
    if ((theta_o = malloc(theta_out_size)) == NULL) die("main: malloc");

    init_weights(w_ih, INPUT_SIZE, HIDDEN_SIZE);
    init_weights(w_ho, HIDDEN_SIZE, OUTPUT_SIZE);
    init_theta(theta_h, HIDDEN_SIZE);
    init_theta(theta_o, OUTPUT_SIZE);


    printf(" Target SSE: %5.3f\n", SSE_MAX);
    sse = evaluate_wrapper(data, count, expected, w_ih, theta_h, w_ho, theta_o);
    printf("Initial SSE: %5.3f\n", sse);

    while (sse > SSE_MAX && growing < 100 && stagnant < 250) {
        backprop_wrapper(data, count, expected, w_ih, theta_h,
                w_ho, theta_o, RATE);
        sse_i = evaluate_wrapper(data, count, expected, w_ih, theta_h,
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
        printf("Current SSE: %5.3f  Generation: %ld [s: %d; g: %d]\t\r", sse,
                gen, stagnant, growing);
        ++gen;
    }
    fputc('\n', stdout);

    free(expected);
    free(data);

    return EXIT_SUCCESS;
}

float randw(int numPrevious)
{
    float weight = ((float)rand())/((float)RAND_MAX);
    weight *= numPrevious;
    weight -= numPrevious / 2.0f;

    return weight;
}

static void
init_weights(float *matrix, size_t dim_x, size_t dim_y) {
    int x, y;
    for (x = 0; x < dim_x; ++x)
        for (y = 0; y < dim_y; ++y)
            matrix[(y * dim_x) + x] = randw(dim_x);
}

static void
init_theta(float *vector, size_t length) {
    int i;
    for (i = 0; i < length; ++i)
        vector[i] = rand() / (float) RAND_MAX;
}

void
die(const char *prefix) {
    perror(prefix);
    exit(EXIT_FAILURE);
}
