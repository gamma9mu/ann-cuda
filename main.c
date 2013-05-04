#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "data.h"
#include "backprop.h"


/* Learning Rate */
#define RATE 0.2

#define SSE_RATE 0.01

float randw ( int numPrevious );
void die(const char *);

int
main(int argc, char *argv[])
{
    (void) argc;
    (void) argv;

    float *data, *expected, *w_ih, *theta_h, *w_ho, *theta_o;
    int i, j, count;
    float sse, sse_max = SSE_RATE;

    srand(time(NULL));

    /* Reads the data and stores it to the memory space.
     * Also returns count for later use.*/
    count = readdata( &data );
    if (readexpected(&expected) != count) {
        fprintf(stderr, "Training output values differ in number from inputs.\n");
        exit(EXIT_FAILURE);
    }

    sse_max *= count;

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

    /* Initialize the weight matrices. */
    for (i = 0; i < INPUT_SIZE; ++i)
        for (j = 0; j < HIDDEN_SIZE; ++j)
            w_ih[(j * INPUT_SIZE) + i] = randw(INPUT_SIZE);
    for (i = 0; i < HIDDEN_SIZE; ++i)
        for (j = 0; j < OUTPUT_SIZE; ++j)
            w_ho[(j * HIDDEN_SIZE) + i] = randw(HIDDEN_SIZE);

    /* Initialize the thetas. */
    for (i = 0; i < HIDDEN_SIZE; ++i)
        theta_h[i] = rand() / (float) RAND_MAX;
    for (i = 0; i < OUTPUT_SIZE; ++i)
        theta_o[i] = rand() / (float) RAND_MAX;


    sse = sse_max + 100;
    while (sse > sse_max) {
        backprop_wrapper(data, count, expected, w_ih, theta_h,
                w_ho, theta_o, RATE);
        sse = evaluate_wrapper(data, count, expected, w_ih, theta_h,
                w_ho, theta_o);
        printf("SSE: %f\n", sse);
    }

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

void
die(const char *prefix) {
    perror(prefix);
    exit(EXIT_FAILURE);
}
