#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "data.h"
#include "backprop.h"

float randw ( int numPrevious );

int
main(int argc, char *argv[])
{
    (void) argc;
    (void) argv;

    float *data, *expected, *w_ih, *theta_h, *w_ho,
          *theta_o, *output;

    int count;


    /* Reads the data and stores it to the memory space.
     * Also returns count for later use.*/
    count = readdata( data );

    /* Computes the sizes to allocate. */
    size_t input_size = (INPUT_SIZE * count) * sizeof(float);
    size_t theta_in_size = HIDDEN_SIZE * sizeof(float);
    size_t theta_out_size = OUTPUT_SIZE * sizeof(float);
    size_t weight_input_size = (HIDDEN_SIZE * INPUT_SIZE) * sizeof(float);
    size_t weight_output_size = (HIDDEN_SIZE * OUTPUT_SIZE) * sizeof(float);
    size_t output_size = (OUTPUT_SIZE * count) * sizeof(float);

    /* Allocates the memory for the different floats */
    expected = malloc(output_size);
    w_ih = malloc(weight_input_size);
    theta_h = malloc(theta_in_size);
    w_ho = malloc(weight_output_size);
    theta_o = malloc(theta_out_size);
    output = malloc(output_size);



    return EXIT_SUCCESS;
}

float randw(int numPrevious)
{
    float weight = ((float)rand())/((float)RAND_MAX);
    weight *= numPrevious;
    weight -= numPrevious / 2.0f;

    return weight;
}
