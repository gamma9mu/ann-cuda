#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "data.h"
#include "backprop.h"

int
main(int argc, char *argv[]) {
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
    size_t hidden_size = (HIDDEN_SIZE * count) * sizeof(float);
    size_t output_size = (OUTPUT_SIZE * count) * sizeof(float);

    /* Allocates the memory for the different floats */
    expected = malloc(output_size);
    w_ih = malloc(hidden_size);
    theta_h = malloc(hidden_size);
    w_ho = malloc(output_size);
    theta_o = malloc(output_size);
    output = malloc(output_size);



    return EXIT_SUCCESS;
}

float randw(int numPrevious) {
    float weight = ((float)rand())/((float)RAND_MAX);
    weight *= numPrevious;
    weight -= numPrevious / 2.0f;

    return weight;
}
