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

    count = readdata( data );

    size_t input_size = (INPUT_SIZE * count) * sizeof(float);


    return EXIT_SUCCESS;
}

