#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ann.h"

#define NUM_ROWS 150
#define NUM_COLUMNS 4
#define BUF_SZ 256

/* Learning Rate */
#define RATE 0.2
/* Maximum average SSE value */
#define SSE_MAX 0.01

static void run_cuda(float*, int, float*, float*, float*, float*, float*);
static void run_st(float*, int, float*, float*, float*, float*, float*);
static float randw ( int numPrevious );
static void init_weights(float*, size_t, size_t);
static void init_theta(float*, size_t);
static void dumpmat(float*, size_t, size_t);
static void die(const char *);
static int  readdata(float**);
static int  readexpected(float**);

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

    run_cuda(data, count, expected, w_ih, theta_h, w_ho, theta_o);
    run_st(data, count, expected, aw_ih, theta_h, aw_ho, theta_o);

    free(expected);
    free(data);

    return EXIT_SUCCESS;
}

static void
run_cuda(float *inputs, int count, float *expected,
         float *w_ih, float *theta_h, float *w_ho, float *theta_o) {
    float sse, sse_i;
    long int gen = 0;
    unsigned int growing = 0, stagnant = 0;

    d_ann_t ann;
    d_data_t data;
    copy_in_ann(&ann, w_ih, theta_h, w_ho, theta_o);
    copy_in_data(&data, inputs, count, expected, 1);

    printf("*** Running CUDA ANN\n");
    printf(" Target SSE: %5.3f\n", SSE_MAX);
    sse = evaluate_wrapper(&ann, &data);
    printf("Initial SSE: %5.3f\n", sse);

    while (sse > SSE_MAX && growing < 100 && stagnant < 250) {
        backprop_wrapper(&ann, &data, RATE);
        sse_i = evaluate_wrapper(&ann, &data);
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
    copy_out_ann(&ann, w_ih, w_ho);
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

/* Performs normalization on a data set read in row-major order, normalizes it,
 * performs cross-validation, and transposes it to column-major order.
 */
static int
readdata( float **data )
{
    FILE *file;
    float *rmData; //Row Major Order
    rmData = (float *) malloc(NUM_COLUMNS * NUM_ROWS * sizeof(float));

    float maxValues[4];
    float minValues[4];
    int lineCount = 0;
    int dataCount = 0;
    int i, j;
    file = fopen("data/iris.data", "r");
    
    //Read data from file and find max and min values for each column
    if ( file != NULL)
    {
        char line [128];

        while ( fgets ( line, sizeof line, file ) != NULL )
        {
            char* token = strtok(line, ",");
            while (token) 
            {
                float temp = atof(token);
                if(lineCount == 0)
                {
                    maxValues[dataCount] = temp;
                    minValues[dataCount] = temp;
                }
                
                if(maxValues[dataCount] < temp)
                    maxValues[dataCount] = temp;

                if(minValues[dataCount] > temp)
                    minValues[dataCount] = temp;

                rmData[(lineCount * NUM_COLUMNS) + dataCount] = temp;
                token = strtok(NULL, ",");
                dataCount++;
            }
            lineCount++;
            dataCount = 0;
        }

        fclose ( file );

        //Normalize Data
        for(i = 0; i < lineCount; i++)
        {
            for(j = 0; j < 4; j++)
            {
                //Normalize
                rmData[(i * NUM_COLUMNS) + j] = ((rmData[(i * NUM_COLUMNS) + j]
                            - minValues[j])/(maxValues[j] - minValues[j]));
            }
        }
        
        *data = rmData;
    }

    return lineCount;
}

static int
readexpected(float **expected) {
    FILE *f;
    char buf[BUF_SZ];
    int fields, lines = 0;
    float *data = (float *) malloc(NUM_COLUMNS * NUM_ROWS * sizeof(float));
    if (data == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    if ((f = fopen("data/iris.output", "r")) == NULL) {
        free(data);
        fprintf(stderr, "Could not open training data result file.\n");
        exit(EXIT_FAILURE);
    }

    while (fgets(buf, BUF_SZ, f) != NULL) {
        char *str = strtok(buf, ",");
        fields = 0;
        while (str && fields < OUTPUT_SIZE) {
            data[(lines * OUTPUT_SIZE) + fields] = strtof(str, NULL);
            str = strtok(NULL, ",");
            ++fields;
        }
        ++lines;
    }
    fclose(f);

    *expected = data;
    return lines;
}
