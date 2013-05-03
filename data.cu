/* Ensure strtof is available. */
#if (!defined(_POSIX_C_SOURCE)) || (_POSIX_C_SOURCE < 200112L)
#define _POSIX_C_SOURCE 200112L;
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

extern "C" {
#include "data.h"
#include "backprop.h"
}

#define BUF_SZ 256


/* Performs normalization on a data set read in row-major order, normalizes it,
 * performs cross-validation, and transposes it to column-major order.
 */

extern "C" int
readdata( float **data )
{
    FILE *file;
    float *rmData; //Row Major Order
    rmData = (float *) malloc(NUM_COLUMNS * NUM_ROWS * sizeof(float));

    float maxValues[4];
    float minValues[4];
    int lineCount = 0;
    int dataCount = 0;
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
                //printf("token: %s\n", token);
                token = strtok(NULL, ",");
                dataCount++;
            }
            lineCount++;
            dataCount = 0;
        }

        fclose ( file );


        /*
        for(int i = 0; i < lineCount; i++)
        {
            for(int j = 0; j < 4; j++)
            {
                printf("Row %i Column %i : %f \n", i, j, data[i][j]);
            }
        }
        

        printf("Max Values %f,%f,%f,%f \n", 
            maxValues[0],maxValues[1],maxValues[2],maxValues[3]);

         printf("Min Values %f,%f,%f,%f \n", 
            minValues[0],minValues[1],minValues[2],minValues[3]);
         */
         
        //Normalize Data
        for(int i = 0; i < lineCount; i++)
        {
            for(int j = 0; j < 4; j++)
            {
                //Normalize
                rmData[(i * NUM_COLUMNS) + j] = 
                    ((rmData[(i * NUM_COLUMNS) + j] - minValues[j])/(maxValues[j] - minValues[j]));
            }
        }
        
        *data = rmData;

        /*
        for(int i = 0; i < lineCount; i++)
        {
            for(int j = 0; j < 4; j++)
            {
                printf("Row %i Column %i : %f \n", i, j, data[i][j]);
            }
        }*/

    }

    return lineCount;
}

extern "C" int
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
