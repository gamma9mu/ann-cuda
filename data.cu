#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

extern "C" {
#include "data.h"
}

/* Performs normalization on a data set read in row-major order, normalizes it,
 * performs cross-validation, and transposes it to column-major order.
 */

__host__ void
readdata( void )
{
    FILE *file;
    float data[NUM_ROWS][NUM_COLUMNS];
    float transpose[NUM_COLUMNS][NUM_ROWS];
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

                data[lineCount][dataCount] = temp;
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
        */

        printf("Max Values %f,%f,%f,%f \n", 
            maxValues[0],maxValues[1],maxValues[2],maxValues[3]);

         printf("Min Values %f,%f,%f,%f \n", 
            minValues[0],minValues[1],minValues[2],minValues[3]);
         
        //Normalize Data
        for(int i = 0; i < lineCount; i++)
        {
            for(int j = 0; j < 4; j++)
            {
                //Normalize
                data[i][j] = 
                    ((data[i][j] - minValues[j])/(maxValues[j] - minValues[j]));

                //Transpose
                transpose[j][i] = data[i][j];
            }
        }
        
        /*
        for(int i = 0; i < lineCount; i++)
        {
            for(int j = 0; j < 4; j++)
            {
                printf("Row %i Column %i : %f \n", i, j, data[i][j]);
            }
        }*/

    }
}
