#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "cann.h"

/* This example is to illustrate how to use cann.
 * It is NOT an example of good machine learning techniques.
 */


double *input;
double *expected;
int samples;

void load_data() 
{
    FILE *in = fopen("datasets/clean_mpg.csv", "r");

    char line[1024];
    while (!feof(in) && fgets(line, 1024, in)) 
    {
        ++samples;
    }
    fseek(in, 0, SEEK_SET);

    printf("Loading %d data points from mpg.csv\n", samples);

    input = malloc(sizeof(double) * samples * 5);
    expected = malloc(sizeof(double)*samples);

    int i, j;
    for (i = 0; i < samples; ++i) 
    {
        double *p = input + i*5;

        if (fgets(line, 1024, in) == NULL) 
        {
            perror("fgets");
            exit(1);
        }

        char *split = strtok(line, ",");
        
        for (j = 0; j < 5; ++j) 
        {
            p[j] = atof(split);
            split = strtok(0, ",");
        }

        split[strlen(split)-1]=0;
        expected[i]=atof(split); 
        //printf("Data point %d is %f %f %f %f %f\n", i, input[i*2],input[(i*2)+1],input[(i*2)+2],input[(i*2)+3],input[(i*2)+4]);
        //printf("Expected %f\n",expected[i]);
    }
    fclose(in);

}


int main(int argc, char *argv[])
{
    printf("Training CANN... MPG dataset\n");
    printf("Method:backpropagation\n");

    srand(time(0));

    /* Load the data from file. */
    load_data();


    /* 5 inputs.
     * 1 hidden layer(s) of 5 neurons.
     * 1 outputs
     */
    cann *ann = cann_init(5, 0, 0, 1);

    int i, j;
    int loops = 1000;


    unsigned long long t0,t1;

    for (i = 0; i < ann->total_weights; ++i) 
    {
        ann->weight[i] += ((double)rand())/RAND_MAX-0.5;
    }


    //Train the network with backpropagation.
    t0=rdtsc();
    printf("Training for %d loops over data.\n", loops);
    for (i = 0; i < loops; ++i) 
    {
        double *op = malloc(sizeof(double) * samples *1);

        for (j = 0; j < samples; ++j) //train all samples
        {

            cann_train(ann, input + j*5, expected + j, 0.000000001);
    	    op[j] = *cann_run(ann, input + j*5);
            //printf("%f ",op[j]);

        }
	   
	printf("\nloss @epoch[%d]: %f\n",i+1,loss(op,expected,samples));
    free(op);

    }

    t1=rdtsc();
    printf("Time to train:%lf seconds\n",(t1-t0)/FREQ);

    
    //TEST
    double *guess = malloc(sizeof(double) * samples *1);
    for (j = 0; j < samples; ++j) 
    {
        guess[j] = *cann_run(ann, input + j*5);
        //printf("pred:%f act:%f\n",guess[j],expected[j]);
    }

    float testrmse = test_rmse(guess,expected,samples);
    printf("Test RMSE: %f\n",testrmse);
    


    cann_free(ann);
    free(input);
    return 0;
}
