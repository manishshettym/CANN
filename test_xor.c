#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cann.h"

float train_accuracy_check(double predicted[] ,double expected[] , int n)
{


    double sum=0;
    double actual_sum=0;
    for (int i = 0; i < n; ++i)
    {
        sum+= fabs(expected[i] - predicted[i]);
        actual_sum+=expected[i];

        //printf("%f %f %f %f\n\n\n",expected[i],predicted[i],sum,actual_sum);
    }

    

    return ((sum/actual_sum)*100);
}

int main(int argc, char *argv[])
{



    printf("TRAINING CANN .. on XOR\n");
    printf("Method: Backpropagation.\n");

    /* This will make the neural network initialize differently each run. */
    /* If you don't get a good result, try again for a different result. */
    srand(time(0)); //seed basically

    /* Input and expected out data for the XOR function. */
    const double input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const double expected[4] = {0, 1, 1, 0};
    int i;

    /* New network with 2 inputs,
     * 1 hidden layer of 2 neurons,
     * and 1 output. */
    cann *ann = cann_init(2, 1, 2, 1);

    double op[4]={0};

    /* Train on the four labeled data points 300 itereations */
    for (i = 0; i < 300; ++i) 
    {
        cann_train(ann, input[0], expected + 0, 3);
        cann_train(ann, input[1], expected + 1, 3);
        cann_train(ann, input[2], expected + 2, 3);
        cann_train(ann, input[3], expected + 3, 3);


    //printf("Output for [%1.f, %1.f] is %f.\n", input[0][0], input[0][1], *cann_run(ann, input[0]));
    //printf("Output for [%1.f, %1.f] is %f.\n", input[1][0], input[1][1], *cann_run(ann, input[1]));
    //printf("Output for [%1.f, %1.f] is %f.\n", input[2][0], input[2][1], *cann_run(ann, input[2]));
    //printf("Output for [%1.f, %1.f] is %f.\n\n\n\n", input[3][0], input[3][1], *cann_run(ann, input[3]));

        //run and check training accuracy
        op[0] = *cann_run(ann, input[0]);
        op[1] = *cann_run(ann, input[1]);
        op[2] = *cann_run(ann, input[2]);
        op[3] = *cann_run(ann, input[3]);    
        printf("Training error @epoch[%d]: %f\n",i+1,train_accuracy_check(op,expected,4));

    }

    /* Run the network and see what it predicts using forward prediction. */
    printf("Output for [%1.f, %1.f] is %f.\n", input[0][0], input[0][1], *cann_run(ann, input[0]));
    printf("Output for [%1.f, %1.f] is %f.\n", input[1][0], input[1][1], *cann_run(ann, input[1]));
    printf("Output for [%1.f, %1.f] is %f.\n", input[2][0], input[2][1], *cann_run(ann, input[2]));
    printf("Output for [%1.f, %1.f] is %f.\n", input[3][0], input[3][1], *cann_run(ann, input[3]));

    cann_free(ann);
    return 0;
}
