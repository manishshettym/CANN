#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include<math.h>

#include "cann.h"

void test1()
{
    printf("TRAINING CANN .. on XOR\n");
    printf("Method: Backpropagation.\n");

    /* This will make the neural network initialize differently each run. */
    /* If you don't get a good result, try again for a different result. */
    srand(time(0)); //seed basically

    /* Input and expected out data for the XOR function. */
    double input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double expected[4] = {0, 1, 1, 0};
    int i;

    /* New network with 2 inputs,
     * 1 hidden layer of 2 neurons,
     * and 1 output. */
    cann *ann = cann_init(2, 1, 2, 1);

    double op[4]={0};

    /* Train on the four labeled data points 300 itereations */
    for (i = 0; i < 500; ++i) 
    {
        cann_train(ann, input[0], expected + 0, 3);
        cann_train(ann, input[1], expected + 1, 3);
        cann_train(ann, input[2], expected + 2, 3);
        cann_train(ann, input[3], expected + 3, 3);

        //run and check training accuracy
        op[0] = *cann_run(ann, input[0]);
        op[1] = *cann_run(ann, input[1]);
        op[2] = *cann_run(ann, input[2]);
        op[3] = *cann_run(ann, input[3]);    
        printf("loss @epoch[%d]: %f\n",i+1,loss(op,expected,4));
	
    }

	printf("\n\n\n\nSUMMARY STATISTICS\n\n");


    /* Run the network and see what it predicts using forward prediction. */
	op[0] = *cann_run(ann, input[0]);
        op[1] = *cann_run(ann, input[1]);
        op[2] = *cann_run(ann, input[2]);
        op[3] = *cann_run(ann, input[3]);

    printf("Output for [%1.f, %1.f] is %f.\n", input[0][0], input[0][1],op[0]);
    printf("Output for [%1.f, %1.f] is %f.\n", input[1][0], input[1][1],op[1]);
    printf("Output for [%1.f, %1.f] is %f.\n", input[2][0], input[2][1],op[2]);
    printf("Output for [%1.f, %1.f] is %f.\n", input[3][0], input[3][1],op[3]);
	
	float testrmse = test_rmse(op,expected,4);
	printf("Test RMSE: %f\n",testrmse);

    cann_free(ann);
}


void test2()
{
    printf("TRAINING CANN .. on XOR\n");
    printf("Method: Backpropagation.\n");

    /* This will make the neural network initialize differently each run. */
    /* If you don't get a good result, try again for a different result. */
    srand(time(0)); //seed basically

    /* Input and expected out data for the XOR function. */
    double input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double expected[4] = {0, 1, 1, 0};
    int i;

    /* New network with 2 inputs,
     * 1 hidden layer of 2 neurons,
     * and 1 output. */
    cann *ann = cann_init(2, 1, 2, 1);

    double op[4]={0};

    /* Train on the four labeled data points 300 itereations */
    for (i = 0; i < 500; ++i) 
    {
        cann_train(ann, input[0], expected + 0, 3);
        cann_train(ann, input[1], expected + 1, 3);
        cann_train(ann, input[2], expected + 2, 3);
        cann_train(ann, input[3], expected + 3, 3);

        //run and check training accuracy
        op[0] = *cann_run(ann, input[0]);
        op[1] = *cann_run(ann, input[1]);
        op[2] = *cann_run(ann, input[2]);
        op[3] = *cann_run(ann, input[3]);    
        printf("loss @epoch[%d]: %f\n",i+1,loss(op,expected,4));
	
    }

	printf("\n\n\n\nSUMMARY STATISTICS\n\n");


    /* Run the network and see what it predicts using forward prediction. */
	op[0] = *cann_run(ann, input[0]);
        op[1] = *cann_run(ann, input[1]);
        op[2] = *cann_run(ann, input[2]);
        op[3] = *cann_run(ann, input[3]);

    printf("Output for [%1.f, %1.f] is %f.\n", input[0][0], input[0][1],op[0]);
    printf("Output for [%1.f, %1.f] is %f.\n", input[1][0], input[1][1],op[1]);
    printf("Output for [%1.f, %1.f] is %f.\n", input[2][0], input[2][1],op[2]);
    printf("Output for [%1.f, %1.f] is %f.\n", input[3][0], input[3][1],op[3]);
	
	float testrmse = test_rmse(op,expected,4);
	printf("Test RMSE: %f\n",testrmse);

	FILE * save_struct = fopen("xor_ann_struct","w");
    FILE * save_weight = fopen("xor_ann_weight","w");
	cann_write(ann,save_struct,save_weight);

    cann_free(ann);
	
}

void test3()
{
	printf("LOADING TRAINED CANN .. for xor\n");
	
	FILE * saved_ann_struct = fopen("xor_ann_struct" , "r");
    FILE * saved_ann_weight = fopen("xor_ann_weight" , "r");

	
	cann *ann = cann_read(saved_ann_struct,saved_ann_weight);
	fclose(saved_ann_struct);
    fclose(saved_ann_weight);

	//Input only
    double input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double op[4]={0};

	/* Run the network and see what it predicts using forward prediction. */
	    op[0] = *cann_run(ann, input[0]);
        op[1] = *cann_run(ann, input[1]);
        op[2] = *cann_run(ann, input[2]);
        op[3] = *cann_run(ann, input[3]);

    printf("Output for [%1.f, %1.f] is %f.\n", input[0][0], input[0][1],op[0]);
    printf("Output for [%1.f, %1.f] is %f.\n", input[1][0], input[1][1],op[1]);
    printf("Output for [%1.f, %1.f] is %f.\n", input[2][0], input[2][1],op[2]);
    printf("Output for [%1.f, %1.f] is %f.\n", input[3][0], input[3][1],op[3]);
	
	//float testrmse = test_rmse(op,expected,4);
	//printf("Test RMSE: %f\n",testrmse);

	cann_free(ann);
		
}





int main()
{

	//test1();
	test2();
	//test3();


}





