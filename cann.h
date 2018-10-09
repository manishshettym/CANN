#include<stdio.h>


//uniform random (0-1)
#define c_random()	( ((double)rand()) / RAND_MAX )

/*Structure defining the whole neural network
 INFO:
 	.no of inputs,outputs,hiddenlayers
 	.activation function for
 		. hidden
 		. output
	.no of weights
	.no of neurons

	.array of weights
	.array of outputs
	.array of error terms

*/

typdef cann_activationfunc


typdef struct cann {

	int inputs,hidden_layers,hidden,outputs;





}