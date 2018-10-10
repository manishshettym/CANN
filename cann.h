#include<stdio.h>


//uniform random (0-1)
#define cann_random()	( ((double)rand()) / RAND_MAX )

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





}cann;

/* Creates and returns a new ann. */
cann * cann_init(int inputs, int hidden_layers, int hidden, int outputs);

/* Creates ANN from file saved with genann_write. */
cann * cann_read(FILE *in);