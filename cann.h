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

struct cann;

typedef double (*cann_actfun)(const struct cann *ann, double a);

typedef struct cann
{

	int inputs,hidden_layers,hidden,outputs;

	//activation function for hidden layer and output
	cann_actfun activation_hidden;
	cann_actfun activation_output;


	//total no of weights
	int total_weights;

	//neurons of ip + hidden + op
	int total_neurons;

	//weight array
	double *weight;

	//output array
	double *output;

	//error delta for hidden and output neuron(total - inputs)
	double *delta;


}cann;



/* Creates and returns a new ann. */
cann * cann_init(int inputs, int hidden_layers, int hidden, int outputs);

/* Creates ANN from file saved with genann_write. */
cann * cann_read(FILE *in);
void cann_write(cann const *ann, FILE *out);
void cann_free(cann *ann);

double const *cann_run(cann const *ann, double const *inputs);
void cann_train(cann const *ann , double const *inputs ,double const* desired_ouputs, double learning_rate );


void cann_randomize(cann *ann);
double cann_act_threshold(const cann *ann, double a);
double cann_act_linear(const cann *ann, double a);
double cann_act_sigmoid(const cann *ann, double a);
