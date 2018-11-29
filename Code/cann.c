#include "cann.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#ifndef cann_act
#define cann_act_hidden cann_act_hidden_indirect
#define cann_act_output cann_act_output_indirect
#endif

unsigned long long rdtsc()
{
  unsigned long long int x;
  unsigned a, d;

  __asm__ volatile("rdtsc" : "=a" (a), "=d" (d));

  return ((unsigned long long)a) | (((unsigned long long)d) << 32);
}


/*SUMMARY STATISTICS*/
float loss(double predicted[] ,double expected[] , int n)
{
	//using MSSE => mean(Sum(squared(errors)))

    double sum_squared_errors=0;
    for (int i = 0; i < n; ++i)
    {
        sum_squared_errors+= pow(expected[i] - predicted[i],2);

    }

    return ((sum_squared_errors/n));
}

float test_rmse(double predicted[] ,double expected[] , int n)
{
	//using RMSE => rrot(mean(squared(errors)))
	double sum_squared_errors=0;
    for (int i = 0; i < n; ++i)
    {
        sum_squared_errors+= pow(expected[i] - predicted[i],2);

    }

    return (sqrt(sum_squared_errors/n));
  
}




/*IMPLEMENTATION*/
double cann_act_hidden_indirect(const struct cann *ann, double a) 
{
    return ann->activation_hidden(ann, a);
}

double cann_act_output_indirect(const struct cann *ann, double a) 
{
    return ann->activation_output(ann, a);
}


double cann_act_sigmoid(const cann *ann, double a)
{
	if(a < -45.0) return 0;

	if(a > 45.0) return 1;

	return 1.0 / (1 + exp(-a));
}

double cann_act_linear(const cann *ann, double a) 
{
    return a;
}

double cann_act_threshold(const cann *ann, double a) 
{
    return a > 0;
}



void cann_randomize(cann *ann)
{
	for (int i = 0; i < ann->total_weights; ++i)
	{
		ann->weight[i] = cann_random() - 0.5;
	}
}


cann *cann_init(int inputs, int hidden_layers, int hidden, int outputs) 
{
	//check for errors

    if (hidden_layers < 0) return 0;
    if (inputs < 1) return 0;
    if (outputs < 1) return 0;
    if (hidden_layers > 0 && hidden < 1) return 0;


    const int hidden_weights = hidden_layers ? (inputs+1) * hidden + (hidden_layers-1) * (hidden+1) * hidden : 0;
    const int output_weights = (hidden_layers ? (hidden+1) : (inputs+1)) * outputs;
    const int total_weights = (hidden_weights + output_weights);

    const int total_neurons = (inputs + hidden * hidden_layers + outputs);

    /* Allocate extra size for weights, outputs, and deltas. */
    const int size = sizeof(cann) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
    cann *ret = malloc(size);
    if (!ret) return 0;

    ret->inputs = inputs;
    ret->hidden_layers = hidden_layers;
    ret->hidden = hidden;
    ret->outputs = outputs;

    ret->total_weights = total_weights;
    ret->total_neurons = total_neurons;

    /* Set array pointer */
    ret->weight = (double*)((char*)ret + sizeof(cann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    cann_randomize(ret);

    ret->activation_hidden = cann_act_sigmoid;
    ret->activation_output = cann_act_sigmoid;

    return ret;
}

double const *cann_run(cann const *ann, double const *inputs) 
{
    double const *w = ann->weight;
    double *o = ann->output + ann->inputs;
    double const *i = ann->output;

    /* Copy the inputs to the scratch area, where we also store each neuron's
     * output, for consistency. This way the first layer isn't a special case. */
    memcpy(ann->output, inputs, sizeof(double) * ann->inputs);

    int h, j, k;

    if (!ann->hidden_layers) 
    {
        //printf("Entered 1\n");
        double *ret = o;
        for (j = 0; j < ann->outputs; ++j) 
        {
            double sum = *w++ * -1.0;
            for (k = 0; k < ann->inputs; ++k) 
            {
                //printf("%f ",*w);
                sum += *w++ * i[k];
            }
            *o++ = cann_act_output(ann, sum);
        }

        return ret;
    }

    /*input layer */
    for (j = 0; j < ann->hidden; ++j) 
    {
        double sum = *w++ * -1.0;
        for (k = 0; k < ann->inputs; ++k) 
        {
            sum += *w++ * i[k];
        }
        *o++ = cann_act_hidden(ann, sum);
    }

    i += ann->inputs;

    /* hidden layers , if any. */
    for (h = 1; h < ann->hidden_layers; ++h) 
    {
        for (j = 0; j < ann->hidden; ++j) 
        {
            double sum = *w++ * -1.0;
            for (k = 0; k < ann->hidden; ++k) 
            {
                sum += *w++ * i[k];
            }
            *o++ = cann_act_hidden(ann, sum);
        }

        i += ann->hidden;
    }

    double const *ret = o;

    /* Figure output layer. */
    for (j = 0; j < ann->outputs; ++j) 
    {
        double sum = *w++ * -1.0;
        for (k = 0; k < ann->hidden; ++k) 
        {
            sum += *w++ * i[k];
        }
        *o++ = cann_act_output(ann, sum);
    }

    /* Sanity check that we used all weights and wrote all outputs. */
    assert(w - ann->weight == ann->total_weights);
    assert(o - ann->output == ann->total_neurons);

    return ret;
}



void cann_train(cann const *ann , double const *inputs ,
				 double const* desired_ouputs, double learning_rate )
{

	//Forward propogate first:
	cann_run(ann,inputs);

	int h,j,k;

	/* deltas for various layers */

	//1. output layer
	{
		double const *op = ann->output + ann->inputs + ann->hidden*ann->hidden_layers; //first op
		double *del = ann->delta + ann->hidden*ann->hidden_layers; //first delta

		double const *t = desired_ouputs; //first desired op


		if(cann_act_output == cann_act_linear || ann->activation_output == cann_act_linear)
		{
            //printf("Entered\n\n");
			for(j= 0; j< ann->outputs ; j++)
			{
				//set del
				*del= (*t - *op);
                //printf("t:%f - op:%f = del:%f\n",*t,*op,*del);
				//move pointers
				del++;
				op++; 
				t++;
			}
		}

		//what if not linear activation
		else
		{
            
			for(j= 0; j< ann->outputs ; j++)
			{
				//set del
				*del= (*t - *op) * *op * (1 - *op); //with derivative multiplied

				//move pointers
				del++;
				op++; 
				t++;
			}
		}
	}



	//2. hidden layer(start from back towards inputs)

	for(h = ann->hidden_layers-1 ; h>=0 ; h--) //will be constant time 
	{
		double const *op = ann->output + ann->inputs + (h * ann->hidden);
        double *del = ann->delta + (h * ann->hidden);

        //delta and weight of next layer
        double const * const dd = ann->delta + ((h+1)*ann->hidden);
        double const * const ww =ann->weight + ((ann->inputs+1) * ann->hidden) + ((ann->hidden+1) * ann->hidden * (h));

        //set deltas
        for(j=0 ; j<ann->hidden ; j++)
        {
        	double delta = 0;

        	for(k=0 ; k<(h == ann->hidden_layers-1 ? ann->outputs : ann->hidden); ++k)
        	{
        		const double forward_delta = dd[k];
                const int windex = k * (ann->hidden + 1) + (j + 1);
                const double forward_weight = ww[windex];
                delta += forward_delta * forward_weight;
        	}

        	*del = *op * (1.0 - *op) * delta;
            del++;
            op++;
        }


	}



	/* TRAINING output layer */
    
    /* Find first output delta. */
    double const *d = ann->delta + ann->hidden * ann->hidden_layers; /* First output delta. */

    /* Find first weight to first output delta. */
    double *w = ann->weight + 
    			(ann->hidden_layers? 
    			((ann->inputs+1) * ann->hidden + (ann->hidden+1) * ann->hidden * (ann->hidden_layers-1))
            	: (0));

    /* Find first output in previous layer. */
    double const * const i = ann->output + (ann->hidden_layers
            ? (ann->inputs + (ann->hidden) * (ann->hidden_layers-1))
            : 0);

    /* Set output layer weights. */
    for (j = 0; j < ann->outputs; ++j) 
    {
        *w++ += *d * learning_rate * -1.0;
        for (k = 1; k < (ann->hidden_layers ? ann->hidden : ann->inputs) + 1; ++k) 
        {
            *w++ += *d * learning_rate * i[k-1];
        }

        ++d;
    }
    assert(w - ann->weight == ann->total_weights);




    /*TRAINING hidden layers*/
    for (h = ann->hidden_layers - 1; h >= 0; --h) 
    {

        /* Find first delta in this layer. */
        double const *d = ann->delta + (h * ann->hidden);

        /* Find first input to this layer. */
        double const *i = ann->output + (h
                ? (ann->inputs + ann->hidden * (h-1))
                : 0);

        /* Find first weight to this layer. */
        double *w = ann->weight + (h
                ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * (ann->hidden) * (h-1))
                : 0);


        for (j = 0; j < ann->hidden; ++j) 
        {
            *w++ += *d * learning_rate * -1.0;
            for (k = 1; k < (h == 0 ? ann->inputs : ann->hidden) + 1; ++k) 
            {
                *w++ += *d * learning_rate * i[k-1];
            }
            ++d;
        }

    }

}

void cann_free(cann *ann)
{
    free(ann);
}


cann *cann_read(FILE *in_struct,FILE *in_weight)
{

	int inputs,hidden_layers,hidden, outputs;
	
	int err_check;

	errno = 0;
	err_check = fscanf(in_struct,"%d %d %d %d",
	 			&inputs,&hidden_layers,&hidden,&outputs);

	if(err_check<4 || errno !=0)
	{
		perror("init issue");
		return NULL;
	}


	cann * ann = cann_init(inputs,hidden_layers,hidden, outputs);
	//setup an ann with totalweights = total records

	for (int i = 0; i < ann->total_weights; ++i)
	{
		errno=0;
		err_check = fscanf(in_weight," %le", ann->weight + i);

		if(err_check<1 || errno!=0)
		{
			perror("weight issue");
			//free the ann
			return NULL;
		}

	}


	return ann;
}

void cann_write(cann const *ann, FILE *out_struct, FILE *out_weight) 
{
    fprintf(out_struct, "%d %d %d %d", ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);

    int i;
    for (i = 0; i < ann->total_weights; ++i) 
    {
        fprintf(out_weight, " %.20e", ann->weight[i]);
    }
}

