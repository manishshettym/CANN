#include "cann.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>







cann *cann_read(FILE *in)
{

	int inputs,hidden_layers,hidden, outputs;
	int err_check;

	errno = 0;
	err_check = fscanf(in,"%d %d %d %d",
	 			&inputs,&hidden_layers,&hidden,&outputs);

	if(err_check<4 || errno !=0)
	{
		perror("init issue");
		return NULL:
	}


	cann * ann = cann_init(inputs,hidden_layers,hidden, outputs);
	//setup an ann with totalweights = total records

	for (int i = 0; i < ann->total_weight; ++i)
	{
		errno=0;
		err_check = fscanf(in," %le", ann->weight + i);

		if(err_check<1 || errno!=0)
		{
			perror("weight issue");
			//free the ann
			return NULL;
		}

	}


	return ann;
}


