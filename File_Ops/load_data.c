
const char *class_names[] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};


void load_data(const char* path, int att_count , int class_count)
{
	FILE *in = fopen(path, "r");
	if(!in)
	{
		printf("Could not open file\n");
		exit(1);
	}


	char line[1024];
	int sample_count=0;
	while(!feof(in) && fgets(line , 1024 , in))
	{
		sample_count++;
	}
	fseek(in,0,SEEK_SET);

	printf("Loading %d datapoints from %s\n",sample_count,path);


	double * input = malloc(sizeof(double) * samples * att_count);
	double * classes = malloc(sizeof(double)* samples * class_count);

	for (int i = 0; i < sample_count; ++i) //for every sample
	{
		double *p = input[i*4];
		double *c = classes[i*3];
		c[0]=c[1]=c[2]=0.0;

		if (fgets(line, 1024, in) == NULL) 
		{
            perror("fgets");
            exit(1);
        }

        char *split = strtok(line, ",");
        for (j = 0; j < att_count; ++j) // for every attribute
        {
            p[j] = atof(split);
            split = strtok(0, ",");
        }

        split[strlen(split)-1] = 0;
        if (strcmp(split, class_names[0]) == 0) {c[0] = 1.0;}
        else if (strcmp(split, class_names[1]) == 0) {c[1] = 1.0;}
        else if (strcmp(split, class_names[2]) == 0) {c[2] = 1.0;}
        else {
            printf("Unknown class %s.\n", split);
            exit(1);
        }

        /* printf("Data point %d is %f %f %f %f  ->   %f %f %f\n", i, p[0], p[1], p[2], p[3], c[0], c[1], c[2]); */
    }

    fclose(in);


		
	}

}