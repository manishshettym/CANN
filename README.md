# CANN - C based Artificial Neural Network

This a simple library which integrates a lot of functionalities for basic ANN operations.
For more details on the library, read the short paper summary - [Paper](https://github.com/ManishShettyM/CANN/blob/master/Analysis/CANN.pdf)

## Comparison with python modules

#### 1. Gate functionalities : Creating ANN for gates like XOR,AND,OR etc in a low level language like C is much faster and hiighly compatible too.
#### 2. Regression : The Auto-Mpg dataset was used to solve a MULTIPLE LINEAR REGRESSION problem using the ANN.

##### Execution Time
|ANN library	|Iterations	|RMSE	|Time(sec)	| 
|:---:|:---:|:---:|:---:|  
|CANN	|10<sup>3</sup>	|9.586	|0.06558	| 
|Keras	|10<sup>3</sup>	|5.516	|13.9793	| 

  
#### 3.Classification : The IRIS dataset was used to solve a 3 class problem using the ANN.

##### Execution Time
|ANN library	|Iterations	|Accuracy	|Time(sec)	| 
|:---:|:---:|:---:|:---:| 
|CANN	|10<sup>3</sup>	|96.7%	|0.1834	| 
|Keras	|10<sup>3</sup>	|96.6%	|33.72608	| 


## Steps to compile and Execute
##### Compile
```gcc -w cann.c <test_main_file.c> -lm```

<test_main_file.c> can be : 
1. test_xor.c 
2. test_iris.c 
3. test_mpg.c 

Corresponding python implementations are: 
1. iris_test.py 
2. mpg_test.py 

##### Execute
> ./a.out
