# CANN
C based Artificial Neural Network

This a simple library which integrates a lot of functionalities for basic ANN operations.

## Compile
gcc -w cann.c <test_main_file.c> -lm

## Execute
./a.out

## Comparison with python modules

### 1.Gate functionalities : Creating ANN for gates like XOR,AND,OR etc in a low level language like C is much faster and hiighly compatible too.

### 2.Regression : The Auto-Mpg dataset was used to solve a MULTIPLE LINEAR REGRESSION problem using the ANN.
Results: 
  
  CANN 
  Iterations:1000 
  RMSE:9.586 
  Time(sec):0.06558 
  
  KERAS 
  Iterations:1000
  RMSE:5.516 
  Time(sec):13.9793 
  
### 3.Classification : The IRIS dataset was used to solve a 3 class problem using the ANN.
Results: 

  CANN 
  Iterations:1000 
  Accuracy:96.7% 
  Time(sec):0.1834 
  
  KERAS 
  Iterations:1000
  Accuracy:96.6% 
  Time(sec):33.72608


 
  

