//g++ -g -o testing -std=c++11 Tanhtest.cpp ../Header/Operator.cpp  ../Header/Tensor.cpp ../Header/Data.cpp ../Header/Shape.cpp

#include <iostream>
#include <string>

#include "..//Header//NeuralNetwork.h"
//#include "..//Header//Temporary_method.h"
int main(int argc, char const *argv[]){

  //Operator<float> *x     = new Tensorholder<float>(Tensor<float>::Truncated_normal(2, 3, 1, 1, 4, 0.0, 0.1), "x");
  Operator<float> *x     = new Tensorholder<float>(Tensor<float>::Constants(2, 3, 1, 1, 4, 1.0), "x");
  //Operator<float> *label = new Tensorholder<float>(Tensor<float>::Truncated_normal(2, 3, 1, 1, 4, 0.0, 0.1), "label");
  Operator<float> *label = new Tensorholder<float>(Tensor<float>::Constants(2, 3, 1, 1, 4, 1.0), "label");

  Operator<float> *w1 = new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, 4, 3, 0.0, 0.1), "weight1");
  Operator<float> *b1 = new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, 3, 0.1), "bias1");
  Operator<float> *w2 = new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, 4, 3, 0.0, 0.1), "weight2");
  Operator<float> *b2 = new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, 3, 0.1), "bias2");
  Operator<float> *w3 = new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, 4, 3, 0.0, 0.1), "weight3");
  Operator<float> *b3 = new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, 3, 0.1), "bias3");

  printf("x\n");
  std::cout<< x->GetResult() << std::endl;
  printf("w1\n");
  std::cout<< w1->GetResult() << std::endl;
  printf("w2\n");
  std::cout<< w2->GetResult() << std::endl;
  printf("w3\n");
  std::cout<< w3->GetResult() << std::endl;
  printf("b1\n");
  std::cout<< b1->GetResult() << std::endl;
  printf("b2\n");
  std::cout<< b2->GetResult() << std::endl;
  printf("b3\n");
  std::cout<< b3->GetResult() << std::endl;

  Operator<float> *out = new Recurrent<float>(x, w1, w2, w3, b1, b2, b3, 3, "rnn");
  out->ComputeForwardPropagate();
  std::cout<< out->GetResult() << std::endl;

  Objective<float> *objective = new MSE<float>(out, label, "MSE");

  SetObjective(objective);

  Optimizer<float> *optimizer = new GradientDescentOptimizer<float>(GetTensorholder(), 0.001, MINIMIZE);

  SetOptimizer(optimizer);


//  out->ComputeBackPropagate();
//  std::cout<< out->GetDelta() << std::endl;

  return 0;
}
