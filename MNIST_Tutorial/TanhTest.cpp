//g++ -g -o testing -std=c++11 Tanhtest.cpp ../Header/Operator.cpp  ../Header/Tensor.cpp ../Header/Data.cpp ../Header/Shape.cpp

#include <iostream>
#include <string>

#include "..//Header//NeuralNetwork.h"
int main(int argc, char const *argv[]){
  Operator<float> *tensor = new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, 1, 3, 0.0, 0.1),"00");
  std::cout<< tensor->GetResult() << std::endl;

  Operator<float> *out = new Tanh<float>(tensor, "11");
  out->ComputeForwardPropagate();
  std::cout<< out->GetResult() << std::endl;
  out->ComputeBackPropagate();
  std::cout<< out->GetDelta() << std::endl;

  return 0;
}
