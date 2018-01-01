/*g++ -g -o testing -std=c++11 Convolution_test.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp*/

#include <iostream>
#include <string>

// #include "..//Header//NeuralNetwork.h"
#include "..//Header//DLOP.h"
#include "..//Header//Temporary_method.h"
#include "MNIST_Reader.h"
//
#define BATCH             100
#define LOOP_FOR_TRAIN    100
// 10,000 is number of Test data
#define LOOP_FOR_TEST     (10000 / BATCH)

int main(int argc, char const *argv[]) {
    Operator<float> *w      = new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, 5, 5, 0.0, 0.6), "weight");
    Operator<float> *w_pool = new Maxpooling4D<float>(w, 2, 2, 2, 2, "maxpool");

    w_pool->ComputeForwardPropagate();

    std::cout << w->GetResult() << '\n';
    std::cout << w_pool->GetResult() << '\n';

    Tensor<float> *del = w_pool->GetDelta();
    int capa           = del->GetData()->GetCapacity();

    for (int i = 0; i < capa; i++) {
        (*del)[i] = 1.0;
    }

    w_pool->ComputeBackPropagate();

    std::cout << w->GetDelta() << '\n';

    delete w;
    delete w_pool;

    return 0;
}
