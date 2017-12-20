/*g++ -g -o testing -std=c++11 SLP_Softmax_Cross_Entropy_With_MNIST_no_use_NeuralNetwork.cpp ../Header/Operator.cpp ../Header/NeuralNetwork.cpp ../Header/Tensor.cpp*/

#include <iostream>
#include <string>

#include "..//Header//NeuralNetwork.h"
#include "..//Header//Temporary_method.h"
#include "MNIST_Reader.h"
//
#define BATCH             2
#define LOOP_FOR_TRAIN    1000
// 10,000 is number of Test data
#define LOOP_FOR_TEST     (10000 / BATCH)

int main(int argc, char const *argv[]) {
    // create input, label data placeholder, placeholder is always managed by NeuralNetwork
    Operator<float> *x = new Placeholder<float>(Tensor<float>::Truncated_normal(1, BATCH, 1, 1, 10, 0.0, 0.6), "x");
    // Operator<float> *label = new Placeholder<float>(Tensor<float>::Constants(1, BATCH, 1, 1, 10, 0.0), "label");
    Operator<float> *res = new Reshape<float>(x, 1, 2, 5, "reshape");

    res->ComputeForwardPropagate();

    res->PrintData();

    // ======================= layer 1=======================
    // Operator<float> *w      = new Variable<float>(Tensor<float>::Zeros(1, 1, 1, 784, 10), "w");
    // Operator<float> *b      = new Variable<float>(Tensor<float>::Zeros(1, 1, 1, 1, 10), "b");
    // Operator<float> *matmul = new MatMul<float>(x, w, "matmul");
    // Operator<float> *add    = new Add<float>(matmul, b, "add");
    //
    //// ======================= Error=======================
    // Operator<float> *err = new SoftmaxCrossEntropy<float>(add, label, 1e-50, "SCE");
    //
    //// ======================= Optimizer=======================
    // Optimizer<float> *optimizer = new GradientDescentOptimizer<float>(err, 0.01, MINIMIZE);

    // ======================= Create Graph ===================
    // optimizer->AddTrainableData(w->GetOutput(), w->GetGradient());
    // optimizer->AddTrainableData(b->GetOutput(), b->GetGradient());

    // ======================= Prepare Data ===================


    return 0;
}
