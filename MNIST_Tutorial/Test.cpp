/*g++ -g -o testing -std=c++11 SLP_Softmax_Cross_Entropy_With_MNIST_no_use_NeuralNetwork.cpp ../Header/Operator.cpp ../Header/NeuralNetwork.cpp ../Header/Tensor.cpp*/

#include <iostream>
#include <string>

#include "..//Header//NeuralNetwork.h"
#include "..//Header//Temporary_method.h"
#include "MNIST_Reader.h"
//
#define BATCH             1
#define LOOP_FOR_TRAIN    1000
// 10,000 is number of Test data
#define LOOP_FOR_TEST     (10000 / BATCH)

int main(int argc, char const *argv[]) {
    // create input, label data placeholder, placeholder is always managed by NeuralNetwork
    Operator<float> *x = new Placeholder<float>(Tensor<float>::Truncated_normal(1, BATCH, 1, 1, 25, 0.0, 0.6), "x");
    // Operator<float> *label = new Placeholder<float>(Tensor<float>::Constants(1, BATCH, 1, 1, 10, 0.0), "label");
    Operator<float> *res = new Reshape<float>(x, 1, 5, 5, "reshape");

    Operator<float> *maxpool = new Maxpooling<float>(res, 2, 2, "maxpool");

    // Operator<float> *weight = new Variable<float>(Tensor<float>::Constants(1, 10, 1, 3, 3, 1.0), "weight");
    //
    // Operator<float> *conv = new Convolution<float>(res, weight, 1, 1, 1, 1, "convolution");

    // Operator<float> *threshold = new Threshold<float>(conv, "threshold");
    //
    // MNISTDataSet<float> *dataset = CreateMNISTDataSet<float>();
    // dataset->CreateTrainDataPair(BATCH);
    //
    // x->FeedOutput(dataset->GetTrainFeedImage());
    // label->FeedOutput(dataset->GetTrainFeedLabel());

    res->ComputeForwardPropagate();
    maxpool->ComputeForwardPropagate();

    maxpool->GetOutput()->PrintShape();
    maxpool->GetOutput()->PrintData();
    res->GetOutput()->PrintShape();
    res->GetOutput()->PrintData();
    // conv->ComputeForwardPropagate();
    //
    // conv->GetOutput()->PrintData(1);
    // conv->GetOutput()->PrintShape();
    // weight->GetOutput()->PrintData(1);
    // weight->GetOutput()->PrintShape();
    // res->GetOutput()->PrintData(1);
    // res->GetOutput()->PrintShape();
    // label->GetOutput()->PrintData(1);
    // label->GetOutput()->PrintShape();


    return 0;
}
