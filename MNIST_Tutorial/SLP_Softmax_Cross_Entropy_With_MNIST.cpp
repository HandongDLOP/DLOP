/*g++ -g -o testing -std=c++11 SLP_Softmax_Cross_Entropy_With_MNIST.cpp ../Header/Operator.cpp ../Header/NeuralNetwork.cpp ../Header/Tensor.cpp*/

#include <iostream>
#include <string>

#include "..//Header//NeuralNetwork.h"
#include "..//Header//Temporary_method.h"
#include "MNIST_Reader.h"
//
#define BATCH             100
#define LOOP_FOR_TRAIN    1000
// 10,000 is number of Test data
#define LOOP_FOR_TEST     (10000 / BATCH)

int main(int argc, char const *argv[]) {
    // Network declare
    NeuralNetwork<float> HGUNN;

    // create input, label data placeholder, placeholder is always managed by NeuralNetwork
    Operator<float> *x     = HGUNN.AddPlaceholder(Tensor<float>::Constants(1, BATCH, 1, 1, 784, 0.0), "x");
    Operator<float> *label = HGUNN.AddPlaceholder(Tensor<float>::Constants(1, BATCH, 1, 1, 10, 0.0), "label");

    // ======================= layer 1=======================
    Operator<float> *w      = new Variable<float>(Tensor<float>::Zeros(1, 1, 1, 784, 10), "w");
    Operator<float> *b      = new Variable<float>(Tensor<float>::Zeros(1, 1, 1, 1, 10), "b");
    Operator<float> *matmul = new MatMul<float>(x, w, "matmul");
    Operator<float> *add    = new Add<float>(matmul, b, "add");

    // ======================= Error=======================
    Operator<float> *err = new SoftmaxCrossEntropy<float>(add, label, 1e-50, "SCE");

    // ======================= Optimizer=======================
    Optimizer<float> *optimizer = new GradientDescentOptimizer<float>(err, 0.01, MINIMIZE);

    // ======================= Create Graph ===================
    HGUNN.CreateGraph(optimizer);

    // ======================= Prepare Data ===================
    MNISTDataSet<float> *dataset = CreateMNISTDataSet<float>();

    // ======================= Training =======================
    HGUNN.PrintGraph(optimizer);

    for (int i = 0; i < LOOP_FOR_TRAIN; i++) {
        dataset->CreateTrainDataPair(BATCH);
        x->FeedOutput(dataset->GetTrainFeedImage());
        label->FeedOutput(dataset->GetTrainFeedLabel());

        HGUNN.Run(optimizer);

        if ((i % 100) == 0) std::cout << "Accuracy is : " << temp::Accuracy(add->GetOutput(), label->GetOutput(), BATCH) << '\n';
    }

    // ======================= Testing ======================
    float test_accuracy = 0.0;

    for (int i = 0; i < (int)LOOP_FOR_TEST; i++) {
        dataset->CreateTestDataPair(BATCH);
        x->FeedOutput(dataset->GetTestFeedImage());
        label->FeedOutput(dataset->GetTestFeedLabel());

        HGUNN.Run(err);
        // I'll implement flexibility about the situation that change of Batch size
        test_accuracy += temp::Accuracy(add->GetOutput(), label->GetOutput(), BATCH);
    }

    std::cout << "Test Accuracy is : " << test_accuracy / (int)LOOP_FOR_TEST << "\n\n";

    // ======================= delete ======================
    // ~MNISTDataSet
    delete dataset;
    // ~Operators
    delete w;
    delete b;
    delete matmul;
    delete add;
    // ~Objectives
    delete err;
    // ~Optimizers
    delete optimizer;

    return 0;
}
