/*g++ -g -o testing -std=c++11 MLP_MSE_With_MNIST.cpp ../Header/Operator.cpp ../Header/NeuralNetwork.cpp ../Header/Tensor.cpp*/

#include <iostream>
#include <string>

#include "..//Header//NeuralNetwork.h"
#include "..//Header//Temporary_method.h"
#include "MNIST_Reader.h"

#define BATCH             100
#define LOOP_FOR_TRAIN    1000
// 10,000 is number of Test data
#define LOOP_FOR_TEST     (10000 / BATCH)

int main(int argc, char const *argv[]) {
    std::cout << "---------------Start-----------------" << '\n';

    NeuralNetwork<double> HGUNN;

    // create input, label data placeholder
    Operator<double> *x1    = HGUNN.AddPlaceholder(Tensor<double>::Constants(1, BATCH, 1, 1, 784, 1.0), "x1");
    Operator<double> *label = HGUNN.AddPlaceholder(Tensor<double>::Constants(1, BATCH, 1, 1, 10, 1.0), "label");

    // ======================= layer 1======================
    Operator<double> *w1      = new Variable<double>(Tensor<double>::Truncated_normal(1, 1, 1, 784, 15, 0.0, 0.6), "w1");
    Operator<double> *b1      = new Variable<double>(Tensor<double>::Constants(1, 1, 1, 1, 15, 1.0), "b1");
    Operator<double> *matmul1 = new MatMul<double>(x1, w1, "matmul1");
    Operator<double> *add1    = new Add<double>(matmul1, b1, "add1");
    Operator<double> *act1    = new Sigmoid<double>(add1, "sig1");

    // ======================= layer 2=======================
    Operator<double> *w2      = new Variable<double>(Tensor<double>::Truncated_normal(1, 1, 1, 15, 10, 0.0, 0.6), "w2");
    Operator<double> *b2      = new Variable<double>(Tensor<double>::Constants(1, 1, 1, 1, 10, 1.0), "b2");
    Operator<double> *matmul2 = new MatMul<double>(act1, w2, "matmul2");
    Operator<double> *add2    = new Add<double>(b2, matmul2, "add2");
    Operator<double> *act2    = new Sigmoid<double>(add2, "sig2");

    // ======================= Error=======================
    Operator<double> *err = new MSE<double>(act2, label, "MSE");

    // ======================= Optimizer=======================
    Optimizer<double> *optimizer = new GradientDescentOptimizer<double>(err, 0.5, MINIMIZE);

    // ======================= Create Graph =======================
    HGUNN.CreateGraph(optimizer);

    // ======================= Prepare Data ===================
    MNISTDataSet<double> *dataset = CreateMNISTDataSet<double>();

    // ======================= Training =======================
    for (int i = 0; i < LOOP_FOR_TRAIN; i++) {
        dataset->CreateTrainDataPair(BATCH);
        x1->FeedOutput(dataset->GetTrainFeedImage());
        label->FeedOutput(dataset->GetTrainFeedLabel());

        HGUNN.Run(optimizer);

        if ((i % 100) == 0) std::cout << "Train Accuracy is : "
                                      << (float)temp::Accuracy(act2->GetOutput(), label->GetOutput(), BATCH)
                                      << '\n';
    }

    // ======================= Testing =======================
    double test_accuracy = 0.0;

    for (int i = 0; i < (int)LOOP_FOR_TEST; i++) {
        dataset->CreateTestDataPair(BATCH);
        x1->FeedOutput(dataset->GetTestFeedImage());
        label->FeedOutput(dataset->GetTestFeedLabel());

        HGUNN.Run(err);
        // I'll implement flexibility about the situation that change of Batch size
        test_accuracy += (float)temp::Accuracy(act2->GetOutput(), label->GetOutput(), BATCH);
    }

    std::cout << "Test Accuracy is : " << test_accuracy / (int)LOOP_FOR_TEST << '\n';

    delete dataset;
    // ~Operators
    delete w1;
    delete b1;
    delete matmul1;
    delete add1;
    delete w2;
    delete b2;
    delete matmul2;
    delete add2;
    // ~Objectives
    delete err;
    // ~Optimizers
    delete optimizer;

    return 0;
}
