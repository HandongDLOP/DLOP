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

    NeuralNetwork HGUNN;

    // create input, label data placeholder
    Operator *x1  = HGUNN.AddPlaceholder(Tensor::Constants(1, BATCH, 1, 1, 784, 1.0), "x1");
    Operator *label = HGUNN.AddPlaceholder(Tensor::Constants(1, BATCH, 1, 1, 10, 1.0), "label");

    // ======================= layer 1======================
    Operator *w1    = new Variable(Tensor::Truncated_normal(1, 1, 1, 784, 15, 0.0, 0.6), "w1");
    Operator *b1    = new Variable(Tensor::Constants(1, 1, 1, 1, 15, 1.0), "b1");
    Operator *matmul1 = new MatMul(x1, w1, "matmul1");
    Operator *add1 = new Add(matmul1, b1, "add1");
    Operator *act1 = new Sigmoid(add1, "sig1");

    // ======================= layer 2=======================
    Operator *w2    = new Variable(Tensor::Truncated_normal(1, 1, 1, 15, 10, 0.0, 0.6), "w2");
    Operator *b2    = new Variable(Tensor::Constants(1, 1, 1, 1, 10, 1.0), "b2");
    Operator *matmul2 = new MatMul(act1, w2, "matmul2");
    Operator *add2 = new Add(b2, matmul2, "add2");
    Operator *act2 = new Sigmoid(add2, "sig2");
    Operator *err   = new MSE(act2, label, "MSE");

    // ======================= Optimizer=======================
    Optimizer *optimizer = new GradientDescentOptimizer(err, 0.5, MINIMIZE);

    // ======================= Create Graph =======================
    HGUNN.CreateGraph(optimizer);

    // ======================= Prepare Data ===================
    MNISTDataSet *dataset = CreateMNISTDataSet();

    // ======================= Training =======================
    HGUNN.PrintGraph(optimizer);

    for (int i = 0; i < LOOP_FOR_TRAIN; i++) {
        dataset->CreateTrainDataPair(BATCH);
        x1->FeedOutput(dataset->GetTrainFeedImage());
        label->FeedOutput(dataset->GetTrainFeedLabel());

        HGUNN.Run(optimizer);

        if ((i % 100) == 0) std::cout << "Accuracy is : " << temp::Accuracy(act2->GetOutput(), label->GetOutput(), BATCH) << '\n';
    }

    // ======================= Testing =======================
    double test_accuracy = 0.0;

    for (int i = 0; i < (int)LOOP_FOR_TEST; i++) {
        dataset->CreateTestDataPair(BATCH);
        x1->FeedOutput(dataset->GetTestFeedImage());
        label->FeedOutput(dataset->GetTestFeedLabel());

        HGUNN.Run(err);
        // I'll implement flexibility about the situation that change of Batch size
        test_accuracy += temp::Accuracy(act2->GetOutput(), label->GetOutput(), BATCH);
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
