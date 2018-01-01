/*g++ -g -o testing -std=c++11 MLP_MSE_With_MNIST.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp*/

#include <iostream>
#include <string>

#include "..//Header//DLOP.h"
#include "..//Header//Temporary_method.h"
#include "MNIST_Reader.h"

#define BATCH             100
#define LOOP_FOR_TRAIN    1000
// 10,000 is number of Test data
#define LOOP_FOR_TEST     (10000 / BATCH)

int main(int argc, char const *argv[]) {
    std::cout << "---------------Start-----------------" << '\n';

    // create input, label data placeholder
    Operator<double> *x1    = new Placeholder<double>(Tensor<double>::Constants(1, BATCH, 1, 1, 784, 1.0), "x1");
    Operator<double> *label = new Placeholder<double>(Tensor<double>::Constants(1, BATCH, 1, 1, 10, 1.0), "label");

    // ======================= layer 1======================
    Operator<double> *w1      = new Tensorholder<double>(Tensor<double>::Truncated_normal(1, 1, 1, 784, 15, 0.0, 0.6), "w1");
    Operator<double> *b1      = new Tensorholder<double>(Tensor<double>::Constants(1, 1, 1, 1, 15, 1.0), "b1");
    Operator<double> *matmul1 = new MatMul<double>(x1, w1, "matmul1");
    Operator<double> *add1    = new Add<double>(matmul1, b1, "add1");
    Operator<double> *act1    = new Sigmoid<double>(add1, "sig1");

    // ======================= layer 2=======================
    Operator<double> *w2      = new Tensorholder<double>(Tensor<double>::Truncated_normal(1, 1, 1, 15, 10, 0.0, 0.6), "w2");
    Operator<double> *b2      = new Tensorholder<double>(Tensor<double>::Constants(1, 1, 1, 1, 10, 1.0), "b2");
    Operator<double> *matmul2 = new MatMul<double>(act1, w2, "matmul2");
    Operator<double> *add2    = new Add<double>(matmul2, b2, "add2");
    Operator<double> *act2    = new Sigmoid<double>(add2, "sig2");

    // ======================= Error=======================
    Operator<double> *err = new MSE<double>(act2, label, "MSE");

    // ======================= Optimizer=======================
    Optimizer<double> *optimizer = new GradientDescentOptimizer<double>(err, 0.5, MINIMIZE);

    // ======================= Create Graph ======================

    optimizer->AddTrainableData(w1->GetResult(), w1->GetGradient());
    optimizer->AddTrainableData(b1->GetResult(), b1->GetGradient());
    optimizer->AddTrainableData(w2->GetResult(), w2->GetGradient());
    optimizer->AddTrainableData(b2->GetResult(), b2->GetGradient());

    // ======================= Prepare Data ===================
    MNISTDataSet<double> *dataset = CreateMNISTDataSet<double>();

    // ======================= Training =======================
    for (int i = 0; i < LOOP_FOR_TRAIN; i++) {
        dataset->CreateTrainDataPair(BATCH);
        x1->SetResult(dataset->GetTrainFeedImage());
        label->SetResult(dataset->GetTrainFeedLabel());

        // ForwardPropagate
        matmul1->ComputeForwardPropagate();
        add1->ComputeForwardPropagate();
        act1->ComputeForwardPropagate();
        matmul2->ComputeForwardPropagate();
        add2->ComputeForwardPropagate();
        act2->ComputeForwardPropagate();
        err->ComputeForwardPropagate();

        // BackPropagate
        err->ComputeBackPropagate();
        act2->ComputeBackPropagate();
        add2->ComputeBackPropagate();
        matmul2->ComputeBackPropagate();
        act1->ComputeBackPropagate();
        add1->ComputeBackPropagate();
        matmul1->ComputeBackPropagate();
        w1->ComputeBackPropagate();
        b1->ComputeBackPropagate();
        w2->ComputeBackPropagate();
        b2->ComputeBackPropagate();

        // UpdateVariable
        optimizer->UpdateVariable();

        if ((i % 100) == 0) std::cout << "Train Accuracy is : "
                                      << (float)temp::Accuracy(act2->GetResult(), label->GetResult(), BATCH)
                                      << '\n';
    }

    // ======================= Testing =======================
    double test_accuracy = 0.0;

    for (int i = 0; i < (int)LOOP_FOR_TEST; i++) {
        dataset->CreateTestDataPair(BATCH);
        x1->SetResult(dataset->GetTestFeedImage());
        label->SetResult(dataset->GetTestFeedLabel());

        // ForwardPropagate
        matmul1->ComputeForwardPropagate();
        add1->ComputeForwardPropagate();
        act1->ComputeForwardPropagate();
        matmul2->ComputeForwardPropagate();
        add2->ComputeForwardPropagate();
        act2->ComputeForwardPropagate();
        err->ComputeForwardPropagate();

        // I'll implement flexibility about the situation that change of Batch size
        test_accuracy += (float)temp::Accuracy(act2->GetResult(), label->GetResult(), BATCH);
    }

    std::cout << "Test Accuracy is : " << test_accuracy / (int)LOOP_FOR_TEST << '\n';

    delete dataset;
    // ~Operators
    delete w1;
    delete b1;
    delete matmul1;
    delete add1;
    delete act1;
    delete w2;
    delete b2;
    delete matmul2;
    delete add2;
    delete act2;
    // ~Objectives
    delete err;
    // ~Optimizers
    delete optimizer;

    return 0;
}
