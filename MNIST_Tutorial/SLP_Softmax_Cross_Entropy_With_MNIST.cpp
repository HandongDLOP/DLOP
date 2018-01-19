/*g++ -g -o testing -std=c++11 SLP_Softmax_Cross_Entropy_With_MNIST.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp ../Header/Objective.cpp ../Header/Optimizer.cpp ../Header/NeuralNetwork.cpp*/

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
    // Network declare
    NeuralNetwork<double> HGUNN;

    // create input, label data placeholder, placeholder is always managed by NeuralNetwork
    Operator<double> *x = HGUNN.AddPlaceholder(new Placeholder<double>(Tensor<double>::Constants(1, BATCH, 1, 1, 784, 0.0), "x"));
    Operator<double> *label = HGUNN.AddPlaceholder(new Placeholder<double>(Tensor<double>::Constants(1, BATCH, 1, 1, 10, 0.0), "label"));

    Operator<double> *w = HGUNN.AddTensorholder(new Tensorholder<double>(Tensor<double>::Zeros(1, 1, 1, 784, 10), "w"));
    Operator<double> *b = HGUNN.AddTensorholder(new Tensorholder<double>(Tensor<double>::Zeros(1, 1, 1, 1, 10), "b"));

    Operator<double> *matmul = HGUNN.AddOperator(new MatMul<double>(x, w, "matmul"));
    Operator<double> *add = HGUNN.AddOperator(new Addfc<double>(matmul, b, "add"));

    // ======================= Error=======================
    Objective<double> *err = HGUNN.SetObjectiveFunction(new SoftmaxCrossEntropy<double>(add, label, 1e-50, "SCE"));

    // ======================= Optimizer=======================
    HGUNN.SetOptimizer(new GradientDescentOptimizer<double>(err, 0.01, MINIMIZE));

    // ======================= Create Graph ===================
    HGUNN.CreateGraph();

    // ======================= Prepare Data ===================
    MNISTDataSet<double> *dataset = CreateMNISTDataSet<double>();

    // ======================= Training =======================
    for (int i = 0; i < LOOP_FOR_TRAIN; i++) {
        dataset->CreateTrainDataPair(BATCH);
        x->SetResult(dataset->GetTrainFeedImage());
        label->SetResult(dataset->GetTrainFeedLabel());

        HGUNN.Training();

        if ((i % 100) == 0) std::cout << "Train Accuracy is : "
                                      << (float)temp::Accuracy(add->GetResult(), label->GetResult(), BATCH)
                                      << '\n';
    }

    // ======================= Testing ======================
    double test_accuracy = 0.0;

    for (int i = 0; i < (int)LOOP_FOR_TEST; i++) {
        dataset->CreateTestDataPair(BATCH);
        x->SetResult(dataset->GetTestFeedImage());
        label->SetResult(dataset->GetTestFeedLabel());

        HGUNN.Testing();
        // I'll implement flexibility about the situation that change of Batch size
        test_accuracy += (float)temp::Accuracy(add->GetResult(), label->GetResult(), BATCH);
    }

    std::cout << "Test Accuracy is : " << test_accuracy / (int)LOOP_FOR_TEST << "\n\n";

    // // ======================= delete ======================
    // // ~MNISTDataSet
    delete dataset;

    return 0;
}
