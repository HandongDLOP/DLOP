/*g++ -g -o testing -std=c++11 MLP_Softmax_Cross_Entropy_With_MNIST.cpp ../Header/Operator.cpp ../Header/NeuralNetwork.cpp ../Header/Tensor.cpp*/

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
    NeuralNetwork HGUNN;

    // create input, label data placeholder
    Operator *x     = HGUNN.AddPlaceholder(Tensor::Constants(1, BATCH, 1, 1, 784, 0.0), "x");
    Operator *label = HGUNN.AddPlaceholder(Tensor::Constants(1, BATCH, 1, 1, 10, 0.0), "label");

    // ======================= layer 1=======================
    Operator *w                = new Variable(Tensor::Zeros(1, 1, 1, 784, 10), "w");
    Operator *b                = new Variable(Tensor::Zeros(1, 1, 1, 1, 10), "b");
    Operator *matmul           = new MatMul(x, w, "matmul");
    Operator *add              = new Add(matmul, b, "add");
    Softmax_Cross_Entropy *err = new Softmax_Cross_Entropy(add, label, 1e-50, "SCE");

    // ======================= Optimizer=======================
    Optimizer *optimizer = new StochasticGradientDescent(err, 0.01, MINIMIZE);

    // ======================= Create Graph ===================
    HGUNN.SetEndOperator(err);
    HGUNN.CreateGraph(optimizer);

    // ======================= Prepare Data ===================
    MNISTDataSet *dataset = CreateMNISTDataSet();

    // ======================= Training =======================
    HGUNN.PrintGraph();

    for (int i = 0; i < LOOP_FOR_TRAIN; i++) {
        dataset->CreateTrainDataPair(BATCH);
        x->FeedOutput(dataset->GetTrainFeedImage());
        label->FeedOutput(dataset->GetTrainFeedLabel());

        HGUNN.Training();
        HGUNN.UpdateVariable();

        if ((i % 100) == 0) std::cout << "Accuracy is : " << temp::Accuracy(add->GetOutput(), label->GetOutput(), BATCH) << '\n';
    }

    // ======================= Testing ======================
    double test_accuracy = 0.0;

    for (int i = 0; i < (int)LOOP_FOR_TEST; i++) {
        dataset->CreateTestDataPair(BATCH);
        x->FeedOutput(dataset->GetTestFeedImage());
        label->FeedOutput(dataset->GetTestFeedLabel());

        HGUNN.Testing();
        // I'll implement flexibility about the situation that change of Batch size
        test_accuracy += temp::Accuracy(add->GetOutput(), label->GetOutput(), BATCH) / (int)LOOP_FOR_TEST;
    }

    std::cout << "Test Accuracy is : " << test_accuracy << "\n\n";

    delete dataset;

    return 0;
}
