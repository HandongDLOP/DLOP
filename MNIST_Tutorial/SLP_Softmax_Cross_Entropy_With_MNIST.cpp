/*g++ -g -o testing -std=c++11 SLP_Softmax_Cross_Entropy_With_MNIST.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp ../Header/Objective.cpp ../Header/Optimizer.cpp ../Header/NeuralNetwork.cpp*/

#include <iostream>
#include <string>

#include "..//Header//NeuralNetwork.h"
#include "..//Header//Temporary_method.h"
#include "MNIST_Reader.h"

#define BATCH             100
#define EPOCH             30
#define LOOP_FOR_TRAIN    (60000 / BATCH)
// 10,000 is number of Test data
#define LOOP_FOR_TEST     (10000 / BATCH)

int main(int argc, char const *argv[]) {
    // Network declare
    NeuralNetwork<double> HGUNN;

    // create input, label data placeholder, placeholder is always managed by NeuralNetwork
    Placeholder<double> *x     = HGUNN.AddPlaceholder(new Placeholder<double>(Tensor<double>::Constants(1, BATCH, 1, 1, 784, 0.0), "x"));
    Placeholder<double> *label = HGUNN.AddPlaceholder(new Placeholder<double>(Tensor<double>::Constants(1, BATCH, 1, 1, 10, 0.0), "label"));

    Operator<double> *w = HGUNN.AddTensorholder(new Tensorholder<double>(Tensor<double>::Zeros(1, 1, 1, 784, 10), "w"));
    Operator<double> *b = HGUNN.AddTensorholder(new Tensorholder<double>(Tensor<double>::Zeros(1, 1, 1, 1, 10), "b"));

    Operator<double> *matmul = HGUNN.AddOperator(new MatMul<double>(x, w, "matmul"));
    Operator<double> *add    = HGUNN.AddOperator(new Add<double>(matmul, b, "add"));

    // ======================= Error=======================
    Objective<double> *err = HGUNN.SetObjectiveFunction(new SoftmaxCrossEntropy<double>(add, label, 1e-50, "SCE"));

    // ======================= Optimizer=======================
    HGUNN.SetOptimizer(new GradientDescentOptimizer<double>(err, 0.01, MINIMIZE));

    // ======================= Create Graph ===================
    HGUNN.CreateGraph();

    // ======================= Prepare Data ===================
    MNISTDataSet<double> *dataset = CreateMNISTDataSet<double>();

    for (int i = 0; i < EPOCH; i++) {
        std::cout << "EPOCH : " << i << '\n';
        // ======================= Training =======================
        double train_accuracy = 0.f;

        for (int j = 0; j < LOOP_FOR_TRAIN; j++) {
            dataset->CreateTrainDataPair(BATCH);
            x->SetTensor(dataset->GetTrainFeedImage());
            label->SetTensor(dataset->GetTrainFeedLabel());

            HGUNN.Training();

            train_accuracy += (float)temp::Accuracy(add->GetResult(), label->GetResult(), BATCH);
            printf("\rTraining complete percentage is %d / %d -> acc : %f", j + 1, LOOP_FOR_TRAIN, train_accuracy / (j + 1));
            fflush(stdout);
        }
        std::cout << '\n';

        // Caution!
        // Actually, we need to split training set between two set for training set and validation set
        // but in this example we do not above action.
        // ======================= Testing ======================
        double test_accuracy = 0.f;

        for (int j = 0; j < (int)LOOP_FOR_TEST; j++) {
            dataset->CreateTestDataPair(BATCH);
            x->SetTensor(dataset->GetTestFeedImage());
            label->SetTensor(dataset->GetTestFeedLabel());

            HGUNN.Testing();

            test_accuracy += (float)temp::Accuracy(add->GetResult(), label->GetResult(), BATCH);
            printf("\rTesting complete percentage is %d / %d -> acc : %f", j + 1, LOOP_FOR_TEST, test_accuracy / (j + 1));
            fflush(stdout);
        }
        std::cout << '\n';
    }
    // we need to save best weight and bias when occur best acc on test time

    // ======================= delete ======================
    // ~MNISTDataSet
    delete dataset;

    return 0;
}
