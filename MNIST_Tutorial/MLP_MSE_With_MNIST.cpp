/*g++ -g -o testing -std=c++11 MLP_MSE_With_MNIST.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp ../Header/Objective.cpp ../Header/Optimizer.cpp ../Header/NeuralNetwork.cpp*/

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
    NeuralNetwork<float> HGUNN;

    // create input, label data placeholder
    Operator<float> *x    = HGUNN.AddPlaceholder(new Placeholder<float>(Tensor<float>::Constants(1, BATCH, 1, 1, 784, 1.0), "x"));
    Operator<float> *label = HGUNN.AddPlaceholder(new Placeholder<float>(Tensor<float>::Constants(1, BATCH, 1, 1, 10, 0.f), "label"));

    // ======================= layer 1======================
    Operator<float> *w1      = HGUNN.AddTensorholder(new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, 784, 15, 0.0, 0.6), "w1"));
    Operator<float> *b1      = HGUNN.AddTensorholder(new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, 15, 1.0), "b1"));
    Operator<float> *matmul1 = HGUNN.AddOperator(new MatMul<float>(x, w1, "matmul1"));
    Operator<float> *add1    = HGUNN.AddOperator(new Addfc<float>(matmul1, b1, "add1"));
    Operator<float> *act1    = HGUNN.AddOperator(new Sigmoid<float>(add1, "relu1"));

    // ======================= layer 2=======================
    Operator<float> *w2      = HGUNN.AddTensorholder(new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, 15, 10, 0.0, 0.6), "w2"));
    Operator<float> *b2      = HGUNN.AddTensorholder(new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, 10, 1.0), "b2"));
    Operator<float> *matmul2 = HGUNN.AddOperator(new MatMul<float>(act1, w2, "matmul2"));
    Operator<float> *add2    = HGUNN.AddOperator(new Addfc<float>(matmul2, b2, "add2"));
    Operator<float> *act2    = HGUNN.AddOperator(new Sigmoid<float>(add2, "relu2"));

    // ======================= Error=======================
    Objective<float> *err = HGUNN.SetObjectiveFunction(new MSE<float>(act2, label, "MSE"));

    // ======================= Optimizer=======================
    HGUNN.SetOptimizer(new GradientDescentOptimizer<float>(err, 0.5, MINIMIZE));

    // ======================= Create Graph ======================
    HGUNN.CreateGraph();

    // ======================= Prepare Data ===================
    MNISTDataSet<float> *dataset = CreateMNISTDataSet<float>();

    for (int i = 0; i < EPOCH; i++) {
        std::cout << "EPOCH : " << i << '\n';
        // ======================= Training =======================
        double train_accuracy = 0.f;

        for (int j = 0; j < LOOP_FOR_TRAIN; j++) {
            dataset->CreateTrainDataPair(BATCH);
            x->SetResult(dataset->GetTrainFeedImage());
            label->SetResult(dataset->GetTrainFeedLabel());

            HGUNN.Training();

            train_accuracy += (float)temp::Accuracy(act2->GetResult(), label->GetResult(), BATCH);
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
            x->SetResult(dataset->GetTestFeedImage());
            label->SetResult(dataset->GetTestFeedLabel());

            HGUNN.Testing();

            test_accuracy += (float)temp::Accuracy(act2->GetResult(), label->GetResult(), BATCH);
            printf("\rTesting complete percentage is %d / %d -> acc : %f", j + 1, LOOP_FOR_TEST, test_accuracy / (j + 1));
            fflush(stdout);
        }
        std::cout << '\n';
    }
    // we need to save best weight and bias when occur best acc on test time
    delete dataset;

    return 0;
}
