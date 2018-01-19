/*g++ -g -o testing -std=c++11 Convolution_With_MNIST_fc2.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp ../Header/Objective.cpp ../Header/Optimizer.cpp ../Header/NeuralNetwork.cpp*/

#include <iostream>
#include <string>

#include "..//Header//NeuralNetwork.h"
#include "..//Header//Temporary_method.h"
#include "MNIST_Reader.h"
//
#define BATCH             100
#define EPOCH             30
#define LOOP_FOR_TRAIN    (60000 / BATCH)
// 10,000 is number of Test data
#define LOOP_FOR_TEST     (10000 / BATCH)

int main(int argc, char const *argv[]) {

    NeuralNetwork<float> HGUNN;

    // create input, label data placeholder, placeholder is always managed by NeuralNetwork
    Operator<float> *x     = HGUNN.AddPlaceholder(new Placeholder<float>(Tensor<float>::Constants(1, BATCH, 1, 1, 784, 1.0), "x"));
    Operator<float> *label = HGUNN.AddPlaceholder(new Placeholder<float>(Tensor<float>::Constants(1, BATCH, 1, 1, 10, 0.0), "label"));
    Operator<float> *res   = HGUNN.AddOperator(new Reshape<float>(x, 1, BATCH, 1, 28, 28, "reshape"));

    // ======================= layer 1=======================
    Operator<float> *w1    = HGUNN.AddTensorholder(new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 10, 1, 3, 3, 0.0, 0.1), "weight"));
    Operator<float> *b1    = HGUNN.AddTensorholder(new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, 10, 0.1), "bias"));
    Operator<float> *conv1 = HGUNN.AddOperator(new Convolution2D<float>(res, w1, 1, 1, 1, 1, "convolution1"));
    Operator<float> *add1  = HGUNN.AddOperator(new Addconv<float>(conv1, b1, "addconv1"));
    Operator<float> *act1  = HGUNN.AddOperator(new Relu<float>(add1, "relu1"));
    Operator<float> *pool1 = HGUNN.AddOperator(new Maxpooling4D<float>(act1, 2, 2, 2, 2, "maxpool1"));

    // ======================= layer 2=======================
    Operator<float> *w2    = HGUNN.AddTensorholder(new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 20, 10, 3, 3, 0.0, 0.1), "weight"));
    Operator<float> *b2    = HGUNN.AddTensorholder(new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, 20, 0.1), "bias"));
    Operator<float> *conv2 = HGUNN.AddOperator(new Convolution2D<float>(pool1, w2, 1, 1, 1, 1, "convolution1"));
    Operator<float> *add2  = HGUNN.AddOperator(new Addconv<float>(conv2, b2, "addconv1"));
    Operator<float> *act2  = HGUNN.AddOperator(new Relu<float>(add2, "relu2"));
    Operator<float> *pool2 = HGUNN.AddOperator(new Maxpooling4D<float>(act2, 2, 2, 2, 2, "maxpool2"));

    // ======================= layer 3=======================
    Operator<float> *flat = HGUNN.AddOperator(new Reshape<float>(pool2, 1, BATCH, 1, 1, 5 * 5 * 20, "flat"));

    Operator<float> *w_flat1 = HGUNN.AddTensorholder(new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, 5 * 5 * 20, 250, 0.0, 0.1), "w"));
    Operator<float> *b_flat1 = HGUNN.AddTensorholder(new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, 250, 0.1), "b"));

    Operator<float> *matmul_flat1 = HGUNN.AddOperator(new MatMul<float>(flat, w_flat1, "matmul"));
    Operator<float> *add_flat1    = HGUNN.AddOperator(new Add<float>(matmul_flat1, b_flat1, "add"));

    // ======================= layer 3=======================
    Operator<float> *w_flat2 = HGUNN.AddTensorholder(new Tensorholder<float>(Tensor<float>::Truncated_normal(1, 1, 1, 250, 10, 0.0, 0.1), "w"));
    Operator<float> *b_flat2 = HGUNN.AddTensorholder(new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, 10, 0.1), "b"));

    Operator<float> *matmul_flat2 = HGUNN.AddOperator(new MatMul<float>(add_flat1, w_flat2, "matmul"));
    Operator<float> *add_flat2    = HGUNN.AddOperator(new Add<float>(matmul_flat2, b_flat2, "add"));

    // ======================= Error=======================
    Objective<float> *err = HGUNN.SetObjectiveFunction(new SoftmaxCrossEntropy<float>(add_flat2, label, 0.0000001, "SCE")); // 중요 조건일 가능성 있음

    // ======================= Optimizer=======================
    HGUNN.SetOptimizer(new GradientDescentOptimizer<float>(err, 0.001, MINIMIZE));

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

            train_accuracy += (float)temp::Accuracy(add_flat2->GetResult(), label->GetResult(), BATCH);
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

            test_accuracy += (float)temp::Accuracy(add_flat2->GetResult(), label->GetResult(), BATCH);
            printf("\rTesting complete percentage is %d / %d -> acc : %f", j + 1, LOOP_FOR_TEST, test_accuracy / (j + 1));
            fflush(stdout);
        }
        std::cout << '\n';
    }
    // we need to save best weight and bias when occur best acc on test time


    delete dataset;

    return 0;
}
