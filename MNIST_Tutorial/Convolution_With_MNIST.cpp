/*g++ -g -o testing -std=c++11 Convolution_With_MNIST.cpp ../Header/Operator.cpp ../Header/NeuralNetwork.cpp ../Header/Tensor.cpp*/

#include <iostream>
#include <string>

#include "..//Header//NeuralNetwork.h"
#include "..//Header//Temporary_method.h"
#include "MNIST_Reader.h"
//
#define BATCH             1
#define LOOP_FOR_TRAIN    1
// 10,000 is number of Test data
#define LOOP_FOR_TEST     (10000 / BATCH)

int main(int argc, char const *argv[]) {
    // create input, label data placeholder, placeholder is always managed by NeuralNetwork
    Operator<float> *x     = new Placeholder<float>(Tensor<float>::Constants(1, BATCH, 1, 1, 784, 1.0), "x");
    Operator<float> *label = new Placeholder<float>(Tensor<float>::Constants(1, BATCH, 1, 1, 10, 0.0), "label");
    Operator<float> *res   = new Reshape<float>(x, 1, 28, 28, "reshape");

    // ======================= layer 1=======================
    Operator<float> *w1    = new Variable<float>(Tensor<float>::Constants(1, 10, 1, 3, 3, 1.0), "weight");
    Operator<float> *b1    = new Variable<float>(Tensor<float>::Constants(1, 1, 1, 1, 10, 1.0), "bias");
    Operator<float> *conv1 = new Convolution<float>(res, w1, 1, 1, 1, 1, "convolution1");
    Operator<float> *add1  = new Addconv<float>(conv1, b1, "addconv1");
    Operator<float> *act1  = new Relu<float>(add1, "relu");

    // ======================= layer 2=======================
    Operator<float> *w2    = new Variable<float>(Tensor<float>::Constants(1, 10, 10, 3, 3, 1.0), "weight");
    Operator<float> *b2    = new Variable<float>(Tensor<float>::Constants(1, 1, 1, 1, 10, 1.0), "bias");
    Operator<float> *conv2 = new Convolution<float>(act1, w2, 1, 1, 1, 1, "convolution1");
    Operator<float> *add2  = new Addconv<float>(conv2, b2, "addconv1");
    Operator<float> *act2  = new Relu<float>(add2, "relu");

    // ======================= layer 3=======================
    Operator<float> *flat = new Reshape<float>(act2, 1, 1, 24 * 24 * 10, "flat");

    Operator<float> *w_flat = new Variable<float>(Tensor<float>::Truncated_normal(1, 1, 1, 24 * 24 * 10, 10, 0.0, 0.6), "w");
    Operator<float> *b_flat = new Variable<float>(Tensor<float>::Zeros(1, 1, 1, 1, 10), "b");

    Operator<float> *matmul = new MatMul<float>(flat, w_flat, "matmul");
    Operator<float> *add    = new Add<float>(matmul, b_flat, "add");

    // ======================= Error=======================
    Operator<float> *err = new SoftmaxCrossEntropy<float>(add, label, 0.2, "SCE");

    // ======================= Optimizer=======================
    Optimizer<float> *optimizer = new GradientDescentOptimizer<float>(err, 0.01, MINIMIZE);

    optimizer->AddTrainableData(w1->GetOutput(),     w1->GetGradient());
    optimizer->AddTrainableData(b1->GetOutput(),     b1->GetGradient());

    optimizer->AddTrainableData(w2->GetOutput(),     w2->GetGradient());
    optimizer->AddTrainableData(b2->GetOutput(),     b2->GetGradient());

    optimizer->AddTrainableData(w_flat->GetOutput(), w_flat->GetGradient());
    optimizer->AddTrainableData(b_flat->GetOutput(), b_flat->GetGradient());

    // ======================= Train=======================
    MNISTDataSet<float> *dataset = CreateMNISTDataSet<float>();
    dataset->CreateTrainDataPair(BATCH);

    for (int i = 0; i < LOOP_FOR_TRAIN; i++) {
        dataset->CreateTrainDataPair(BATCH);
        x->FeedOutput(dataset->GetTrainFeedImage());
        label->FeedOutput(dataset->GetTrainFeedLabel());

        // ======================= Forward=======================
        res->ComputeForwardPropagate();
        conv1->ComputeForwardPropagate();
        add1->ComputeForwardPropagate();
        act1->ComputeForwardPropagate();

        conv2->ComputeForwardPropagate();
        add2->ComputeForwardPropagate();
        act2->ComputeForwardPropagate();

        flat->ComputeForwardPropagate();
        matmul->ComputeForwardPropagate();
        add->ComputeForwardPropagate();
        err->ComputeForwardPropagate();

        // // ======================= Backward=======================
        // err->ComputeBackPropagate();
        // add->ComputeBackPropagate();
        // matmul->ComputeBackPropagate();
        // flat->ComputeBackPropagate();
        //
        // act2->ComputeBackPropagate();
        // add2->ComputeBackPropagate();
        // conv2->ComputeBackPropagate();
        //
        // act1->ComputeBackPropagate();
        // add1->ComputeBackPropagate();
        // conv1->ComputeBackPropagate();
        // res->ComputeBackPropagate();
        //
        // b_flat->ComputeBackPropagate();
        // w_flat->ComputeBackPropagate();
        // b2->ComputeBackPropagate();
        // w2->ComputeBackPropagate();
        // b1->ComputeBackPropagate();
        // w1->ComputeBackPropagate();
        //
        // // ======================= Update=======================
        // optimizer->UpdateVariable();


        err->GetOutput()->PrintData(1);
        err->GetOutput()->PrintShape();
        add->GetOutput()->PrintData(1);
        add->GetOutput()->PrintShape();
        matmul->GetOutput()->PrintShape();
        flat->GetOutput()->PrintShape();
        act2->GetOutput()->PrintShape();
        add2->GetOutput()->PrintShape();
        conv2->GetOutput()->PrintShape();
        act1->GetOutput()->PrintShape();
        add1->GetOutput()->PrintShape();
        conv1->GetOutput()->PrintShape();
        res->GetOutput()->PrintShape();
        x->GetOutput()->PrintShape();

        // HGUNN.Run(optimizer);

        // if ((i % 100) == 0) std::cout << "Train Accuracy is : "
        // << (float)temp::Accuracy(add->GetOutput(), label->GetOutput(), BATCH)
        // << '\n';
    }

    // ======================= Testing =======================
    // double test_accuracy = 0.0;

    // for (int i = 0; i < (int)LOOP_FOR_TEST; i++) {
    // dataset->CreateTestDataPair(BATCH);
    // x->FeedOutput(dataset->GetTestFeedImage());
    // label->FeedOutput(dataset->GetTestFeedLabel());
    //
    //// HGUNN.Run(err);
    //// I'll implement flexibility about the situation that change of Batch size
    // test_accuracy += (float)temp::Accuracy(add->GetOutput(), label->GetOutput(), BATCH);
    // }
    //
    // std::cout << "Test Accuracy is : " << test_accuracy / (int)LOOP_FOR_TEST << '\n';


    return 0;
}
