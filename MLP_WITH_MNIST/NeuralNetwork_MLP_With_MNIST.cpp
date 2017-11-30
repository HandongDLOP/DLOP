/*g++ -g -o testing -std=c++11 MLP_Softmax_Cross_Entropy_With_MNIST.cpp ../Header/Operator.cpp ../Header/NeuralNetwork.cpp ../Header/Tensor.cpp*/

#include <iostream>
#include <string>

#include "..//Header//NeuralNetwork.h"
#include "..//Header//Temporary_method.h"
#include "MNIST_Reader.h"

#define BATCH    100

int main(int argc, char const *argv[]) {
    std::cout << "---------------Start-----------------" << '\n';

    // Data Load
    MNISTDataSet *dataset = CreateMNISTDataSet();

    NeuralNetwork HGUNN;

    // create input data placeholder
    Tensor   *_x1 = Tensor::Constants(1, BATCH, 1, 1, 784, 1.0);
    Operator *x1  = HGUNN.AddPlaceholder(_x1, "x1");

    // create label ata placeholder
    Tensor   *_ans = Tensor::Constants(1, BATCH, 1, 1, 10, 1.0);
    Operator *ans  = HGUNN.AddPlaceholder(_ans, "answer");

    // ======================= layer 1=======================
    // Tensor *_w1 = Tensor::Truncated_normal(1, 1, 1, 784, 10, 0.0, 0.6);
    Tensor   *_w1 = Tensor::Zeros(1, 1, 1, 784, 10);
    Operator *w1 = new Variable(_w1, "w1", 1);

    // Tensor *_b1 = Tensor::Constants(1, 1, 1, 1, 10, 1.0);
    Tensor   *_b1 = Tensor::Zeros(1, 1, 1, 1, 10);
    Operator *b1 = new Variable(_b1, "b1", 1);

    Operator *mat_1 = new MatMul(x1, w1, "mat_1");

    Operator *add_1 = new Add(mat_1, b1, "add_1");

    // Operator *act_1 = new Relu(add_1, "relu_1");
    //
    //// ======================= layer 2=======================
    //
    // Tensor *_w2 = Tensor::Truncated_normal(1, 1, 1, 15, 10, 0.0, 0.6);
    //// Tensor   *_w2 = Tensor::Zeros(1, 1, 1, 15, 10);
    // Operator *w2 = new Variable(_w2, "w2", 1);
    //
    // Tensor *_b2 = Tensor::Constants(1, 1, 1, 1, 10, 1.0);
    //// Tensor   *_b2 = Tensor::Zeros(1, 1, 1, 1, 10);
    // Operator *b2 = new Variable(_b2, "b2", 1);
    //
    // Operator *mat_2 = new MatMul(act_1, w2, "mat_2");
    //
    // Operator *add_2 = new Add(b2, mat_2, "add_2");

    // Operator *act_2 = new Relu(add_2, "relu_2");
    // Operator *act_2 = new Sigmoid(add_2, "sig_2");

    Softmax_Cross_Entropy *err = new Softmax_Cross_Entropy(add_1, ans, 1e-50, "SCE");
    // Cross_Entropy * err = new Cross_Entropy(act_1, ans, "CE");

    Optimizer *optimizer = new StochasticGradientDescent(err, 0.01, MINIMIZE);

    // ======================= Create Graph =======================
    HGUNN.SetEndOperator(err);
    HGUNN.CreateGraph(optimizer);

    // ======================= Training =======================

    int loops = 0;

    if (argc != 2) {
        std::cout << "There is no count of training" << '\n';
        loops = 1000;
    } else loops = atoi(argv[1]);

    HGUNN.PrintGraph();

    for (int i = 0; i < loops; i++) {
        if ((i % 100) == 0) std::cout << "loops : " << i << '\n';

        dataset->CreateTrainDataPair(BATCH);
        x1->FeedOutput(dataset->GetTrainFeedImage());
        ans->FeedOutput(dataset->GetTrainFeedLabel());

        HGUNN.Training();
        HGUNN.UpdateVariable();

        // err->GetSoftmaxResult()->PrintData(1);

        if ((i % 100) == 0) std::cout << "Accuracy is : " << temp::Accuracy(add_1->GetOutput(), ans->GetOutput(), BATCH) << '\n';
        // if ((i % 100) == 0) {
        //     std::cout << "cost is : " << '\n';
        //     HGUNN.PrintData(err, 1);
        //     err->GetSoftmaxResult()->PrintData(1);
        // }
    }

    // ======================= Testing =======================

    std::cout << "\n<<<Testing>>>\n" << '\n';

    double test_accuracy = 0.0;
    loops = 10000 / BATCH;

    for (int i = 0; i < loops; i++) {
        // std::cout << "\ninput : " << i << '\n';
        dataset->CreateTestDataPair(BATCH);
        x1->FeedOutput(dataset->GetTestFeedImage());
        ans->FeedOutput(dataset->GetTestFeedLabel());

        HGUNN.Testing();

        // std::cout << Accuracy(add_1->GetOutput(), ans->GetOutput()) << '\n';

        test_accuracy += temp::Accuracy(add_1->GetOutput(), ans->GetOutput(), BATCH) / (double)loops;
    }

    std::cout << "Accuracy is : " << test_accuracy << '\n';


    // HGUNN.PrintData(w1);
    // HGUNN.PrintData(b1, 1);

    delete dataset;

    std::cout << "---------------End-----------------" << '\n';
    return 0;
}
