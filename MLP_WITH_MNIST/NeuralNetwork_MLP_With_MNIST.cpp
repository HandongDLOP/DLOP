#include <iostream>
#include <string>

#include "..//Header//NeuralNetwork.h"
#include "MNIST_Reader.h"

#define BATCH    100


int main(int argc, char const *argv[]) {
    std::cout << "---------------Start-----------------" << '\n';

    NeuralNetwork HGUNN;

    // create input data placeholder
    Tensor   *_x1 = Tensor::Constants(1, BATCH, 1, 1, 784, 1.1);
    Operator *x1  = HGUNN.AddPlaceholder(_x1, "x1");

    // create label ata placeholder
    Tensor   *_ans = Tensor::Constants(1, BATCH, 1, 1, 10, 1.0);
    Operator *ans  = HGUNN.AddPlaceholder(_ans, "answer");

    // ======================= layer 1=======================
    Tensor   *_w1 = Tensor::Truncated_normal(1, 1, 1, 784, 784, 0.0, 0.6);
    Operator *w1  = new Variable(_w1, "w1", 1);

    Tensor   *_b1 = Tensor::Constants(1, 1, 1, 1, 784, 1.0);
    Operator *b1  = new Variable(_b1, "b1", 1);

    Operator *mat_1 = new MatMul(x1, w1, "mat_1");

    Operator *add_1 = new Add(mat_1, b1, "add_1");

    // Operator *act_1 = new Relu(add_1, "relu_1");
    Operator *act_1 = new Sigmoid(add_1, "sig_1");

    // ======================= layer 2=======================
    Tensor   *_w2 = Tensor::Truncated_normal(1, 1, 1, 784, 10, 0.0, 0.6);
    Operator *w2  = new Variable(_w2, "w2", 1);

    Tensor   *_b2 = Tensor::Constants(1, 1, 1, 1, 10, 1.0);
    Operator *b2  = new Variable(_b2, "b2", 1); // 오류 발생 원인 찾기

    Operator *mat_2 = new MatMul(act_1, w2, "mat_2");

    Operator *add_2 = new Add(b2, mat_2, "add_2");

    // Operator *act_2 = new Relu(add_2, "relu_2");
    Operator *act_2 = new Sigmoid(add_2, "sig_2");

    Operator *err = new MSE(act_2, ans, "MSE");

    Optimizer *optimizer = new StochasticGradientDescent(err, 1.5, MINIMIZE);

    // ======================= Create Graph =======================
    HGUNN.SetEndOperator(err);
    HGUNN.CreateGraph(optimizer);

    // ======================= Training =======================

    if (argc != 2) {
        std::cout << "There is no count of training" << '\n';
        return 0;
    }

    DataSet *dataset = CreateDataSet();

    for (int i = 0; i < atoi(argv[1]); i++) {
        // std::cout << "\n\nepoch : " << i << '\n';
        dataset->CreateDataPair(TRAIN, BATCH);
        x1->FeedOutput(dataset->GetFeedImage(TRAIN));
        ans->FeedOutput(dataset->GetFeedLabel(TRAIN));

        HGUNN.Training();
        HGUNN.UpdateVariable();
    }

    // ======================= Testing =======================

    for (int i = 0; i < 5; i++) {
        std::cout << "\n\ninput : " << i << '\n';
        dataset->CreateDataPair(TEST, BATCH);
        x1->FeedOutput(dataset->GetFeedImage(TEST));
        ans->FeedOutput(dataset->GetFeedLabel(TEST));

        HGUNN.Testing();

        act_2->GetOutput()->PrintData();
        ans->GetOutput()->PrintData();
    }

    std::cout << "---------------End-----------------" << '\n';
    return 0;
}
