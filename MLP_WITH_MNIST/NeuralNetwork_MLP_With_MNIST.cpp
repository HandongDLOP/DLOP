#include <iostream>
#include <string>

#include "..//Header//NeuralNetwork.h"
#include "MNIST_Reader.h"

#define BATCH    1

int Argmax(double *data, int Dimension) {
    int index  = 0;
    double max = data[0];

    for (int dim = 1; dim < Dimension; dim++) {
        if (data[dim] > max) {
            max   = data[dim];
            index = dim;
        }
    }

    return index;
}

double Accuracy(Operator *pred, Operator *ans) {
    double *****pred_data = pred->GetOutput()->GetData();
    double *****ans_data  = ans->GetOutput()->GetData();

    double accuracy = 0.0;

    int pred_index = 0;
    int ans_index  = 0;

    for (int ba = 0; ba < BATCH; ba++) {
        pred_index = Argmax(pred_data[0][ba][0][0], 10);
        ans_index  = Argmax(ans_data[0][ba][0][0], 10);

        // std::cout << pred_index << " " << ans_index << '\n';

        if (pred_index == ans_index) accuracy += 1.0 / BATCH;
    }

    return accuracy;
}

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
    Tensor   *_w1 = Tensor::Truncated_normal(1, 1, 1, 784, 10, 0.0, 0.6);
    // Tensor   *_w1 = Tensor::Zeros(1, 1, 1, 784, 10);
    Operator *w1  = new Variable(_w1, "w1", 1);

    Tensor   *_b1 = Tensor::Constants(1, 1, 1, 1, 10, 1.0);
    // Tensor   *_b1 = Tensor::Zeros(1, 1, 1, 1, 10);
    Operator *b1  = new Variable(_b1, "b1", 1);

    Operator *mat_1 = new MatMul(x1, w1, "mat_1");

    Operator *add_1 = new Add(mat_1, b1, "add_1");

    // Operator *act_1 = new Relu(add_1, "relu_1");
    Operator *act_1 = new Sigmoid(add_1, "sig_1");

    Operator *err = new MSE(act_1, ans, "MSE");

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
        if ((i % 10) == 0) std::cout << "epoch : " << i << '\n';
        dataset->CreateDataPair(TRAIN, BATCH, i);
        x1->FeedOutput(dataset->GetFeedImage(TRAIN));
        ans->FeedOutput(dataset->GetFeedLabel(TRAIN));

        HGUNN.Training();
        HGUNN.UpdateVariable();

        // std::cout << "pred" << '\n';
        // act_1->GetOutput()->PrintData();
        // std::cout << "ans" << '\n';
        // ans->GetOutput()->PrintData();

        if ((i % 10) == 0) std::cout << "Accuracy is : " << Accuracy(act_1, ans) << '\n';
    }

    // ======================= Testing =======================

    for (int i = 0; i < 2; i++) {
        std::cout << "\n\ninput : " << i << '\n';
        dataset->CreateDataPair(TEST, BATCH, i);
        x1->FeedOutput(dataset->GetFeedImage(TEST));
        ans->FeedOutput(dataset->GetFeedLabel(TEST));

        HGUNN.Testing();

        // std::cout << "pred" << '\n';
        // act_1->GetOutput()->PrintData();
        // std::cout << "ans" << '\n';
        // ans->GetOutput()->PrintData();

        std::cout << "Accuracy is : " << Accuracy(act_1, ans) << '\n';
    }

    delete dataset;

    std::cout << "---------------End-----------------" << '\n';
    return 0;
}
