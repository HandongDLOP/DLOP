#include <iostream>
#include <string>

#include "..//Header//NeuralNetwork.h"
#include "MNIST_Reader.h"

#define BATCH    50

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

double Accuracy(Tensor *pred, Tensor *ans) {
    double *****pred_data = pred->GetData();
    double *****ans_data  = ans->GetData();

    double accuracy = 0.0;

    int pred_index = 0;
    int ans_index  = 0;

    for (int ba = 0; ba < BATCH; ba++) {
        pred_index = Argmax(pred_data[0][ba][0][0], 10);
        ans_index  = Argmax(ans_data[0][ba][0][0], 10);

        if (pred_index == ans_index) {
            accuracy += 1.0 / BATCH;
        } else {
            // std::cout << pred_index << '\n';
        }
    }

    return accuracy;
}

int main(int argc, char const *argv[]) {
    std::cout << "---------------Start-----------------" << '\n';

    // Data Load
    DataSet *dataset = CreateDataSet();

    NeuralNetwork HGUNN;

    // create input data placeholder
    Tensor   *_x1 = Tensor::Constants(1, BATCH, 1, 1, 784, 1.1);
    Operator *x1  = HGUNN.AddPlaceholder(_x1, "x1");

    // create label ata placeholder
    Tensor   *_ans = Tensor::Constants(1, BATCH, 1, 1, 10, 1.0);
    Operator *ans  = HGUNN.AddPlaceholder(_ans, "answer");

    // ======================= layer 1=======================
    Tensor *_w1 = Tensor::Truncated_normal(1, 1, 1, 784, 15, 0.0, 0.6);
    // Tensor   *_w1 = Tensor::Zeros(1, 1, 1, 784, 15);
    Operator *w1 = new Variable(_w1, "w1", 1);

    Tensor *_b1 = Tensor::Constants(1, 1, 1, 1, 15, 1.0);
    // Tensor   *_b1 = Tensor::Zeros(1, 1, 1, 1, 15);
    Operator *b1 = new Variable(_b1, "b1", 1);

    Operator *mat_1 = new MatMul(x1, w1, "mat_1");

    Operator *add_1 = new Add(mat_1, b1, "add_1");

    // Operator *act_1 = new Relu(add_1, "relu_1");
    Operator *act_1 = new Sigmoid(add_1, "sig_1");

    // ======================= layer 2=======================

    Tensor *_w2 = Tensor::Truncated_normal(1, 1, 1, 15, 10, 0.0, 0.6);
    // Tensor   *_w2 = Tensor::Zeros(1, 1, 1, 15, 10);
    Operator *w2 = new Variable(_w2, "w2", 1);

    Tensor *_b2 = Tensor::Constants(1, 1, 1, 1, 10, 1.0);
    // Tensor   *_b2 = Tensor::Zeros(1, 1, 1, 1, 10);
    Operator *b2 = new Variable(_b2, "b2", 1);

    Operator *mat_2 = new MatMul(act_1, w2, "mat_2");

    Operator *add_2 = new Add(b2, mat_2, "add_2");

    // Operator *act_2 = new Relu(add_2, "relu_2");
    Operator *act_2 = new Sigmoid(add_2, "sig_2");

    Operator *err = new MSE(act_2, ans, "MSE");

    // Softmax_Cross_Entropy * err = new Softmax_Cross_Entropy(add_2, ans, "MSE");

    Optimizer *optimizer = new StochasticGradientDescent(err, 0.5, MINIMIZE);

    // ======================= Create Graph =======================
    HGUNN.SetEndOperator(err);
    HGUNN.CreateGraph(optimizer);

    // ======================= Training =======================

    if (argc != 2) {
        std::cout << "There is no count of training" << '\n';
        return 0;
    }

    HGUNN.PrintGraph();

    for (int i = 0; i < atoi(argv[1]); i++) {
        if ((i % 100) == 0) std::cout << "epoch : " << i << '\n';

        dataset->CreateTrainDataPair(BATCH);
        x1->FeedOutput(dataset->GetTrainFeedImage());
        ans->FeedOutput(dataset->GetTrainFeedLabel());

        HGUNN.Training();
        HGUNN.UpdateVariable();

        // HGUNN.PrintData();

        HGUNN.PrintData(act_2);
        HGUNN.PrintData(ans);

        // if ((i % 100) == 0) std::cout << "Accuracy is : " << Accuracy(err->GetSoftmaxResult(), ans->GetOutput()) << '\n';
        if ((i % 100) == 0) std::cout << "Accuracy is : " << Accuracy(act_2->GetOutput(), ans->GetOutput()) << '\n';
    }

    // ======================= Testing =======================

    for (int i = 0; i < 2; i++) {
        std::cout << "\n\ninput : " << i << '\n';
        dataset->CreateTestDataPair(BATCH);
        x1->FeedOutput(dataset->GetTestFeedImage());
        ans->FeedOutput(dataset->GetTestFeedLabel());

        HGUNN.Testing();

        // std::cout << "pred" << '\n';
        // act_1->GetOutput()->PrintData();
        // std::cout << "ans" << '\n';
        // ans->GetOutput()->PrintData();

        // std::cout << "Accuracy is : " << Accuracy(err->GetSoftmaxResult(), ans->GetOutput()) << '\n';
        std::cout << "Accuracy is : " << Accuracy(act_2->GetOutput(), ans->GetOutput()) << '\n';
    }

    delete dataset;

    std::cout << "---------------End-----------------" << '\n';
    return 0;
}
