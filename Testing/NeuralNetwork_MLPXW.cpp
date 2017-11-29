#include <iostream>
#include <string>

#include "..//Header//NeuralNetwork.h"

#define BATCH    100

// 데이터 전처리
Tensor* CreateInput() {
    // std::cout << "CreatrInput()" << '\n';
    double *****data_input = NULL;
    int *shape_input       = new int[5] { 1, BATCH, 1, 1, 2 };
    int  rank_input        = 5;

    data_input    = new double ****[1];
    data_input[0] = new double ***[BATCH];

    for (int i = 0; i < BATCH; i++) {
        data_input[0][i]    = new double **[1];
        data_input[0][i][0] = new double *[1];

        if (i % 4 == 0) data_input[0][i][0][0] = new double[2] { 0, 0 };
        else if (i % 4 == 1) data_input[0][i][0][0] = new double[2] { 1, 0 };
        else if (i % 4 == 2) data_input[0][i][0][0] = new double[2] { 0, 1 };
        else if (i % 4 == 3) data_input[0][i][0][0] = new double[2] { 1, 1 };
    }

    return new Tensor(data_input, shape_input, rank_input);
}

Tensor* CreateLabel() {
    // std::cout << "CreatrLabel()" << '\n';
    double *****data_label = NULL;
    int *shape_label       = new int[5] { 1, BATCH, 1, 1, 2 };
    int  rank_label        = 5;

    data_label    = new double ****[1];
    data_label[0] = new double ***[BATCH];

    for (int i = 0; i < BATCH; i++) {
        data_label[0][i]    = new double **[1];
        data_label[0][i][0] = new double *[1];

        if (i % 4 == 0) data_label[0][i][0][0] = new double[2] { 1, 0 };
        else if (i % 4 == 1) data_label[0][i][0][0] = new double[2] { 0, 1 };
        else if (i % 4 == 2) data_label[0][i][0][0] = new double[2] { 0, 1 };
        else if (i % 4 == 3) data_label[0][i][0][0] = new double[2] { 1, 0 };
    }

    return new Tensor(data_label, shape_label, rank_label);
}

int Argmax(double *data, int Dimension) {
    int index  = 0;
    double max = data[0];

    for (int dim = 1; dim < Dimension; dim++) {
        if (data[dim] > max) {
            max   = data[dim];
            index = dim;
        }
        // std::cout << data[dim] << '\n';
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
        pred_index = Argmax(pred_data[0][ba][0][0], 2);
        ans_index  = Argmax(ans_data[0][ba][0][0], 2);

        // std::cout << pred_index << " " << ans_index << '\n';

        if (pred_index == ans_index) accuracy += 1.0 / BATCH;
    }

    return accuracy;
}

int main(int argc, char const *argv[]) {
    std::cout << "---------------Start-----------------" << '\n';

    NeuralNetwork HGUNN;

    // create input data placeholder
    Tensor   *_x1 = Tensor::Constants(1, BATCH, 1, 1, 2, 1.0);
    Operator *x1  = HGUNN.AddPlaceholder(_x1, "x1");

    // create label ata placeholder
    Tensor   *_ans = Tensor::Constants(1, BATCH, 1, 1, 2, 1.0);
    Operator *ans  = HGUNN.AddPlaceholder(_ans, "answer");

    // ======================= layer 1=======================
    Tensor   *_w1 = Tensor::Truncated_normal(1, 1, 1, 2, 4, 0.0, 0.6);
    Operator *w1  = new Variable(_w1, "w1", 1);

    Tensor   *_b1 = Tensor::Constants(1, 1, 1, 1, 4, 1.0);
    Operator *b1  = new Variable(_b1, "b1", 1); // 오류 발생 원인 찾기

    Operator *mat_1 = new MatMul(x1, w1, "mat_1");

    Operator *add_1 = new Add(mat_1, b1, "add_1");

    // Operator *act_1 = new Relu(add_1, "relu_1");
    Operator *act_1 = new Sigmoid(add_1, "sig_1");

    // ======================= layer 2=======================
    Tensor   *_w2 = Tensor::Truncated_normal(1, 1, 1, 4, 2, 0.0, 0.6);
    Operator *w2  = new Variable(_w2, "w2", 1);

    Tensor   *_b2 = Tensor::Constants(1, 1, 1, 1, 2, 1.0);
    Operator *b2  = new Variable(_b2, "b2", 1); // 오류 발생 원인 찾기

    Operator *mat_2 = new MatMul(act_1, w2, "mat_2");

    Operator *add_2 = new Add(b2, mat_2, "add_2");

    // Operator *act_2 = new Relu(add_2, "relu_2");
    Operator *act_2 = new Sigmoid(add_2, "sig_2");

    Operator *err = new MSE(act_2, ans, "MSE");

    Optimizer *optimizer = new StochasticGradientDescent(err, 0.6, MINIMIZE);

    // Operator *err1 = new MSE(err, ans, "MSE_1");
    //
    // Operator *err2 = new MSE(err, ans, "MSE_2");

    // ======================= Create Graph =======================
    HGUNN.SetEndOperator(err);
    // HGUNN.SetEndOperator(err1);
    // HGUNN.SetEndOperator(err2);
    HGUNN.CreateGraph(optimizer);

    // ======================= Training =======================

    if (argc != 2) {
        std::cout << "There is no count of training" << '\n';
        return 0;
    }

    HGUNN.PrintGraph();

    for (int i = 0; i < atoi(argv[1]); i++) {
        // for (int i = 0; i < 1; i++) {
        if ((i % 100) == 0) std::cout << "epoch : " << i << '\n';
        x1->FeedOutput(CreateInput());
        ans->FeedOutput(CreateLabel());

        HGUNN.Training();
        HGUNN.UpdateVariable();

        // HGUNN.PrintData();
    }

    // ======================= Testing =======================

    for (int i = 0; i < 5; i++) {
        std::cout << "\ninput : " << i << '\n';
        x1->FeedOutput(CreateInput());
        ans->FeedOutput(CreateLabel());

        HGUNN.Testing();

        // act_2->GetOutput()->PrintData();
        // ans->GetOutput()->PrintData();

        std::cout << "Accuracy is : " << Accuracy(act_2->GetOutput(), ans->GetOutput()) << '\n';
    }

    std::cout << "---------------End-----------------" << '\n';
    return 0;
}
