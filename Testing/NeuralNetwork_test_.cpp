#include <iostream>

#include "..//Header//NeuralNetwork.h"

#define BATCH    4

int main(int argc, char const *argv[]) {
	std::cout << "---------------Start-----------------" << '\n';

	NeuralNetwork HGUNN;

    // create input data
    Tensor   *_x1 = Tensor::Constants(1, BATCH, 1, 1, 2, 1.0);
    Operator *x1  = HGUNN.AddPlaceholder(_x1, "x1");

    // create label data
    Tensor   *_ans = Tensor::Constants(1, BATCH, 1, 1, 2, 1.0);
    Operator *ans  = HGUNN.AddPlaceholder(_ans, "answer");

    // ======================= layer 1=======================
    Tensor   *_w1 = Tensor::Constants(1, 1, 1, 2, 2, 2.0);
    Operator *w1  = new Variable(_w1, "w1", 1);

    Tensor   *_b1 = Tensor::Constants(1, 1, 1, 1, 2, 1.0);
    Operator *b1  = new Variable(_b1, "b1", 1); // 오류 발생 원인 찾기

    Operator *mat_1 = new MatMul(x1, w1, "mat_1");

    Operator *add_1 = new Add(mat_1, b1, "add_1");

    // Operator *act_1 = new Relu(add_1, "relu_1");
    Operator *act_1 = new Sigmoid(add_1, "sig_1");

    Operator *err = new MSE(act_1, ans, "MSE");

    // ======================= Create Graph =======================

    HGUNN.CreateGraph(STOCHASTIC_GRADIENT_DESCENT, err);

    // ======================= Training =======================

    if (argc != 2) {
        std::cout << "There is no count of training" << '\n';
        return 0;
    }

    for (int i = 0; i < atoi(argv[1]); i++) {
        std::cout << "epoch : " << i << '\n';

        HGUNN.Training();
    }

    // ======================= Testing =======================

    for (int i = 0; i < 1; i++) {
        std::cout << "input : " << i << '\n';


        HGUNN.Testing();
    }

    std::cout << "---------------End-----------------" << '\n';
	return 0;
}
