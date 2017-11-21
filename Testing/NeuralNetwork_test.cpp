#include <iostream>
#include <algorithm>

#include "..//Header//NeuralNetwork.h"

int main(int argc, char const *argv[]) {
	std::cout << "---------------Start-----------------" << '\n';

	NeuralNetwork HGUNN;

	// create input data
	Tensor *_x1 = Tensor::Constants(1, 3, 0, 0, 0, 1.0);
	Operator *x1 = new Variable(_x1, "x1", 0);

	// create label data
	Tensor *_ans = Tensor::Constants(1, 3, 0, 0, 0, 0.0);
	_ans->GetData()[0] = 0;
	_ans->GetData()[1] = 0;
	_ans->GetData()[2] = 0;
	Operator *ans = new Variable(_ans, "answer", 0);

	// ======================= layer 1=======================
    Tensor *_w1 = Tensor::Constants(3, 3, 0, 0, 0, 2.0);
    Operator *w1 = new Variable(_w1, "w1", 1);

    Tensor *_b1 = Tensor::Constants(1, 3, 0, 0, 0, 1.0);
    Operator *b1 = new Variable(_b1, "b1", 1); // 오류 발생 원인 찾기

	Operator *mat_1 = new MatMul(x1, w1);

	Operator *add_1 = new Add(mat_1, b1);

	Operator *sig_1 = new Sigmoid(add_1, "sig_1");

	Operator *err = new MSE(sig_1, ans, "MSE");

	HGUNN.CreateGraph(STOCHASTIC_GRADIENT_DESCENT, err);

	for (int i = 0; i < 1; i++) {
		std::cout << "epoch : "<< i << '\n';
		HGUNN.Training();
	}

    std::cout << "---------------End-----------------" << '\n';
	return 0;
}
