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
	_ans->GetData()[1] = 1;
	_ans->GetData()[2] = 0;
	Operator *ans = new Variable(_ans, "answer", 0);

	// ======================= layer 1=======================
    Tensor *_w1 = Tensor::Truncated_normal(3, 3, 0, 0, 0, 0.0, 0.6);
    Operator *w1 = new Variable(_w1, "w1", 1);

    Tensor *_b1 = Tensor::Constants(1, 3, 0, 0, 0, 1.0);
    Operator *b1 = new Variable(_b1, "b1", 1); // 오류 발생 원인 찾기

	Operator *mat_1 = new MatMul(x1, w1);

	Operator *add_1 = new Add(mat_1, b1);

	Operator *relu_1 = new Relu(add_1, "relu_1");

	// ======================= layer 2=======================
	Tensor *_w2 = Tensor::Truncated_normal(3, 3, 0, 0, 0, 0.0, 0.6);
	Operator *w2 = new Variable(_w2, "w2", 1);

	Tensor *_b2 = Tensor::Constants(1, 3, 0, 0, 0, 1.0);
	Operator *b2 = new Variable(_b2, "b2", 1);     // 오류 발생 원인 찾기

	Operator *mat_2 = new MatMul(relu_1, w2);

	Operator *add_2 = new Add(mat_2, b2);

	Operator *relu_2 = new Relu(add_2, "relu_2");

	Operator *err = new MSE(relu_2, ans, "MSE");

	// 이 작업들을 추후에는 모두 create graph에서 실행해야 한다.
	HGUNN.SetEndOperator(err);
	HGUNN.AllocOptimizer(STOCHASTIC_GRADIENT_DESCENT);

	for (int i = 0; i < 100; i++) {
		std::cout << "epoch : "<< i << '\n';
		HGUNN.Training();
	}

    std::cout << "---------------End-----------------" << '\n';
	return 0;
}
