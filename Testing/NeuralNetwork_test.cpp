#include <iostream>

#include "..//Header//NeuralNetwork.h"

int main(int argc, char const *argv[]) {
	std::cout << "---------------Start-----------------" << '\n';

	Tensor * x1 = Tensor::Truncated_normal(1,1,1,2,1,0.0,0.6);

	x1->PrintData();

	x1->Reset();

	x1->PrintData();

	delete x1;
	
    std::cout << "---------------End-----------------" << '\n';
	return 0;
}
