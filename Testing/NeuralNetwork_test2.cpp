#include <iostream>
#include <algorithm>

#include "..//Header//NeuralNetwork.h"

int main(int argc, char const *argv[]) {
    std::cout << "---------------Start-----------------" << '\n';

    NeuralNetwork HGUNN;

    // Tensor * var_1 = Tensor::Constants(2, {2,1}, 4.5);
    Tensor *var_1 = Tensor::Constants(5, 4, 0, 0, 0, 1.0);

    Variable x(var_1, "x");

    Tensor *var_2 = Tensor::Constants(4, 1, 0, 0, 0, 0.6);

    Variable y(var_2, "y");

    Tensor *var_3 = Tensor::Constants(5, 1, 0, 0, 0, 1.0);

    Variable z(var_3, "z");

    Tensor *var_4 = Tensor::Constants(5, 1, 0, 0, 0, 0.5);

    Variable k(var_4, "z");

    MatMul temp_1(&x, &y);

    Add temp_2(&temp_1, &z);

    // add_1->GetOutput()->PrintData();

    Relu relu_1(&temp_2, "relu_1");

    MSE temp_4(&relu_1, &k, "temp_4");

    HGUNN.SetEndOperator(&temp_4);

    for (int i = 0; i < 10; i++) {
        HGUNN.Training();
    }


    return 0;
}
