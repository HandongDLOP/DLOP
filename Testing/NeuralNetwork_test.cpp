#include <iostream>
#include <algorithm>

#include "..//Header//NeuralNetwork.h"

int main(int argc, char const *argv[]) {
    std::cout << "---------------Start-----------------" << '\n';

    NeuralNetwork HGUNN;

    Tensor * var_1 = new Tensor(2, {5, 4}, TRUNCATED_NORMAL);

    var_1->PrintData();

    Operator *x = new Variable(var_1, "x");

    Operator *copy_1 = new Copy(x, "copy_1");

    Operator *copy_2 = new Copy(copy_1, "copy_2");

    Operator *copy_3 = new Copy(copy_2, "copy_3");

    HGUNN.Training(NULL, copy_3);

    return 0;
}
