#include <iostream>
#include <string>

#include "NeuralNetwork.h"


int main(int argc, char const *argv[]) {
    NeuralNetwork HGUNN;

    Tensor * var_1 = new Tensor(2, {5, 4}, TRUNCATED_NORMAL);

    var_1->PrintData();

    Operator *x = new Variable(var_1, "x");

    Operator *copy_1 = new Copy(x, "copy_1");

    Operator *copy_2 = new Copy(copy_1, "copy_2");

    HGUNN.Training(x, copy_2);


    return 0;
}
