#include <iostream>
#include "Tensor.h"
#include "NeuralNetwork.h"
#include "Operator.h"
#include "Activation_from_Operator.h"
#include "Operator_from_Operator.h"
#include "Objective.h"
#include "Objective_from_Objective.h"
#include "Optimization.h"

int main(int argc, char const *argv[]) {
    std::cout << "---------------Start-----------------" << '\n';

    NeuralNetwork HGUNN(10);

    // Tensor Weight = new Tensor();
    // or = new Variable();
    // Tensor Weight = new Tensor();

    HGUNN.AddOperator(new Convolution(  /*some parameter in here*/));
    // HGUNN.addOperator() = new Convolution(  /*some parameter in here*/)

    HGUNN.AddOperator(new Relu());

    HGUNN.AddOperator(new MaxPooling());

    // Tensor Reshape Method 구현

    HGUNN.AddObjective(new SoftMax());

    // delete HGUNN;

    std::cout << "---------------Done-----------------" << '\n';

    return 0;
}
