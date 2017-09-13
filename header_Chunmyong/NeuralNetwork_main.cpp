#include <iostream>
#include "Tensor.h"
#include "NeuralNetwork.h"
#include "Layer.h"
#include "Activation.h"
#include "Optimization.h"
#include "Operator.h"
#include "Objective.h"

int main(int argc, char const *argv[]) {
    std::cout << "---------------Start-----------------" << '\n';

    NeuralNetwork *HGUNN = new NeuralNetwork(3);

    // Tensor Weight = new Tensor();
    // or = new Variable();
    // Tensor Weight = new Tensor();

    HGUNN->CreateLayer(new Convolution(  /*some parameter in here*/));

    HGUNN->CreateLayer(new Relu());

    HGUNN->CreateLayer(new MaxPooling());
    //
    // HGUNN->CreateLayer(new Convolution());
    //
    // HGUNN->CreateLayer(new Relu());
    //
    // HGUNN->CreateLayer(new MaxPooling());
    //
    // // Tensor Reshape Method 구현
    //
    // HGUNN->CreateLayer(new MatMul());
    //
    // HGUNN->CreateLayer(new Relu());
    //
    // HGUNN->CreateLayer(new MatMul());
    //
    // HGUNN->CreateLayer(new SoftMax());

    delete HGUNN;

    std::cout << "---------------Done-----------------" << '\n';

    return 0;
}
