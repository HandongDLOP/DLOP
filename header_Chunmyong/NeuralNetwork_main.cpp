#include <iostream>
#include "Manna.h"
#include "NeuralNetwork.h"
#include "Operator.h"
#include "Activation_from_Operator.h"
#include "Operator_from_Operator.h"
#include "Objective.h"
#include "Optimization.h"

int main(int argc, char const *argv[]) {
    std::cout << "---------------Start-----------------" << '\n';

    NeuralNetwork HGUNN;

    // Manna Weight = new Manna();
    // or = new Variable();
    // Manna Weight = new Manna();

    MetaParameter * pConvParam = new ConvParam();
    HGUNN.PutOperator("Convolution", pConvParam);


    // HGUNN.addOperator() = new Convolution(  /*some parameter in here*/)



    // delete HGUNN;

    std::cout << "---------------Done-----------------" << '\n';

    return 0;
}
