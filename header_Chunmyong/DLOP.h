#ifndef OPERATOR_SET_H
#define OPERATOR_SET_H    value

enum DTYPE{
    INT,
    FLOAT
}

// for basic Parent Class in DLOP
#include "NeuralNetwork.h"

// Child Class of Operator
#include "Placeholder.h"
#include "Convolution.h"
#include "Variable.h"
#include "Relu.h"
#include "Maxpooling.h"

// Child Class of Objective

// Child Class of Optimization

// Child Class of <NULL>


#endif  // OPERATOR_SET_H
