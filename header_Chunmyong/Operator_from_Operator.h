#ifndef OPERATOR_FROM_OPERATOR_H_
#define OPERATOR_FROM_OPERATOR_H_    value

#include <iostream>

#include "Manna.h"
#include "Operator.h"

class Convolution : public Operator {
public:
    Convolution(Manna *pInput, MetaParameter *pParam, LayerType LayerType = HIDDEN) : Operator(pInput, pParam, HIDDEN) {
        Alloc(pInput, pParam);
    }

    virtual ~Convolution() {
        std::cout << "Convolution::~Convolution()" << '\n';
    }
};

class MaxPooling : public Operator {
public:
    using Operator::Operator;

    virtual ~MaxPooling() {
        std::cout << "MaxPooling::~MaxPooling()" << '\n';
    }
};

class MatMul : public Operator {
public:
    using Operator::Operator;

    virtual ~MatMul() {
        std::cout << "MatMul::~MatMul()" << '\n';
    }
};

#endif  // OPERATOR_FROM_OPERATOR_H_
