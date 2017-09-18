#ifndef OPERATOR_FROM_OPERATOR_H_
#define OPERATOR_FROM_OPERATOR_H_   value

#include <iostream>
#include "Tensor.h"
#include "Operator.h"

// class Operator {
// protected:
// Tensor m_input;
// Tensor m_output;
//
// public:
// Operator();
// virtual ~Operator();
// };

class Convolution : public Operator {
public:
    Convolution() {
        std::cout << "Convolution::Convolution() : public Operator" << '\n';
    }

    virtual ~Convolution() {
        std::cout << "Convolution::~Convolution()" << '\n';
    }
};

class MaxPooling : public Operator {
public:
    MaxPooling() {
        std::cout << "MaxPooling::MaxPooling() : public Operator" << '\n';
    }

    virtual ~MaxPooling() {
        std::cout << "MaxPooling::~MaxPooling()" << '\n';
    }
};

class MatMul : public Operator {
public:
    MatMul() {
        std::cout << "MatMul::MatMul() : public Operator" << '\n';
    }

    virtual ~MatMul() {
        std::cout << "MatMul::~MatMul()" << '\n';
    }
};

#endif  // OPERATOR_FROM_OPERATOR_H_
