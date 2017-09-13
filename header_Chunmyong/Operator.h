#ifndef OPERATOR_H_
#define OPERATOR_H_    value

#include <iostream>
#include "Tensor.h"
#include "Layer.h"

// class Operator {
// protected:
// Tensor m_input;
// Tensor m_output;
//
// public:
// Operator();
// virtual ~Operator();
// };

class Convolution : public Layer {
public:
    Convolution() {
        std::cout << "Convolution::Convolution() : public Layer" << '\n';
    }

    virtual ~Convolution() {
        std::cout << "Convolution::~Convolution()" << '\n';
    }
};

class MaxPooling : public Layer {
public:
    MaxPooling() {
        std::cout << "MaxPooling::MaxPooling() : public Layer" << '\n';
    }

    virtual ~MaxPooling() {
        std::cout << "MaxPooling::~MaxPooling()" << '\n';
    }
};

class MatMul : public Layer {
public:
    MatMul() {
        std::cout << "MatMul::MatMul() : public Layer" << '\n';
    }

    virtual ~MatMul() {
        std::cout << "MatMul::~MatMul()" << '\n';
    }
};

#endif  // OPERATOR_H_
