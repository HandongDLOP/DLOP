#ifndef ACTIVATION_FROM_OPERATOR_H_
#define ACTIVATION_FROM_OPERATOR_H_    value

#include <string>
#include "Tensor.h"
#include "Operator.h"

class Relu : public Operator {
public:
    Relu() {
        std::cout << "Relu::Relu() : public Operator" << '\n';
    }

    virtual ~Relu() {
        std::cout << "Relu::~Relu()" << '\n';
    }
};

class Sigmoid : public Operator {
public:
    Sigmoid() {
        std::cout << "Sigmoid::Sigmoid() : public Operator" << '\n';
    }

    virtual ~Sigmoid() {
        std::cout << "Sigmoid::~Sigmoid()" << '\n';
    }
};


#endif  // ACTIVATION_FROM_OPERATOR_H_
