#ifndef VARIABLE_H_
#define VARIABLE_H_    value

#include <iostream>
#include <string>

#include "Tensor.h"
#include "Operator.h"

class Variable : public Operator {
public:
    Variable() : Operator() {
        std::cout << "Variable::Variable()" << '\n';
    }

    Variable(std::string pName) : Operator(pName) {
        std::cout << "Variable::Variable(std::string)" << '\n';
    }

    virtual ~Variable() {
        std::cout << "Variable::~Variable()" << '\n';
    }
};

#endif  // VARIABLE_H_
