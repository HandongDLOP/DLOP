#ifndef PLACEHOLDER_H
#define PLACEHOLDER_H    value

#include <iostream>

#include "Tensor.h"
#include "Operator.h"

class Placeholder : public Operator {
private:
    /* data */

public:
    Placeholder() : Operator() {
        std::cout << "Placeholder::Placeholder()" << '\n';
    }

    Placeholder(std::string pName) : Operator(pName) {
        std::cout << "Placeholder::Placeholder(std::string)" << '\n';
    }

    virtual ~Placeholder() {
        std::cout << "Placeholder::~Placeholder()" << '\n';
    }
};

#endif  // PLACEHOLDER_H
