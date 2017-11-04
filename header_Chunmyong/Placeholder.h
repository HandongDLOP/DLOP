#ifndef PLACEHOLDER_H
#define PLACEHOLDER_H    value

#include <iostream>

#include "Tensor.h"
#include "Operator.h"

class placeholder : public Operator {
private:
    /* data */

public:
    placeholder(std::string name) : Operator(name) {
        std::cout << "placeholder::placeholder()" << '\n';
    }

    virtual ~placeholder() {
        std::cout << "placeholder::~placeholder()" << '\n';
    }
};

#endif  // PlaceHOLDER_H
