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

    Placeholder(TensorShape *pshape) : Operator(pshape){
        std::cout << "Placeholder::Placeholder(TensorShape *)" << '\n';
        Alloc(pshape);
    }

    Placeholder(TensorShape *pshape, std::string pName) : Operator(pshape, pName) {
        std::cout << "Placeholder::Placeholder(TensorShape *, std::string)" << '\n';
        Alloc(pshape);
    }

    virtual ~Placeholder() {
        std::cout << "Placeholder::~Placeholder()" << '\n';
    }

    virtual bool Alloc(TensorShape *pshape) {
        Tensor *temp_output = new Tensor(pshape);

        SetOutput(temp_output);

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';


        return true;
    }

    virtual bool ComputeBackPropagate() {
        std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        return true;
    }
};

#endif  // PLACEHOLDER_H
