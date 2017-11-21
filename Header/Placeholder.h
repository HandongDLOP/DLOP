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

    Placeholder(Tensor *pTensor, std::string pName) : Operator(pTensor, pName) {
        std::cout << "Placeholder::Placeholder(Tensor *, std::string)" << '\n';
        Alloc(pTensor);
    }

    virtual ~Placeholder() {
        std::cout << "Placeholder::~Placeholder()" << '\n';
    }

    virtual bool Alloc(TensorShape *pshape) {
        Tensor *temp_output = new Tensor(pshape);

        SetOutputDim(pshape);

        SetOutput(temp_output);

        return true;
    }

    virtual bool Alloc(Tensor *pTensor) {

        SetOutputDim(pTensor->Getshape());

        SetOutput(pTensor);

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
