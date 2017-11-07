#ifndef VARIABLE_H_
#define VARIABLE_H_    value

#include <iostream>
#include <string>

#include "Tensor.h"
#include "Operator.h"

class Variable : public Operator {
public:
    Variable(std::string pName) : Operator(pName) {
    std::cout << "Variable::Variable(std::string)" << '\n';
    }

    Variable(Tensor *pTensor, std::string pName) : Operator(pTensor, pName) {
        std::cout << "Variable::Variable(Tensor *, std::string)" << '\n';

        Alloc(pTensor);
    }

    virtual ~Variable() {
        std::cout << "Variable::~Variable()" << '\n';
    }

    virtual bool Alloc(Tensor *pTensor) {
        SetOutputDim(pTensor->Getshape());

        SetOutput(pTensor);

        Tensor * temp_Gradient = new Tensor(pTensor);

        SetGradient(temp_Gradient);

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

#endif  // VARIABLE_H_
