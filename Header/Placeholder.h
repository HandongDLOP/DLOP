#ifndef PLACEHOLDER_H_
#define PLACEHOLDER_H_    value

#include <iostream>

#include "Tensor.h"
#include "Operator.h"

class Placeholder : public Operator {
private:
public:
    Placeholder(std::string pName) : Operator(pName) {
        std::cout << "Placeholder::Placeholder(std::string)" << '\n';
    }

    Placeholder(Tensor *pTensor, std::string pName) : Operator(pTensor, pName) {
        std::cout << "Placeholder::Placeholder(Tensor *, std::string)" << '\n';

        Alloc(pTensor);
    }

    virtual ~Placeholder() {
        std::cout << "Placeholder::~Placeholder()" << '\n';
    }

    virtual bool Alloc(Tensor *pTensor) {
        SetOutput(pTensor);

        // no meaning
        Tensor *delta = new Tensor(pTensor->GetShape());
        SetDelta(delta);

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        return true;
    }

    virtual bool ComputeBackPropagate() {
        // std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        return true;
    }
};



#endif  // PLACEHOLDER_H_
