#ifndef RELU_H_
#define RELU_H_    value

#include <iostream>
#include <string>

#include "Tensor.h"
#include "Operator.h"

class Relu : public Operator {
public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Relu::Alloc())
    Relu(Operator *pInput, std::string pName) : Operator(pInput, pName) {
        std::cout << "/* Relu::Relu(Operator *) */" << '\n';

        // Alloc(pInput);
    }


    virtual ~Relu() {
        std::cout << "Relu::~Relu()" << '\n';
    }

    virtual bool Alloc(Operator *pInput) {

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

#endif  // RELU_H_
