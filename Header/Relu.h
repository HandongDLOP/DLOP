#ifndef RELU_H_
#define RELU_H_    value

#include <iostream>
#include <string>

#include "Tensor.h"
#include "Operator.h"

class Relu : public Operator {
public:
    Relu() : Operator() {
        std::cout << "Relu::Relu()" << '\n';
    }

    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Relu::Alloc())
    Relu(Operator *pInput, MetaParameter *pParam) : Operator(pInput, pParam) {
        // Alloc(pInput, pParam);
    }

    // do not use MetaParameter
    Relu(Operator *pInput, Operator *pWeight) : Operator(pInput, pWeight) {
        // Alloc(pInput, pWeight);
    }

    Relu(Operator *pInput, Operator *pWeight, std::string pName) : Operator(pInput, pWeight, pName) {
        // Alloc(pInput, pWeight);
    }

    virtual ~Relu() {
        std::cout << "Relu::~Relu()" << '\n';
    }
};

#endif  // RELU_H_
