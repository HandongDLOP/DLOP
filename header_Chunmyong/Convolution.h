#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_    value

#include <iostream>
#include <string>

#include "Tensor.h"
#include "Operator.h"

class Convolution : public Operator {
public:
    Convolution(Operator *pInput, std::string name) : Operator(pInput, name) {
        std::cout << "Convolution::Convolution(pInput)" << '\n';
    }

    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Convolution::Alloc())
    Convolution(Operator *pInput, MetaParameter *pParam) : Operator(pInput, pParam) {
        // Alloc(pInput, pParam);
    }

    // do not use MetaParameter
    Convolution(Operator *pInput, Operator *pWeight) : Operator(pInput, pWeight) {
        // Alloc(pInput, pWeight);
    }

    virtual ~Convolution() {
        std::cout << "Convolution::~Convolution()" << '\n';
    }
};

#endif  // CONVOLUTION_H_
