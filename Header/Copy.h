#ifndef COPY_H_
#define COPY_H_    value

#include <iostream>
#include <string>

#include "Tensor.h"
#include "Operator.h"

class Copy : public Operator {
public:
    Copy(Operator *pInput) : Operator(pInput) {
        std::cout << "Copy::Copy(Operator *, MetaParameter *)" << '\n';
        // Alloc(pInput, pParam);
    }
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Copy::Alloc())
    Copy(Operator *pInput, std::string pName) : Operator(pInput, pName) {
        std::cout << "Copy::Copy(Operator *, MetaParameter *)" << '\n';
        // Alloc(pInput, pParam);
    }

    virtual ~Copy() {
        std::cout << "Copy::~Copy()" << '\n';
    }

    virtual bool ComputeForwardPropagate(){

        std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        SetOutput(GetInput());

        GetInput()->PrintData();

        return true;
    }

};

#endif  // COPY_H_
