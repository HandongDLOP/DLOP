#ifndef ADD_H_
#define ADD_H_    value

#include <iostream>
#include <string>

#include "Tensor.h"
#include "Operator.h"

class Add : public Operator {
public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Add::Alloc())
    Add(Operator *pInput1, Operator *pInput2) : Operator(pInput1, pInput2) {
        std::cout << "Add::Add(Operator *, MetaParameter *)" << '\n';
        // Alloc(pInput, pParam);
    }

    virtual ~Add() {
        std::cout << "Add::~Add()" << '\n';
    }
};

#endif  // ADD_H_
