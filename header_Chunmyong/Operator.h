#ifndef Operator_H_
#define Operator_H_

#include <iostream>
#include <string>
#include "Tensor.h"

// Operator 의 연산에서 필요한 요소를 구현합니다.
class Operator {
    // 멤버로 가져야 하는 친구는 노드일 것으로 예상됩니다.

private:
    // 다차원 인풋과 아웃풋을 표현할 수 있는 어떠한 형태가 필요하다.
    int m_inputDim;
    int m_outputDim;

    Tensor *m_pInput;
    Tensor *m_aOutput;
    Tensor *m_aWeight;

    // Training 과정을 공부한 후 다시 확인해야 할 부분
    Tensor *m_aGradient;
    Tensor *m_aDelta;
    Tensor *m_aDeltabar;

    // 동적 할당 및 제거
    int  Alloc();
    void Delete();

public:
    Operator() {
        std::cout << "Operator::Operator()" << '\n';
        // Alloc();
    }

    virtual ~Operator() {
        std::cout << "Operator::~Operator()" << '\n';
    }



    // Get, Set
    void Getter() const;
    void Setter();

};

#endif  // Operator_H_
