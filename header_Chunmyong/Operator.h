#ifndef OPERATOR_H_
#define OPERATOR_H_

#include <iostream>
#include <string>
#include "Tensor.h"

// Operator 의 연산에서 필요한 요소를 구현합니다.
class Operator {
    // 멤버로 가져야 하는 친구는 노드일 것으로 예상됩니다.

private:
    // 다차원 인풋과 아웃풋을 표현할 수 있는 어떠한 형태가 필요하다.
    // Tensor class의 GetDim을 이용하기로 합니다.
    Tensor m_InputDim;
    Tensor m_OutputDim;

    Tensor *m_pInput;
    Tensor *m_aOutput;
    Tensor *m_aWeight;

    // Training 과정을 공부한 후 다시 확인해야 할 부분
    Tensor *m_aGradient;
    Tensor *m_aDelta;
    Tensor *m_aDeltabar;

    // for Linked List
    // 만약 BackPropagate가 되는 그래프가 하나 더 만들어지게 되면
    Operator *NextOperator;
    Operator *PrevOperator;

    // identifier
    std::string *identifier;

    // 동적 할당 및 제거
    int  Alloc();
    void Delete();

public:
    Operator() {
        std::cout << "Operator::Operator()" << '\n';
        Alloc();
    }

    virtual ~Operator() {
        std::cout << "Operator::~Operator()" << '\n';
        Delete();
    }

    // Setter
    void SetInputDim();
    void SetOutputDim();

    // Input의 경우는 클래스 밖에서 접근되기에 setter를 두지 않습니다.
    void SetOutput();
    void SetWeight();

    void SetGradient();
    void SetDelta();
    void SetDeltabar();

    void SetNextOperator();
    // ~ Setter

    // Getter
    void GetInputDim() const;
    void GetOutputDim() const;

    void GetInput() const;
    void GetOutput() const;
    void GetWeight() const;

    void GetGradient() const;
    void GetDelta() const;
    void GetDeltabar() const;

    void GetPrevOperator() const;
    void GetNextOperator() const;
    // ~ Getter

    // Propagate
    bool PrePropagate(); // Propagate 진행 방향 및 순서
    virtual bool Propagate() = 0; // Execution of Propagate on Operator



    // BackPropagate
    bool PreBackPropagate(); // BackPropagate 진행 방향 및 순서
    virtual bool BackPropagate() = 0; // Execution of BackPropagate on Operator



};

#endif  // OPERATOR_H_
