#ifndef OPERATOR_H_
#define OPERATOR_H_

#include <iostream>
#include <string>

#include "Shape.h"
#include "Manna.h"
#include "MetaParameter.h"

// enum class를 사용하는 방안도 고려 중에 있음
typedef enum LayerType {
    INPUT,
    HIDDEN,
    OUTPUT
} LayerType;

class Operator {
private:
    // N-dim 을 나타낼 수 있는 데이터 타입
    // 이 경우 Placeholder에서 정의하여 검사할 것 : 곧 Operato class에서 삭제 예정
    // Placeholder 는 queue의 형태가 될 것이라고 생각 됨
    Shape m_InputDim;
    Shape m_OutputDim;

    Manna *m_pInput;
    Manna *m_aOutput;
    Manna *m_aWeight;

    // Training 과정을 공부한 후 다시 확인해야 할 부분
    Manna *m_aGradient;
    Manna *m_aDelta;
    // Manna *m_Deltabar; // Layer단에서 사용하게 되기에, 항상 필요하지는 않다.

    // for Linked List
    // Pointer array를 만들기 위한 공간으로 Alloc할 때 공간을 동적할당한다.
    Operator **m_aInputOperator;
    Operator **m_pOutputOperator;

    int m_OutputDgree = 0;
    int m_InputDegree = 0;

    // identifier
    std::string m_name;

    // Layer Type : (default) HIDDEN
    LayerType m_LayerType = HIDDEN;

    // 동적 할당 및 제거 (오퍼레이터마다 다르게 정의될 가능성이 큼, metaParameter가 다르기 때문에 )
    virtual bool Alloc(Manna *pInput, MetaParameter *pParam);
    virtual void Delete();

public:
    Operator(Manna *pInput, MetaParameter *pParam, LayerType LayerType = HIDDEN) {
        std::cout << "Operator::Operator() 상속자 상속상태" << '\n';
        Alloc(pInput, pParam);
    }

    virtual ~Operator() {
        std::cout << "Operator::~Operator()" << '\n';

        Delete();
    }

    // 추후 Getter 경우는 enum 상수를 이용하여 받는 형식을 차용할 예정입니다.
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

    // Getter (파생 클래스에서 사용합니다.)
    void GetInputDim() const;
    void GetOutputDim() const;

    void GetInput() const;
    void GetOutput() const;
    void GetWeight() const;

    void GetGradient() const;
    void GetDelta() const;
    void GetDeltabar() const;

    void GetInputOperator() const;
    void GetOutputOperator() const;

    void GetOutputDgree() const;
    void GetInputDgree() const;

    void GetName() const;

    void GetLayerType() const;
    // ~ Getter

    // Propagate
    bool         ForwardPropagate(); // ForwardPropagate 진행 방향 및 순서
    virtual bool ComputeForwardPropagate();  // Compute to (추후 interface 형태로 수정 예정)


    // BackPropagate
    bool         BackPropagate(); // BackPropagate 진행 방향 및 순서
    virtual bool ComputeBackPropagate();  // compute delta and detabar(if we need to) (추후 interface 형태로 수정 예정)


    // UpdateWeight
    bool UpdateWeight();
};

#endif  // OPERATOR_H_
