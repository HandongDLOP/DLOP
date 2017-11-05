#ifndef OPERATOR_H_
#define OPERATOR_H_

#include <iostream>
#include <string>

#include "Tensor.h"
#include "MetaParameter.h"

class Operator {
private:
    // N-dim 을 나타낼 수 있는 데이터 타입
    // 이 경우 Placeholder에서 정의하여 검사할 것 : 곧 Operato class에서 삭제 예정
    // Placeholder 는 queue의 형태가 될 것이라고 생각 됨
    TensorShape m_InputDim;
    TensorShape m_OutputDim;

    // Constructor에서 받는 input은 Operator이지만, 실제로 사용은 Tensor이다.
    Tensor *m_pInput;
    Tensor *m_aOutput;
    Tensor *m_aWeight;

    // Training 과정을 공부한 후 다시 확인해야 할 부분
    Tensor *m_aGradient;
    Tensor *m_aDelta;
    // Tensor *m_Deltabar; // Layer단에서 사용하게 되기에, 항상 필요하지는 않다.

    // for Linked List
    // Pointer array를 만들기 위한 공간으로 Alloc할 때 공간을 동적할당한다.
    Operator **m_aOutputOperator;
    Operator **m_aInputOperator;

    Operator *m_aInputTest;

    int m_OutputDegree = 0;
    int m_InputDegree  = 0;

    int m_currentOutputDegree = 0;
    int m_currentInputDegree  = 0;

    // identifier // 이제 Operator를 변수로 접근할 수 있게 되어 필요가 없다.
    std::string m_name;


    // 동적 할당 및 제거 (오퍼레이터마다 다르게 정의될 가능성이 큼, metaParameter가 다르기 때문에 )

public:
    Operator() {
        std::cout << "Operator::Operator() 상속자 상속상태" << '\n';
    }

    Operator(std::string pName) {
        std::cout << "Operator::Operator(std::string) 상속자 상속상태" << '\n';

        m_name = pName;
    }

    Operator(Operator *pInput) {
        std::cout << "Operator::Operator(Operator *) 상속자 상속상태" << '\n';
        // Operdtor 연결관계 Alloc
        _AddInputEdge(pInput);
        m_aInputOperator[m_InputDegree - 1]->_AddOutputEdge(this);
    }

    Operator(Operator *pInput, std::string pName) {
        std::cout << "Operator::Operator(Operator *, std::string) 상속자 상속상태" << '\n';
        // Operdtor 연결관계 Alloc
        _AddInputEdge(pInput);
        m_aInputOperator[m_InputDegree - 1]->_AddOutputEdge(this);

        m_name = pName;
    }

    Operator(Operator *pInput, MetaParameter *pParam) : Operator(pInput) {
        std::cout << "Operator::Operator(Operator *, MetaParameter *) 상속자 상속상태" << '\n';
    }

    Operator(Operator *pInput, MetaParameter *pParam, std::string pName) : Operator(pInput, pName) {
        std::cout << "Operator::Operator(Operator *, MetaParameter *, std::string) 상속자 상속상태" << '\n';
    }

    Operator(Operator *pInput, Operator *pWeight) : Operator(pInput) {
        std::cout << "Operator::Operator(Operator *, Operator *) 상속자 상속상태" << '\n';
    }

    Operator(Operator *pInput, Operator *pWeight, std::string pName) : Operator(pInput, pName) {
        std::cout << "Operator::Operator(Operator *, Operator *, std::string) 상속자 상속상태" << '\n';
    }

    virtual ~Operator() {
        std::cout << "Operator::~Operator()" << '\n';
        Delete();
    }

    // 추후 Private으로 옮길 의향 있음
    virtual bool Alloc(Operator *pInput, MetaParameter *pParam = NULL);
    virtual void Delete();

    bool         _AddInputEdge(Operator *pInput);
    bool         _AddOutputEdge(Operator *pOutput);

    //// Setter
    // void SetInputDim();
    // void SetOutputDim();
    //
    //// Input의 경우는 클래스 밖에서 접근되기에 setter를 두지 않습니다.
    // void SetOutput();
    // void SetWeight();
    //
    // void SetGradient();
    // void SetDelta();
    // void SetDeltabar();
    //
    // void SetNextOperator();
    //
    void IncreaseCurrentOutputDegree() {
        m_currentOutputDegree++;
    }

    void IncreaseCurrentInputDegree() {
        m_currentInputDegree++;
    }

    //
    //// ~ Setter
    //
    //// Getter (파생 클래스에서 사용합니다.)
    // void GetInputDim() const;
    // void GetOutputDim() const;
    //
    // void GetInput() const;
    // void GetOutput() const;
    // void GetWeight() const;
    //
    // void GetGradient() const;
    // void GetDelta() const;
    // void GetDeltabar() const;
    //
    // void GetInputOperator() const;
    // void GetOutputOperator() const;

    int GetOutputDgree() const {
        return m_OutputDegree;
    }

    int GetInputDgree() const {
        return m_InputDegree;
    }

    int GetCurrentOutputDgree() const {
        return m_currentOutputDegree;
    }

    int GetCurrentInputDgree() const {
        return m_currentInputDegree;
    }

    void GetInputOperator() const {
        for (int i = 0; i < m_InputDegree; i++) {
            std::cout << m_aInputOperator[i] << '\n';
        }
    }

    // void GetLayerType() const;
    //// ~ Getter

    // Propagate
    bool         ForwardPropagate(); // ForwardPropagate 진행 방향 및 순서
    virtual bool ComputeForwardPropagate();  // Compute to (추후 interface 형태로 수정 예정)


    // BackPropagate
    bool         BackPropagate(); // BackPropagate 진행 방향 및 순서
    virtual bool ComputeBackPropagate();  // compute delta and detabar(if we need to) (추후 interface 형태로 수정 예정)


    //// UpdateWeight
    // bool UpdateWeight();
};

#endif  // OPERATOR_H_
