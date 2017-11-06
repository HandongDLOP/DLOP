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
    TensorShape *m_pInputDim  = NULL;
    TensorShape *m_pOutputDim = NULL;

    // Constructor에서 받는 input은 Operator이지만, 실제로 사용은 Tensor이다.
    Tensor *m_pInput  = NULL;
    Tensor *m_aOutput = NULL;
    Tensor *m_aWeight = NULL;

    // Training 과정을 공부한 후 다시 확인해야 할 부분
    Tensor *m_aGradient = NULL;
    Tensor *m_aDelta    = NULL;
    // Tensor *m_Deltabar; // Layer단에서 사용하게 되기에, 항상 필요하지는 않다.

    // for Linked List
    // Pointer array를 만들기 위한 공간으로 Alloc할 때 공간을 동적할당한다.
    Operator **m_aOutputOperator = NULL;
    Operator **m_aInputOperator  = NULL;

    int m_OutputDegree = 0;
    int m_InputDegree  = 0;

    int m_currentOutputDegree = 0;
    int m_currentInputDegree  = 0;

    // identifier // 이제 Operator를 변수로 접근할 수 있게 되어 필요가 없다.
    std::string m_name = "NO NAME";


    // 동적 할당 및 제거 (오퍼레이터마다 다르게 정의될 가능성이 큼, metaParameter가 다르기 때문에 )

public:
    Operator() {
        std::cout << "Operator::Operator() 상속자 상속상태" << '\n';
    }

    Operator(std::string pName) {
        std::cout << "Operator::Operator(std::string) 상속자 상속상태" << '\n';
        m_name = pName;
    }

    // ===========================================================================================

    Operator(Tensor *pTensor) {
        std::cout << "Operator::Operator(Tensor *, std::string pName) 상속자 상속상태" << '\n';
        Alloc(pTensor);
    }

    Operator(Tensor *pTensor, std::string pName) : Operator(pName) {
        std::cout << "Operator::Operator(Tensor *, std::string pName) 상속자 상속상태" << '\n';
        Alloc(pTensor);
    }

    // ===========================================================================================

    Operator(Operator *pInput) {
        std::cout << "Operator::Operator(Operator *) 상속자 상속상태" << '\n';
        Alloc(pInput);
    }

    Operator(Operator *pInput, std::string pName) : Operator(pName) {
        std::cout << "Operator::Operator(Operator *, std::string) 상속자 상속상태" << '\n';
        Alloc(pInput);
    }

    // ===========================================================================================

    Operator(Operator *pInput, MetaParameter *pParam) : Operator(pInput) {
        std::cout << "Operator::Operator(Operator *, MetaParameter *) 상속자 상속상태" << '\n';
    }

    Operator(Operator *pInput, MetaParameter *pParam, std::string pName) : Operator(pInput, pName) {
        std::cout << "Operator::Operator(Operator *, MetaParameter *, std::string) 상속자 상속상태" << '\n';
    }

    // ===========================================================================================

    Operator(Operator *pInput1, Operator *pInput2) : Operator(pInput1) {
        std::cout << "Operator::Operator(Operator *, Operator *) 상속자 상속상태" << '\n';
    }

    Operator(Operator *pInput1, Operator *pInput2, std::string pName) : Operator(pInput1, pName) {
        std::cout << "Operator::Operator(Operator *, Operator *, std::string) 상속자 상속상태" << '\n';
    }

    // ===========================================================================================

    virtual ~Operator() {
        std::cout << "Operator::~Operator()" << '\n';
        // parent class Delete
        Delete();
        // 자식 클래스에서는 새롭게 만들어지는 MetaParameter에 관한 delete를 만들어주어야 한다
    }

    // ===========================================================================================

    // 추후 Private으로 옮길 의향 있음
    bool         Alloc(Tensor *pTensor);
    bool         Alloc(Operator *pInput);
    virtual bool Alloc(MetaParameter *pParam = NULL);

    virtual void Delete();
    bool         PropagateDelete();

    // ===========================================================================================

    bool _AddInputEdge(Operator *pInput);
    bool _AddOutputEdge(Operator *pOutput);

    // ===========================================================================================

    //// Setter
    // void SetInputDim();
    // void SetOutputDim();
    //
    //// Input의 경우는 클래스 밖에서 접근되기에 setter를 두지 않습니다.
    void SetOutput(Tensor *pTensor) {
        // shllow copy
        m_aOutput->Setshape(pTensor->Getshape());
        m_aOutput->SetData(pTensor->GetData());
        m_aOutput->SetFlatDim(pTensor->GetFlatDim());
    }

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

    // ===========================================================================================

    //
    //// Getter (파생 클래스에서 사용합니다.)
    // void GetInputDim() const;
    TensorShape* GetOutputDim() const {
        return m_pOutputDim;
    }

    Tensor* GetInput() const {
        return m_pInput;
    }

    Tensor* GetOutput() const {
        return m_aOutput;
    }

    // void GetWeight() const;
    //
    // void GetGradient() const;
    // void GetDelta() const;
    // void GetDeltabar() const;
    //
    Operator** GetInputOperator() const {
        return m_aInputOperator;
    }

    Operator** GetOutputOperator() const {
        return m_aOutputOperator;
    }

    int GetOutputDegree() const {
        return m_OutputDegree;
    }

    int GetInputDegree() const {
        return m_InputDegree;
    }

    int GetCurrentOutputDegree() const {
        return m_currentOutputDegree;
    }

    int GetCurrentInputDegree() const {
        return m_currentInputDegree;
    }

    std::string GetName() {
        return m_name;
    }

    // ===========================================================================================

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
