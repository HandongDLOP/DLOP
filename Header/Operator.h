#ifndef OPERATOR_H_
#define OPERATOR_H_

#include <iostream>
#include <string>
#include <algorithm>

// #include "Tensor.h"
#include "MetaParameter.h"
#include "Factory.h"

class Operator {
private:

    Tensor *m_aOutput = NULL;

    // Training 과정을 공부한 후 다시 확인해야 할 부분
    // Gradient의 경우는 자신의 Output Operator에서 계산해서 이미 넘겨준 상태 (계산 과정 잘 생각해보기)
    Tensor *m_aGradient = NULL;
    Tensor *m_aDelta    = NULL;

    // for Linked List
    // Pointer array를 만들기 위한 공간으로 Alloc할 때 공간을 동적할당한다.
    Operator **m_apOutputOperator = NULL;
    Operator **m_apInputOperator  = NULL;

    int m_OutputDegree = 0;
    int m_InputDegree  = 0;

    int m_currentOutputDegree = 0;
    int m_currentInputDegree  = 0;

    // for Optimizer
    // 추후 삭제 예정
    Optimizer *m_aOptimizer = NULL;

    // identifier
    std::string m_name = "NO NAME";

    // Private Operator

private:

    bool _AddInputEdge(Operator *pInput);
    bool _AddOutputEdge(Operator *pOutput);

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

    Operator(TensorShape *pshape) {
        std::cout << "Operator::Operator(TensorShape *) 상속자 상속상태" << '\n';
        Alloc(pshape);
    }

    Operator(TensorShape *pshape, std::string pName) : Operator(pName) {
        std::cout << "Operator::Operator(TensorShape *, std::string) 상속자 상속상태" << '\n';
        Alloc(pshape);
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

    Operator(Operator *pInput, MetaParameter *pParam) {
        std::cout << "Operator::Operator(Operator *, MetaParameter *) 상속자 상속상태" << '\n';
        Alloc(pInput);
    }

    Operator(Operator *pInput, MetaParameter *pParam, std::string pName) : Operator(pName) {
        std::cout << "Operator::Operator(Operator *, MetaParameter *, std::string) 상속자 상속상태" << '\n';
        Alloc(pInput);
    }

    // ===========================================================================================

    Operator(Operator *pInput1, Operator *pInput2) {
        std::cout << "Operator::Operator(Operator *, Operator *) 상속자 상속상태" << '\n';
        Alloc(pInput1, pInput2);
    }

    Operator(Operator *pInput1, Operator *pInput2, std::string pName) : Operator(pName) {
        std::cout << "Operator::Operator(Operator *, Operator *, std::string) 상속자 상속상태" << '\n';
        Alloc(pInput1, pInput2);
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
    virtual bool Alloc(Tensor *pTensor);
    virtual bool Alloc(TensorShape *pshape);
    virtual bool Alloc(Operator *pInput);
    virtual bool Alloc(Operator *pInput1,
                       Operator *pInput2);
    virtual bool Alloc(MetaParameter *pParam = NULL);
    bool         AllocOptimizer(Optimizer_name pOptimizer_name);

    virtual void Delete();
    bool         DeleteInputOperator();

    // ===========================================================================================

    bool AddEdgebetweenOperators(Operator *pInput);

    // ===========================================================================================

    //// Setter
    // Gradient 부분은 Trainable한 부분에서만 만들기에 NULL로 초기화할 가능성이 생길 것으로 보인다.
    void SetGradient(Tensor *pTensor) {
        m_aGradient = pTensor;
    }

    void SetGradient(float *pData) {
        m_aGradient->SetData(pData);
    }

    void SetDelta(Tensor *pTensor) {
        m_aDelta = pTensor;
    }

    void SetDelta(float *pData) {
        m_aDelta->SetData(pData);
    }

    void SetOptimizer(Optimizer *pOptimizer) {
        m_aOptimizer = pOptimizer;
    }

    // ===========================================================================================

    void IncreaseCurrentOutputDegree() {
        m_currentOutputDegree++;
    }

    void IncreaseCurrentInputDegree() {
        m_currentInputDegree++;
    }

    // ===========================================================================================

    //
    //// Getter (파생 클래스에서 사용합니다.)

    // void GetWeight() const;
    //
    Tensor* GetGradient() const {
        return m_aGradient;
    }

    Tensor* GetDelta() const {
        return m_aDelta;
    }

    // void GetDeltabar() const;
    //
    Operator** GetInputOperator() const {
        return m_apInputOperator;
    }

    Operator** GetOutputOperator() const {
        return m_apOutputOperator;
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

    Optimizer* GetOptimizer() {
        return m_aOptimizer;
    }

    std::string GetName() {
        return m_name;
    }

    // ===========================================================================================

    // Propagate
    bool         ForwardPropagate();        // ForwardPropagate 진행 방향 및 순서
    virtual bool ComputeForwardPropagate(); // Compute to (추후 interface 형태로 수정 예정)

    // BackPropagate
    bool         BackPropagate();           // BackPropagate 진행 방향 및 순서
    virtual bool ComputeBackPropagate();    // compute delta and detabar(if we need to) (추후 interface 형태로 수정 예정)


    // ===========================================================================================

    Operator* CheckEndOperator();

    //// UpdateWeight
    // bool UpdateWeight();
};

#endif // OPERATOR_H_
