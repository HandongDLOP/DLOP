#ifndef OPERATOR_H_
#define OPERATOR_H_

#include "MetaParameter.h"
#include "Optimizer//GradientDescentOptimizer.h"

template<typename DTYPE>
class Operator {
public:
    typedef typename Tensor<DTYPE>::TENSOR_DTYPE TENSOR_DTYPE;
private:
    Tensor<DTYPE> *m_aOutput = NULL;

    // Training 과정을 공부한 후 다시 확인해야 할 부분
    // Gradient의 경우는 자신의 Output Operator에서 계산해서 이미 넘겨준 상태 (계산 과정 잘 생각해보기)
    Tensor<DTYPE> *m_aGradient = NULL;
    Tensor<DTYPE> *m_aDelta    = NULL;

    // for Linked List
    // Pointer array를 만들기 위한 공간으로 Alloc할 때 공간을 동적할당한다.
    Operator<DTYPE> **m_apOutputOperator = NULL;
    Operator<DTYPE> **m_apInputOperator  = NULL;

    int m_OutputDegree = 0;
    int m_InputDegree  = 0;

    int m_currentOutputDegree = 0;
    int m_currentInputDegree  = 0;

    // identifier
    std::string m_name = "NO NAME";

    int m_Trainable = 0;
    // Private Operator

private:
    bool _AddInputEdge(Operator<DTYPE> *pInput);
    bool _AddOutputEdge(Operator<DTYPE> *pOutput);

public:
    Operator() {
        std::cout << "Operator<DTYPE>::Operator()" << '\n';
    }

    Operator(std::string pName) {
        std::cout << "Operator<DTYPE>::Operator(std::string)" << '\n';
        m_name = pName;
    }

    // ===========================================================================================

    Operator(Tensor<DTYPE> *pTensor) {
        std::cout << "Operator<DTYPE>::Operator(Tensor<DTYPE> *, std::string pName)" << '\n';
        this->Alloc(pTensor);
    }

    Operator(Tensor<DTYPE> *pTensor, std::string pName) : Operator(pName) {
        std::cout << "Operator<DTYPE>::Operator(Tensor<DTYPE> *, std::string pName)" << '\n';
        this->Alloc(pTensor);
    }

    // ===========================================================================================

    Operator(Operator<DTYPE> *pInput) {
        std::cout << "Operator<DTYPE>::Operator(Operator<DTYPE> *)" << '\n';
        this->Alloc(pInput);
    }

    Operator(Operator<DTYPE> *pInput, std::string pName) : Operator(pName) {
        std::cout << "Operator<DTYPE>::Operator(Operator<DTYPE> *, std::string)" << '\n';
        this->Alloc(pInput);
    }

    // ===========================================================================================

    Operator(Operator<DTYPE> *pInput, MetaParameter<DTYPE> *pParam) {
        std::cout << "Operator<DTYPE>::Operator(Operator<DTYPE> *, MetaParameter<DTYPE> *)" << '\n';
        this->Alloc(pInput);
    }

    Operator(Operator<DTYPE> *pInput, MetaParameter<DTYPE> *pParam, std::string pName) : Operator(pName) {
        std::cout << "Operator<DTYPE>::Operator(Operator<DTYPE> *, MetaParameter<DTYPE> *, std::string)" << '\n';
        this->Alloc(pInput);
    }

    // ===========================================================================================

    Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1) {
        std::cout << "Operator<DTYPE>::Operator(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        this->Alloc(pInput0, pInput1);
    }

    Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName) : Operator(pName) {
        std::cout << "Operator<DTYPE>::Operator(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        this->Alloc(pInput0, pInput1);
    }

    // ===========================================================================================

    virtual ~Operator() {
        std::cout << "Operator<DTYPE>::~Operator()" << '\n';

        // parent class Delete
        this->Delete();

        // 자식 클래스에서는 새롭게 만들어지는 MetaParameter에 관한 delete를 만들어주어야 한다
    }

    // ===========================================================================================

    // 추후 Private으로 옮길 의향 있음
    virtual bool Alloc(Tensor<DTYPE> *pTensor);
    virtual bool Alloc(Operator<DTYPE> *pInput);
    virtual bool Alloc(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1);
    virtual bool Alloc(MetaParameter<DTYPE> *pParam = NULL);

    bool         AllocOptimizer(Optimizer<DTYPE> *pOptimizer);

    virtual void Delete();

    // bool         DeleteInputOperator();

    // ===========================================================================================

    bool AddEdgebetweenOperators(Operator<DTYPE> *pInput);

    // ===========================================================================================

    //// Setter
    // Gradient 부분은 Trainable한 부분에서만 만들기에 NULL로 초기화할 가능성이 생길 것으로 보인다.
    void SetOutput(Tensor<DTYPE> *pTensor) {
        m_aOutput = pTensor;
    }

    void FeedOutput(Tensor<DTYPE> *pTensor) {
        m_aOutput = pTensor;
    }

    void SetGradient(Tensor<DTYPE> *pTensor) {
        m_aGradient = pTensor;
    }

    void SetDelta(Tensor<DTYPE> *pTensor) {
        m_aDelta = pTensor;
    }

    void SetTrainable(int pTrainable) {
        m_Trainable = pTrainable;
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

    Tensor<DTYPE>* GetOutput() const {
        return m_aOutput;
    }

    Tensor<DTYPE>* GetGradient() const {
        return m_aGradient;
    }

    Tensor<DTYPE>* GetDelta() const {
        return m_aDelta;
    }

    Operator<DTYPE>** GetInputOperator() const {
        return m_apInputOperator;
    }

    Operator<DTYPE>** GetOutputOperator() const {
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

    std::string GetName() const{
        return m_name;
    }

    int GetTrainable() const{
        return m_Trainable;
    }

    // ===========================================================================================

    // Propagate
    bool         ForwardPropagate();        // ForwardPropagate 진행 방향 및 순서
    virtual bool ComputeForwardPropagate();  // Compute to (추후 interface 형태로 수정 예정)

    // BackPropagate
    bool         BackPropagate(); // BackPropagate 진행 방향 및 순서
    virtual bool ComputeBackPropagate();  // compute delta and detabar(if we need to) (추후 interface 형태로 수정 예정)


    // ===========================================================================================

    void PrintGraph(int depth = 0);

    void PrintData(int forceprint = 0);

    void PrintOutput(int forceprint = 0) {
        std::cout << this->GetName() << ": Shape of Output" << '\n';
        this->GetOutput()->PrintShape();
        std::cout << this->GetName() << ": Value of Output" << '\n';
        this->GetOutput()->PrintData(forceprint);
    }

    void PrintDelta(int forceprint = 0) {
        std::cout << this->GetName() << ": Shape of Delta" << '\n';
        this->GetDelta()->PrintShape();
        std::cout << this->GetName() << ": Value of Delta" << '\n';
        this->GetDelta()->PrintData(forceprint);
    }

    void PrintGradient(int forceprint = 0) {
        std::cout << this->GetName() << ": Shape of Gradient" << '\n';
        this->GetGradient()->PrintShape();
        std::cout << this->GetName() << ": Value of Gradient" << '\n';
        this->GetGradient()->PrintData(forceprint);
    }

    // ===========================================================================================

    // Operator<DTYPE>* CheckEndOperator();
};

#endif  // OPERATOR_H_
