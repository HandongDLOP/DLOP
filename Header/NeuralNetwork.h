#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Operator//Placeholder.h"
#include "Operator//Variable.h"

#include "Operator//Relu.h"
#include "Operator//Sigmoid.h"
// #include "Operator//Softmax.h"

#include "Operator//AddXB.h"
#include "Operator//MatMulXW.h"

#include "Objective//MSE.h"
// #include "Objective//CrossEntropy.h"
#include "Objective//SoftmaxCrossEntropy.h"

template<typename DTYPE>
class NeuralNetwork {
private:
    // Operator 개수
    // 추후에는 noOperator를 정하지 않아도 되는 방법을 알아보고자 합니다.

    // Placeholder Manager
    // Operator<DTYPE> * m_aBaseOperator = new Operator("Base_Operator");

    // 그래프 형식으로 바꿔야 합니다.
    // 그래프가 되기 위해서는 다음 오퍼레이터의 링크를 건네는 Operator가 필요합니다.
    Operator<DTYPE> *m_aStart = new Operator<DTYPE>("Base Operator");  // (Default)
    // Operator<DTYPE> *m_aEnd   = new Operator("Final Operator");     // (Default)

    // Optimizer<DTYPE> *m_pOptimizer = NULL;

public:
    // Operator의 개수를 정합니다.
    NeuralNetwork() {
        std::cout << "NeuralNetwork<DTYPE>::NeuralNetwork()" << '\n';
        this->Alloc();
    }

    virtual ~NeuralNetwork() {
        this->Delete();
        std::cout << "NeuralNetwork<DTYPE>::~NeuralNetwork()" << '\n';
    }

    // ===========================================================================================

    // 추후 private로 옮길 의향 있음
    int Alloc();
    int AllocOptimizer(Optimizer<DTYPE> *pOptimizer);

    void Delete();
    int DeletePlaceholder();
    // int DeleteOperator();

    // ===========================================================================================

    // Placeholder 추가 // 추후 이렇게 하지 않아도 연결할 수 있는 방법 찾기
    // Operator<DTYPE>* AddPlaceholder(TensorShape *pshape);
    // Operator<DTYPE>* AddPlaceholder(TensorShape *pshape, std::string pName);
    Operator<DTYPE>* AddPlaceholder(Tensor<DTYPE> *pTensor, std::string pName);

    // Propagate
    // Prameter에 basket이 추가될 수 있음
    int ForwardPropagate(Operator<DTYPE> *pStart, Operator<DTYPE> *pEnd);
    int BackPropagate(Operator<DTYPE> *pStart, Optimizer<DTYPE> *pOptimizer);

    //// For NeuralNetwork Training
    // int Training(Operator<DTYPE> *pStart, Operator<DTYPE> *pEnd);
    // int Testing(Operator<DTYPE> *pStart, Operator<DTYPE> *pEnd);
    //
    // int Training(Operator<DTYPE> *pEnd);
    // int Testing(Operator<DTYPE> *pEnd);
    //
    // int Training(Optimizer<DTYPE> *pOptimizer);
    // int Testing(Optimizer<DTYPE> *pOptimizer);

    int Run(Operator<DTYPE> *pStart, Operator<DTYPE> *pEnd);
    int Run(Operator<DTYPE> *pEnd);
    int Run(Optimizer<DTYPE> *pOptimizer);


    // void SetEndOperator(Operator<DTYPE> *pEnd) {
    // m_aEnd->AddEdgebetweenOperators(pEnd);
    // }

    // void SetOptimizer(Optimizer<DTYPE> *pOptimizer) {
    // m_pOptimizer = pOptimizer;
    // }

    void PrintGraph(Operator<DTYPE> *pEnd);
    void PrintGraph(Optimizer<DTYPE> *pOptimizer);

    // 추후에는 그래프에 있는 Operator인지도 확인해야 한다.
    void PrintData(Optimizer<DTYPE> *pOptimizer, int forceprint = 0);

    void PrintData(Operator<DTYPE> *pOperator, int forceprint = 0);

    // ===========================================================================================
    int CreateGraph(Optimizer<DTYPE> *pOptimizer);

    // ===========================================================================================

    void UpdateVariable(Optimizer<DTYPE> *pOptimizer) {
        pOptimizer->UpdateVariable();
    }

    Operator<DTYPE>* GetBaseOperator() {
        return m_aStart;
    }

    // Operator<DTYPE> * GetFinalOperator(){
    // return m_aEnd;
    // }
};

#endif  // NEURALNETWORK_H_
