#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Operator//Placeholder.h"
#include "Operator//Variable.h"

#include "Operator//Relu.h"
#include "Operator//Sigmoid.h"
#include "Operator//Softmax.h"

#include "Operator//AddXB.h"
#include "Operator//MatMulXW.h"

#include "Objective//MSE.h"
#include "Objective//CrossEntropy.h"
#include "Objective//SoftmaxCrossEntropy.h"

class NeuralNetwork {
private:
    // Operator 개수
    // 추후에는 noOperator를 정하지 않아도 되는 방법을 알아보고자 합니다.

    // Placeholder Manager
    // Operator * m_aBaseOperator = new Operator("Base_Operator");

    // 그래프 형식으로 바꿔야 합니다.
    // 그래프가 되기 위해서는 다음 오퍼레이터의 링크를 건네는 Operator가 필요합니다.
    Operator *m_aStart = new Operator("Base Operator");  // (Default)
    // Operator *m_aEnd   = new Operator("Final Operator");     // (Default)

    // Optimizer *m_pOptimizer = NULL;

public:
    // Operator의 개수를 정합니다.
    NeuralNetwork();
    virtual ~NeuralNetwork();

    // ===========================================================================================

    // 추후 private로 옮길 의향 있음
    bool Alloc();
    bool AllocOptimizer(Optimizer *pOptimizer);

    void Delete();
    bool DeletePlaceholder();
    // bool DeleteOperator();

    // ===========================================================================================

    // Placeholder 추가 // 추후 이렇게 하지 않아도 연결할 수 있는 방법 찾기
    // Operator* AddPlaceholder(TensorShape *pshape);
    // Operator* AddPlaceholder(TensorShape *pshape, std::string pName);
    Operator* AddPlaceholder(Tensor *pTensor, std::string pName);

    // Propagate
    // Prameter에 basket이 추가될 수 있음
    bool ForwardPropagate(Operator *pStart, Operator *pEnd);
    bool BackPropagate(Operator *pStart, Optimizer *pOptimizer);

    //// For NeuralNetwork Training
    // bool Training(Operator *pStart, Operator *pEnd);
    // bool Testing(Operator *pStart, Operator *pEnd);
    //
    // bool Training(Operator *pEnd);
    // bool Testing(Operator *pEnd);
    //
    // bool Training(Optimizer *pOptimizer);
    // bool Testing(Optimizer *pOptimizer);

    bool Run(Operator *pStart, Operator *pEnd);
    bool Run(Operator *pEnd);
    bool Run(Optimizer *pOptimizer);


    // void SetEndOperator(Operator *pEnd) {
    // m_aEnd->AddEdgebetweenOperators(pEnd);
    // }

    // void SetOptimizer(Optimizer *pOptimizer) {
    // m_pOptimizer = pOptimizer;
    // }

    void PrintGraph(Operator *pEnd);
    void PrintGraph(Optimizer *pOptimizer);

    // 추후에는 그래프에 있는 Operator인지도 확인해야 한다.
    void PrintData(Optimizer *pOptimizer, int forceprint = 0);

    void PrintData(Operator *pOperator, int forceprint = 0);

    // ===========================================================================================
    bool CreateGraph(Optimizer *pOptimizer);

    // ===========================================================================================

    void UpdateVariable(Optimizer *pOptimizer) {
        pOptimizer->UpdateVariable();
    }

    Operator* GetBaseOperator() {
        return m_aStart;
    }

    // Operator * GetFinalOperator(){
    // return m_aEnd;
    // }
};

#endif  // NEURALNETWORK_H_
