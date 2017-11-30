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
#include "Objective//Cross_Entropy.h"
#include "Objective//Softmax_Cross_Entropy.h"



class NeuralNetwork {
private:
    // Operator 개수
    // 추후에는 noOperator를 정하지 않아도 되는 방법을 알아보고자 합니다.

    // Placeholder Manager
    // Operator * m_aBaseOperator = new Operator("Base_Operator");

    // 그래프 형식으로 바꿔야 합니다.
    // 그래프가 되기 위해서는 다음 오퍼레이터의 링크를 건네는 Operator가 필요합니다.
    Operator *m_pStart = new Operator("Base Operator");  // (Default)
    Operator *m_aEnd   = new Operator("Final Operator");     // (Default)

    Optimizer *m_aOptimizer = NULL;

public:
    // Operator의 개수를 정합니다.
    NeuralNetwork();
    virtual ~NeuralNetwork();

    // ===========================================================================================

    // 추후 private로 옮길 의향 있음
    bool Alloc();
    bool AllocOptimizer(Optimizer *pOptimizer);

    void Delete();
    bool DeleteOperator();

    // ===========================================================================================

    // Placeholder 추가 // 추후 이렇게 하지 않아도 연결할 수 있는 방법 찾기
    // Operator* AddPlaceholder(TensorShape *pshape);
    // Operator* AddPlaceholder(TensorShape *pshape, std::string pName);
    Operator* AddPlaceholder(Tensor *pTensor, std::string pName);

    // Propagate
    // Prameter에 basket이 추가될 수 있음
    bool ForwardPropagate(Operator *pStart, Operator *pEnd);
    bool BackPropagate(Operator *pStart, Operator *pEnd);

    // For NeuralNetwork Training
    bool Training(Operator *pStart = NULL, Operator *pEnd = NULL);
    bool Testing(Operator *pStart = NULL, Operator *pEnd = NULL);

    void SetEndOperator(Operator *pEnd) {
        m_aEnd->AddEdgebetweenOperators(pEnd);
    }

    void SetOptimizer(Optimizer *pOptimizer) {
        m_aOptimizer = pOptimizer;
    }

    void PrintGraph(Operator *pStart = NULL, Operator *pEnd = NULL);

    // 추후에는 그래프에 있는 Operator인지도 확인해야 한다.
    void PrintData(int forceprint = 0);

    void PrintData(Operator *pOperator, int forceprint = 0);

    void PrintData(Operator *pStart, Operator *pEnd, int forceprint = 0);

    // ===========================================================================================
    bool CreateGraph(Optimizer *pOptimizer);

    // ===========================================================================================

    void UpdateVariable() {
        m_aOptimizer->UpdateVariable();
    }
};

#endif  // NEURALNETWORK_H_
