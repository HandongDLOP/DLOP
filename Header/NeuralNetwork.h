#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <iostream>
#include <string>


#include "Placeholder.h"
#include "Variable.h"

#include "Relu.h"
#include "Sigmoid.h"
#include "Softmax.h"

#include "AddXB.h"
#include "MatMulXW.h"

#include "MSE.h"
#include "Softmax_Cross_Entropy.h"


enum RUNNINGOPTION {
    TRAINING,
    TESTING
};

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

    void PrintGraph(Operator * pStart = NULL, Operator * pEnd = NULL);

    void PrintData(Operator * pStart = NULL, Operator * pEnd = NULL);

    // ===========================================================================================
    bool CreateGraph(Optimizer *pOptimizer);

    // ===========================================================================================

    void UpdateVariable() {
        m_aOptimizer->UpdateVariable();
    }
};

#endif  // NEURALNETWORK_H_
