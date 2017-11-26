#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <iostream>
#include <string>


#include "Placeholder.h"
#include "Variable.h"

#include "Relu.h"
#include "Sigmoid.h"

#include "Add.h"
#include "MatMulXW.h"

#include "MSE.h"


class NeuralNetwork {
private:
    // Operator 개수
    // 추후에는 noOperator를 정하지 않아도 되는 방법을 알아보고자 합니다.

    // Placeholder Manager
    // Operator * m_aBaseOperator = new Operator("Base_Operator");

    // 그래프 형식으로 바꿔야 합니다.
    // 그래프가 되기 위해서는 다음 오퍼레이터의 링크를 건네는 Operator가 필요합니다.
    Operator *_m_pStart = new Operator("Base Operator");  // (Default)
    Operator *_m_aEnd   = _m_pStart;     // (Default)

public:
    // Operator의 개수를 정합니다.
    NeuralNetwork();
    virtual ~NeuralNetwork();

    // ===========================================================================================

    // 추후 private로 옮길 의향 있음
    bool Alloc();
    bool AllocOptimizer(Optimizer_name pOptimizer_name);

    void Delete();
    bool DeleteOperator();

    // ===========================================================================================

    // Placeholder 추가 // 추후 이렇게 하지 않아도 연결할 수 있는 방법 찾기
    // Operator* AddPlaceholder(TensorShape *pshape);
    // Operator* AddPlaceholder(TensorShape *pshape, std::string pName);
    Operator* AddPlaceholder(Tensor *pTensor, std::string pName);

    // Propagate
    // Prameter에 basket이 추가될 수 있음
    bool ForwardPropagate(Operator *_pStart, Operator *_pEnd);
    bool BackPropagate(Operator *_pStart, Operator *_pEnd);

    // For NeuralNetwork Training
    bool Training(Operator *_pStart = NULL, Operator *_pEnd = NULL);
    bool Testing(Operator *_pStart = NULL, Operator *_pEnd = NULL);

    // Set _m_aEnd : 추후에는 모델이 만들어질 때 자동으로 alloc되게 변환해야 함  // 임시 함수

    // ===========================================================================================
    bool CreateGraph(Optimizer_name pOptimizer_name, Operator *pEnd);
    bool CreateGraph(Optimizer_name pOptimizer_name);

    void SetEndOperator(Operator *pEnd) {
        _m_aEnd = pEnd;
    }
    bool SetEndOperator();

};

#endif  // NEURALNETWORK_H_
