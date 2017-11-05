#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <iostream>
#include <string>

#include "Operator.h"
#include "Objective.h"

#include "Placeholder.h"
#include "Convolution.h"
#include "Variable.h"
#include "Relu.h"
#include "Maxpooling.h"
#include "Copy.h"

class NeuralNetwork {
private:
    // Operator 개수
    // 추후에는 noOperator를 정하지 않아도 되는 방법을 알아보고자 합니다.

    // Placeholder Manager
    // Operator * m_aBaseOperator = new Operator("Base_Operator");

    // 그래프 형식으로 바꿔야 합니다.
    // 그래프가 되기 위해서는 다음 오퍼레이터의 링크를 건네는 Operator가 필요합니다.
    Operator * _m_aStart = NULL;    // (Default)
    Operator * _m_aEnd = NULL;      // (Default)

    bool Alloc();
    void Delete();

public:
    // Operator의 개수를 정합니다.
    NeuralNetwork();
    virtual ~NeuralNetwork();

    // Placeholder 추가
    Operator* AddPlaceholder();
    Operator* AddPlaceholder(std::string pName);

    // Propagate
    // Prameter에 basket이 추가될 수 있음
    bool ForwardPropagate(Operator *_pStart, Operator *_pEnd);
    bool BackPropagate(Operator *_pStart, Operator *_pEnd);

    // For NeuralNetwork Training
    bool Training(Operator *_pStart = NULL, Operator *_pEnd = NULL);
    bool Testing(Operator *_pStart = NULL, Operator *_pEnd = NULL);

};

#endif  // NEURALNETWORK_H_
