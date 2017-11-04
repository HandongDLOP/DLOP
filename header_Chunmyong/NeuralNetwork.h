#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <iostream>

#include "Shape.h"
#include "Tensor.h"
#include "Operator.h"
#include "Objective.h"
#include "MetaParameter.h"
#include "Placeholder.h"

class NeuralNetwork {
private:
    // Operator 개수
    // 추후에는 noOperator를 정하지 않아도 되는 방법을 알아보고자 합니다.

    // 그래프 형식으로 바꿔야 합니다.
    // 그래프가 되기 위해서는 다음 오퍼레이터의 링크를 건네는 Operator가 필요합니다.
    Operator * _m_aStart = NULL;    // (Default)
    Operator * _m_aEnd = NULL;      // (Default)

    // Placeholder Manager
    // Basket이 될 가능성이 있음
    placeholder ** _Placeholder_manager = NULL;

    bool Alloc();
    void Delete();

public:
    // Operator의 개수를 정합니다.
    NeuralNetwork();
    virtual ~NeuralNetwork();

    // Placeholder 추가
    Operator* Placeholder(std::string name);

    // Propagate
    // Prameter에 basket이 추가될 수 있음
    bool ForwardPropagate(Operator *_pStart, Operator *_pEnd);

    // BackPropagate
    bool BackPropagate(Operator *_pStart, Operator *_pEnd);

    // For NeuralNetwork Training
    bool Training(Operator *_pStart = NULL, Operator *_pEnd = NULL);
    bool Testing(Operator *_pStart = NULL, Operator *_pEnd = NULL);

};

#endif  // NEURALNETWORK_H_
