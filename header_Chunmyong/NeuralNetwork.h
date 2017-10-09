#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <iostream>
#include <String>

#include "Shape.h"
#include "Ark.h"
#include "Operator.h"
#include "Objective.h"
#include "MetaParameter.h"

class NeuralNetwork {
private:
    // Operator 개수
    // 추후에는 noOperator를 정하지 않아도 되는 방법을 알아보고자 합니다.

    // 그래프 형식으로 바꿔야 합니다.
    // 그래프가 되기 위해서는 다음 오퍼레이터의 링크를 건네는 Operator가 필요합니다.
    Operator *m_aOperator;

    // 각 Operator 객체를 저장할 array를 만듭니다.
    bool Alloc();
    void Delete();

public:
    // Operator의 개수를 정합니다.
    NeuralNetwork();
    virtual ~NeuralNetwork();

    // 추후 Op 파라미터 enum 형식으로 바꿀 계획에 있음
    bool PutOperator(std::string Op, MetaParameter *pParam, LayerType = HIDDEN);

    // Propagate
    bool ForwardPropagate();

    // BackPropagate
    bool BackPropagate();

    // For NeuralNetwork Training
    bool Training(const int p_maxEpoch);
    bool Testing();

};

#endif  // NEURALNETWORK_H_
